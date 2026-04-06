"""Live BTC/USDT chart with selectable indicators.

Fetches 3 days of 1m history from data.binance.vision, then streams
live 1m klines via Binance WebSocket. Indicators are toggled in the browser.
Only the last --view-hours of data is rendered; earlier data is kept for
indicator warmup. Indicator results are cached and only recomputed when
new candles arrive.

Usage:
    PYTHONPATH=. python scripts/live_chart.py
    PYTHONPATH=. python scripts/live_chart.py --port 8051 --days 3 --view-hours 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from indicators.base import IndicatorConfig
from indicators.fvg import FVGIndicator
from indicators.td_sequential import TDSequentialIndicator
from indicators.ut_bot import UTBotIndicator
from scripts.fetch_binance_btc import fetch_klines

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_ohlc: pd.DataFrame = pd.DataFrame()
_view_hours: int = 12

# Indicator cache: recompute only when candle count changes
_cache_lock = threading.Lock()
_cache: dict = {"n": 0, "results": {}}

AVAILABLE_INDICATORS = {
    "ut_bot": "UT Bot Alert",
    "td_seq": "TD Sequential",
    "fvg": "Fair Value Gap",
}

TZ_DISPLAY = "Etc/GMT+7"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_history(days: int) -> pd.DataFrame:
    today = date.today()
    start = today - timedelta(days=days)
    print(f"Loading 1m history: {start} → {today}")
    df = fetch_klines(start, today, interval="1m")
    print(f"  {len(df)} candles loaded")
    return df


def _merge_ws_candle(kline: dict) -> None:
    global _ohlc
    open_ts = kline["t"] / 1000.0
    row = {
        "timestamp": open_ts,
        "open": float(kline["o"]),
        "high": float(kline["h"]),
        "low": float(kline["l"]),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
    }
    with _lock:
        if _ohlc.empty:
            return
        last_ts = _ohlc.iloc[-1]["timestamp"]
        if open_ts == last_ts:
            idx = _ohlc.index[-1]
            for k, v in row.items():
                _ohlc.at[idx, k] = v
        elif open_ts > last_ts:
            _ohlc = pd.concat([_ohlc, pd.DataFrame([row])], ignore_index=True)


# ---------------------------------------------------------------------------
# WebSocket thread
# ---------------------------------------------------------------------------

def _ws_thread() -> None:
    import websockets

    async def _listen():
        url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        async for ws in websockets.connect(url, ping_interval=20):
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    k = msg.get("k")
                    if k:
                        _merge_ws_candle(k)
            except Exception:
                await asyncio.sleep(1)

    asyncio.run(_listen())


# ---------------------------------------------------------------------------
# Indicator cache
# ---------------------------------------------------------------------------

def _get_cached_indicators(ohlc: pd.DataFrame, selected: list[str]) -> dict:
    """Return cached indicator results, recomputing only when candle count changes."""
    n = len(ohlc)
    with _cache_lock:
        if _cache["n"] == n:
            # Same candle count — return cached (even for in-progress candle updates,
            # the tiny price change doesn't warrant full recompute)
            missing = [s for s in selected if s not in _cache["results"]]
            if not missing:
                return {k: _cache["results"][k] for k in selected if k in _cache["results"]}

    # Recompute needed indicators
    results = {}
    if "ut_bot" in selected:
        ut = UTBotIndicator(IndicatorConfig("UTBot", {"atr_period": 10, "key_value": 1.0}))
        results["ut_bot"] = ut.compute(ohlc, "1m")
    if "td_seq" in selected:
        td = TDSequentialIndicator(IndicatorConfig("TDSeq", {}))
        results["td_seq"] = td.compute(ohlc, "1m")
    if "fvg" in selected:
        fvg = FVGIndicator(IndicatorConfig("FVG", {"threshold_percent": 0.0, "auto": False}))
        results["fvg"] = fvg.compute(ohlc, "1m")

    with _cache_lock:
        _cache["n"] = n
        _cache["results"].update(results)
        return {k: _cache["results"][k] for k in selected if k in _cache["results"]}


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def _build_figure(full_ohlc: pd.DataFrame, selected: list[str], view_hours: int) -> go.Figure:
    if full_ohlc.empty:
        return go.Figure()

    indicators = _get_cached_indicators(full_ohlc, selected)

    n_full = len(full_ohlc)
    view_candles = view_hours * 60
    view_start = max(0, n_full - view_candles)

    # Slice for display
    disp = full_ohlc.iloc[view_start:].reset_index(drop=True)
    n = len(disp)
    dt = pd.to_datetime(disp["timestamp"], unit="s", utc=True).dt.tz_convert(TZ_DISPLAY)
    high_arr = disp["high"].to_numpy()
    low_arr = disp["low"].to_numpy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.03,
    )

    # -- Candlestick --
    fig.add_trace(go.Candlestick(
        x=dt, open=disp["open"], high=disp["high"],
        low=disp["low"], close=disp["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="BTC/USDT",
    ), row=1, col=1)

    # -- UT Bot --
    if "ut_bot" in indicators:
        res = indicators["ut_bot"]
        trail = res.values["trail"][view_start:]
        buy = np.array(res.values["buy"][view_start:], dtype=bool)
        sell = np.array(res.values["sell"][view_start:], dtype=bool)

        fig.add_trace(go.Scatter(
            x=dt, y=trail, mode="lines",
            line=dict(color="orange", width=1), name="UT Trail",
        ), row=1, col=1)
        if buy.any():
            fig.add_trace(go.Scatter(
                x=dt[buy], y=low_arr[buy] * 0.9998,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                name="UT Buy",
            ), row=1, col=1)
        if sell.any():
            fig.add_trace(go.Scatter(
                x=dt[sell], y=high_arr[sell] * 1.0002,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                name="UT Sell",
            ), row=1, col=1)

    # -- TD Sequential (use scatter+text instead of annotations for speed) --
    if "td_seq" in indicators:
        res = indicators["td_seq"]
        td_up = res.values["td_up"][view_start:]
        td_dn = res.values["td_dn"][view_start:]

        # Bearish setup/countdown — above candles in red
        up_mask = td_up > 0
        if up_mask.any():
            labels = [str(int(v)) if v <= 9 else chr(64 + int(v) - 9) for v in td_up[up_mask]]
            fig.add_trace(go.Scatter(
                x=dt[up_mask], y=high_arr[up_mask] * 1.0003,
                mode="text", text=labels,
                textfont=dict(size=8, color="#ef5350"),
                textposition="top center", name="TD Up",
                showlegend=False,
            ), row=1, col=1)

        # Bullish setup/countdown — below candles in green
        dn_mask = td_dn > 0
        if dn_mask.any():
            labels = [str(int(v)) if v <= 9 else chr(64 + int(v) - 9) for v in td_dn[dn_mask]]
            fig.add_trace(go.Scatter(
                x=dt[dn_mask], y=low_arr[dn_mask] * 0.9997,
                mode="text", text=labels,
                textfont=dict(size=8, color="#26a69a"),
                textposition="bottom center", name="TD Dn",
                showlegend=False,
            ), row=1, col=1)

    # -- FVG --
    if "fvg" in indicators:
        res = indicators["fvg"]
        all_recs = res.metadata.get("all_records", [])
        for rec in all_recs:
            if rec.mitigated or rec.start_index < view_start:
                continue
            color = "rgba(38,166,154,0.15)" if rec.is_bullish else "rgba(239,83,80,0.15)"
            border = "#26a69a" if rec.is_bullish else "#ef5350"
            ri = rec.start_index - view_start
            x0 = dt.iloc[ri] if 0 <= ri < n else dt.iloc[0]
            fig.add_shape(
                type="rect", x0=x0, x1=dt.iloc[-1],
                y0=rec.min_level, y1=rec.max_level,
                fillcolor=color, line=dict(color=border, width=0.5),
                row=1, col=1,
            )

    # -- Volume --
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(disp["close"], disp["open"])]
    fig.add_trace(go.Bar(
        x=dt, y=disp["volume"],
        marker_color=colors, name="Volume", opacity=0.5,
    ), row=2, col=1)

    # -- Current price --
    last_close = float(disp["close"].iloc[-1])
    fig.add_hline(
        y=last_close, line=dict(color="white", width=0.5, dash="dot"),
        annotation_text=f"{last_close:,.2f}",
        annotation_font=dict(size=10, color="white"),
        row=1, col=1,
    )

    active = [AVAILABLE_INDICATORS[k] for k in selected if k in AVAILABLE_INDICATORS]
    subtitle = " + ".join(active) if active else "No indicators"

    fig.update_layout(
        template="plotly_dark",
        title=f"BTC/USDT 1m (UTC-7)  |  {subtitle}  |  Last {view_hours}h",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=60, r=20, t=60, b=30),
        uirevision="constant",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def _create_app() -> Dash:
    app = Dash(__name__)
    app.title = "BTC Live Chart"

    indicator_options = [{"label": v, "value": k} for k, v in AVAILABLE_INDICATORS.items()]
    view_options = [
        {"label": "3h", "value": 3},
        {"label": "6h", "value": 6},
        {"label": "12h", "value": 12},
        {"label": "24h", "value": 24},
        {"label": "3d", "value": 72},
    ]

    app.layout = html.Div(
        style={"backgroundColor": "#131722", "minHeight": "100vh", "padding": "10px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "20px",
                        "marginBottom": "5px", "flexWrap": "wrap"},
                children=[
                    html.Span("Indicators:", style={"color": "#d1d4dc", "fontSize": "14px"}),
                    dcc.Checklist(
                        id="indicator-select",
                        options=indicator_options,
                        value=["ut_bot", "td_seq"],
                        inline=True,
                        style={"color": "#d1d4dc", "fontSize": "13px"},
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px"},
                    ),
                    html.Span("|", style={"color": "#444"}),
                    html.Span("View:", style={"color": "#d1d4dc", "fontSize": "14px"}),
                    dcc.RadioItems(
                        id="view-hours",
                        options=view_options,
                        value=_view_hours,
                        inline=True,
                        style={"color": "#d1d4dc", "fontSize": "13px"},
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px"},
                    ),
                    html.Span(id="status",
                              style={"color": "#666", "fontSize": "12px", "marginLeft": "auto"}),
                ],
            ),
            dcc.Graph(id="chart", style={"height": "85vh"}),
            dcc.Interval(id="refresh", interval=10000, n_intervals=0),
        ],
    )

    @app.callback(
        [Output("chart", "figure"), Output("status", "children")],
        [Input("refresh", "n_intervals"),
         Input("indicator-select", "value"),
         Input("view-hours", "value")],
    )
    def update(n_intervals, selected, view_hours):
        with _lock:
            ohlc = _ohlc.copy()
        if ohlc.empty:
            return go.Figure(), "Loading..."

        t0 = time.time()
        fig = _build_figure(ohlc, selected or [], int(view_hours or 12))
        elapsed = time.time() - t0
        last_ts = ohlc["timestamp"].iloc[-1]
        last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
        status = f"{len(ohlc)} candles | Last: {last_dt.strftime('%H:%M:%S')} UTC | Render: {elapsed:.1f}s"
        return fig, status

    return app


def main() -> None:
    global _ohlc, _view_hours

    parser = argparse.ArgumentParser(description="Live BTC chart with indicators")
    parser.add_argument("--days", type=int, default=3, help="Days of history to load")
    parser.add_argument("--view-hours", type=int, default=12, help="Hours to display (default 12)")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port")
    args = parser.parse_args()

    _view_hours = args.view_hours
    history = _load_history(args.days)
    with _lock:
        _ohlc = history

    # Pre-warm indicator cache so first render is fast
    print("Pre-computing indicators...")
    _get_cached_indicators(history, list(AVAILABLE_INDICATORS.keys()))
    print("  Done")

    ws = threading.Thread(target=_ws_thread, daemon=True)
    ws.start()
    print("WebSocket connected for live updates")

    app = _create_app()
    print(f"Opening http://localhost:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
