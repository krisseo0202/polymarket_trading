"""Fetch yesterday-to-now 1s BTC data, aggregate to 1m, chart last 3h with UT Bot + TD Sequential.

Usage:
    python scripts/verify_binance_chart.py
    python scripts/verify_binance_chart.py --hours 6
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators.base import IndicatorConfig
from indicators.td_sequential import TDSequentialIndicator
from indicators.ut_bot import UTBotIndicator
from scripts.fetch_binance_btc import fetch_klines


def aggregate_1s_to_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-second OHLCV into 1-minute candles."""
    df = df_1s.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["bucket"] = df["dt"].dt.floor("1min")

    agg = df.groupby("bucket").agg(
        timestamp=("timestamp", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).sort_index().reset_index(drop=True)
    return agg


def build_chart(ohlc: pd.DataFrame, hours: int) -> go.Figure:
    """Build TradingView-style candlestick chart with UT Bot and TD Sequential."""
    # Run indicators on full data for warmup, then slice for display
    td = TDSequentialIndicator(IndicatorConfig("TDSeq", {}))
    ut = UTBotIndicator(IndicatorConfig("UTBot", {"atr_period": 10, "key_value": 1.0}))

    td_result = td.compute(ohlc, timeframe="1m")
    ut_result = ut.compute(ohlc, timeframe="1m")

    # Slice to last N hours
    cutoff_ts = ohlc["timestamp"].max() - hours * 3600
    mask = ohlc["timestamp"] >= cutoff_ts
    display = ohlc[mask].copy().reset_index(drop=True)
    offset = mask.values.argmax()  # index where display starts in full array

    trail = ut_result.values["trail"][offset:]
    ut_buy = ut_result.values["buy"][offset:]
    ut_sell = ut_result.values["sell"][offset:]
    td_up = td_result.values["td_up"][offset:]
    td_dn = td_result.values["td_dn"][offset:]

    display["dt"] = pd.to_datetime(display["timestamp"], unit="s", utc=True).dt.tz_convert("Etc/GMT+7")

    # --- Build figure ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=display["dt"], open=display["open"], high=display["high"],
        low=display["low"], close=display["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="BTC/USDT",
    ), row=1, col=1)

    # UT Bot trailing stop line
    fig.add_trace(go.Scatter(
        x=display["dt"], y=trail,
        mode="lines", line=dict(color="orange", width=1),
        name="UT Bot Trail",
    ), row=1, col=1)

    # UT Bot buy/sell markers
    buy_mask = ut_buy.astype(bool)
    sell_mask = ut_sell.astype(bool)
    if buy_mask.any():
        fig.add_trace(go.Scatter(
            x=display["dt"][buy_mask], y=display["low"][buy_mask] * 0.9998,
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
            name="UT Buy",
        ), row=1, col=1)
    if sell_mask.any():
        fig.add_trace(go.Scatter(
            x=display["dt"][sell_mask], y=display["high"][sell_mask] * 1.0002,
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
            name="UT Sell",
        ), row=1, col=1)

    # TD Sequential annotations
    for i in range(len(display)):
        # TD Up (bearish setup/countdown) — above candles in red
        if td_up[i] > 0:
            label = str(int(td_up[i])) if td_up[i] <= 9 else chr(64 + int(td_up[i]) - 9)  # A,B,C,D
            fig.add_annotation(
                x=display["dt"].iloc[i], y=float(display["high"].iloc[i]),
                text=label, showarrow=False, font=dict(size=8, color="#ef5350"),
                yshift=12, row=1, col=1,
            )
        # TD Down (bullish setup/countdown) — below candles in green
        if td_dn[i] > 0:
            label = str(int(td_dn[i])) if td_dn[i] <= 9 else chr(64 + int(td_dn[i]) - 9)
            fig.add_annotation(
                x=display["dt"].iloc[i], y=float(display["low"].iloc[i]),
                text=label, showarrow=False, font=dict(size=8, color="#26a69a"),
                yshift=-12, row=1, col=1,
            )

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(display["close"], display["open"])]
    fig.add_trace(go.Bar(
        x=display["dt"], y=display["volume"],
        marker_color=colors, name="Volume", opacity=0.5,
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"BTC/USDT 1m — Last {hours}h (UTC-7)  |  UT Bot + TD Sequential",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=800, width=1400,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=3, help="Hours to display")
    parser.add_argument("--output", default="data/btc_verify_chart.html", help="Output HTML path")
    args = parser.parse_args()

    # Fetch yesterday to now
    today = date.today()
    yesterday = today - timedelta(days=1)
    print(f"Fetching 1s data: {yesterday} → {today} ...")
    df_1s = fetch_klines(yesterday, today, interval="1s")
    print(f"Got {len(df_1s)} 1s candles")

    # Aggregate
    df_1m = aggregate_1s_to_1m(df_1s)
    print(f"Aggregated to {len(df_1m)} 1m candles")

    # Chart
    fig = build_chart(df_1m, hours=args.hours)
    fig.write_html(args.output)
    print(f"Chart saved to {args.output}")
    print("Open in browser to view.")


if __name__ == "__main__":
    main()
