"""Render TD Sequential on BTC 5-minute bars built from S3 1s data.

Usage:
    .venv/bin/python scripts/td_chart_s3.py --date 2026-04-21
    .venv/bin/python scripts/td_chart_s3.py --date 2026-04-21 --hours 00-12
"""

from __future__ import annotations

import argparse
import io
from datetime import date as date_cls
from typing import Literal

import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators.base import IndicatorConfig
from indicators.td_sequential import TDSequentialIndicator
from indicators.ut_bot import UTBotIndicator
from src.backtest.s3_snapshot_loader import (
    BTC_FILENAME,
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
)

REGION = "us-west-2"


def fetch_1s_from_s3(client, date: str, hours: list[str]) -> pd.DataFrame:
    frames = []
    for hh in hours:
        key = f"{DEFAULT_PREFIX}/{date}/{hh}/{BTC_FILENAME}"
        try:
            obj = client.get_object(Bucket=DEFAULT_BUCKET, Key=key)
        except client.exceptions.NoSuchKey:
            print(f"  skip {key} (missing)")
            continue
        frames.append(pd.read_csv(io.BytesIO(obj["Body"].read())))
        print(f"  got {key}")
    if not frames:
        raise RuntimeError("No 1s CSVs fetched")
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


def aggregate_1s_to_5m(df_1s: pd.DataFrame) -> pd.DataFrame:
    df = df_1s.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["bucket"] = df["dt"].dt.floor("5min")
    agg = (
        df.groupby("bucket")
        .agg(
            timestamp=("timestamp", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_index()
        .reset_index(drop=True)
    )
    return agg


def build_chart(
    ohlc: pd.DataFrame,
    date: str,
    td_mode: Literal["classic", "pine_simple"] = "classic",
) -> go.Figure:
    td = TDSequentialIndicator(IndicatorConfig("TDSeq", {"mode": td_mode}))
    res = td.compute(ohlc, timeframe="5m")
    td_up = res.values["td_up"]
    td_dn = res.values["td_dn"]

    ut = UTBotIndicator(IndicatorConfig("UTBot", {"atr_period": 10, "key_value": 1.0}))
    ut_res = ut.compute(ohlc, timeframe="5m")
    trail = ut_res.values["trail"]
    ut_buy = ut_res.values["buy"].astype(bool)
    ut_sell = ut_res.values["sell"].astype(bool)

    display = ohlc.copy()
    display["dt"] = (
        pd.to_datetime(display["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Etc/GMT+8")
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=display["dt"], open=display["open"], high=display["high"],
        low=display["low"], close=display["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="BTC 5m",
    ), row=1, col=1)

    up_mask = np.asarray(td_up) > 0
    dn_mask = np.asarray(td_dn) > 0
    if up_mask.any():
        fig.add_trace(go.Scatter(
            x=display["dt"][up_mask], y=display["high"][up_mask],
            mode="text", text=[str(int(v)) for v in td_up[up_mask]],
            textposition="top center",
            textfont=dict(size=9, color="#ef5350"),
            name="TD Up", showlegend=False,
        ), row=1, col=1)
    if dn_mask.any():
        fig.add_trace(go.Scatter(
            x=display["dt"][dn_mask], y=display["low"][dn_mask],
            mode="text", text=[str(int(v)) for v in td_dn[dn_mask]],
            textposition="bottom center",
            textfont=dict(size=9, color="#26a69a"),
            name="TD Dn", showlegend=False,
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=display["dt"], y=trail, mode="lines",
        line=dict(color="orange", width=1), name="UT Bot Trail",
    ), row=1, col=1)
    if ut_buy.any():
        fig.add_trace(go.Scatter(
            x=display["dt"][ut_buy], y=display["low"][ut_buy] * 0.9995,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
            name="UT Buy",
        ), row=1, col=1)
    if ut_sell.any():
        fig.add_trace(go.Scatter(
            x=display["dt"][ut_sell], y=display["high"][ut_sell] * 1.0005,
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
            name="UT Sell",
        ), row=1, col=1)

    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(display["close"], display["open"])]
    fig.add_trace(go.Bar(
        x=display["dt"], y=display["volume"],
        marker_color=colors, name="Volume", opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        title=f"BTC 5m — {date} (PST, UTC-8)  |  TD [{td_mode}] + UT Bot (S3)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=800, width=1600,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig


def parse_hours(spec: str | None) -> list[str]:
    if not spec:
        return [f"{h:02d}" for h in range(24)]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return [f"{h:02d}" for h in range(int(lo), int(hi) + 1)]
    return [f"{int(h):02d}" for h in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(date_cls.today()),
                        help="UTC date, YYYY-MM-DD")
    parser.add_argument("--hours", default=None,
                        help="Hour range 'HH-HH' or csv 'H,H' (default: all 24)")
    parser.add_argument("--output", default="data/btc_5m_td_s3.html")
    parser.add_argument("--td-mode", default="classic",
                        choices=["classic", "pine_simple"],
                        help="classic = non-consecutive countdown after 9 (up to 13); "
                             "pine_simple = joshua0702 extended setup count (up to 16)")
    args = parser.parse_args()

    hours = parse_hours(args.hours)
    client = boto3.client("s3", region_name=REGION)

    print(f"Fetching s3://{DEFAULT_BUCKET}/{DEFAULT_PREFIX}/{args.date}/ for hours {hours[0]}..{hours[-1]}")
    df_1s = fetch_1s_from_s3(client, args.date, hours)
    print(f"Got {len(df_1s)} 1s rows")

    df_5m = aggregate_1s_to_5m(df_1s)
    print(f"Aggregated to {len(df_5m)} 5m bars")

    fig = build_chart(df_5m, args.date, td_mode=args.td_mode)
    fig.write_html(args.output)
    print(f"Chart saved to {args.output}")


if __name__ == "__main__":
    main()
