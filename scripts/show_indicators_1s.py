"""Quick look at how technical indicators render on 1-minute BTC bars,
resampled from 1s data.

Decision-time convention: at any clock time in [T+60s, T+120s), the bot reads
the row labeled T (the most recently CLOSED 1-minute bar). No lookahead into
the in-progress minute.

Usage:
    .venv/bin/python scripts/show_indicators_1s.py
    .venv/bin/python scripts/show_indicators_1s.py --csv data/2026-04-08/btc_live_1s.csv --rows 60 --timeframe 1min
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.base import IndicatorConfig
from indicators.fvg import FVGIndicator
from indicators.rsi import RSIIndicator
from indicators.td_sequential import TDSequentialIndicator
from indicators.ut_bot import UTBotIndicator


def at(features: pd.DataFrame, decision_ts, timeframe: str) -> pd.Series:
    """Return the single row the bot would read at `decision_ts`.

    Picks the most recently CLOSED bar: row label = floor(decision_ts / tf) - 1*tf.
    Raises if no such bar exists in `features`.
    """
    d = pd.Timestamp(decision_ts)
    if d.tz is None:
        d = d.tz_localize("UTC")
    tf = pd.Timedelta(timeframe)
    target = d.floor(timeframe) - tf
    if target not in features.index:
        before = features.index[features.index <= target]
        if len(before) == 0:
            raise KeyError(f"No closed bar at or before {target} in features")
        target = before[-1]
    return features.loc[target]


def closed_bars(df_1s: pd.DataFrame, rule: str, now_ts=None) -> pd.DataFrame:
    """Resample 1s OHLCV to `rule` and return ONLY fully-closed bars.

    Every emitted row is labeled by its open time and is guaranteed to cover
    a complete [open, open + rule) window relative to `now_ts`. The
    in-progress bar at the tail is dropped so indicators see no leakage.

    Args:
        df_1s: 1-second OHLCV with DatetimeIndex (UTC).
        rule: pandas offset alias ('1min', '15s', '5min'…). '1s' is passed
              through unchanged.
        now_ts: clock time of the read. Defaults to the last index value.
                Bars at or after floor(now_ts, rule) are dropped.
    """
    if rule == "1s":
        return df_1s
    last_ts = pd.Timestamp(now_ts) if now_ts is not None else df_1s.index[-1]
    if last_ts.tz is None:
        last_ts = last_ts.tz_localize("UTC")
    cutoff = last_ts.floor(rule)  # start of the in-progress bar at now_ts
    trimmed = df_1s[df_1s.index < cutoff]
    out = trimmed.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/2026-04-08/btc_live_1s.csv")
    p.add_argument("--timeframe", default="1min", help="resample rule: 1s, 15s, 1min, 5min …")
    p.add_argument("--rows", type=int, default=60, help="show last N bars after warmup")
    p.add_argument("--warmup", type=int, default=60, help="warmup bars before first shown row")
    p.add_argument("--at", default=None,
                   help="ISO timestamp (UTC). Print the single row the bot would read at that decision time.")
    args = p.parse_args()

    raw = pd.read_csv(args.csv)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="s", utc=True)
    raw = raw.set_index("timestamp")

    df = closed_bars(raw, args.timeframe, now_ts=args.at)
    df = df.tail(args.rows + args.warmup).copy()

    rsi = RSIIndicator(IndicatorConfig(name="RSI", params={"period": 14})).compute(df, "1s")
    ut = UTBotIndicator(IndicatorConfig(name="UTBot",
                                        params={"atr_period": 10, "key_value": 1.0, "ema_period": 1})).compute(df, "1s")
    fvg = FVGIndicator(IndicatorConfig(name="FVG", params={"threshold_percent": 0.0, "auto": True})).compute(df, "1s")
    td = TDSequentialIndicator(IndicatorConfig(name="TD", params={})).compute(df, "1s")

    out = df[["open", "high", "low", "close", "volume"]].copy()
    out["rsi14"] = rsi.values["rsi"]
    out["ut_trail"] = ut.values["trail"]
    out["ut_buy"] = ut.values["buy"].astype(int)
    out["ut_sell"] = ut.values["sell"].astype(int)
    out["td_up"] = td.values["td_up"]
    out["td_dn"] = td.values["td_dn"]
    out["tdst_support"] = td.values["tdst_support"]
    out["tdst_resistance"] = td.values["tdst_resistance"]

    out = out.tail(args.rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print(f"Source: {args.csv}")
    print(f"Timeframe: {args.timeframe}  warmup={args.warmup} bars  shown={len(out)} bars")
    print("Bars are LABELED BY OPEN TIME. Row labeled T = bar covering [T, T + 1 timeframe).")
    print("At decision time D, read the row with label = floor(D / timeframe) - 1 timeframe.")
    print("─" * 110)
    print("HEAD")
    print(out.head(8))
    print("\nTAIL")
    print(out.tail(8))
    print("\nDESCRIBE (numeric)")
    print(out[["close", "rsi14", "ut_trail", "td_up", "td_dn"]].describe())
    print("\nFVG SUMMARY (entire window)")
    print(f"  bull_count={fvg.values['bull_count']}  bear_count={fvg.values['bear_count']}  "
          f"bull_mitigated={fvg.values['bull_mitigated']}  bear_mitigated={fvg.values['bear_mitigated']}")
    print(f"  ut_buy_signals={int(out['ut_buy'].sum())}  ut_sell_signals={int(out['ut_sell'].sum())}")

    if args.at:
        row = at(out, args.at, args.timeframe)
        print(f"\nAT decision_ts={args.at} → reading bar labeled {row.name}")
        print(row.to_string())


if __name__ == "__main__":
    main()
