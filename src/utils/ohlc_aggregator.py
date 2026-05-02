"""Aggregate irregular (ts, price[, volume]) ticks into fixed-interval OHLC bars.

Used by the multi-timeframe feature computer to build 1m/3m/5m/15m/30m/60m/4h
candles from the shared BTC price history. Each bar is anchored to the floor
of its start timestamp (e.g. a 5m bar starting at 2026-04-18T22:00 has
bar_ts = 1776549600 UTC seconds).

Tolerates gaps: empty buckets are simply omitted. Callers that need a
contiguous index should reindex afterward.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def aggregate_to_ohlc(
    ticks: Sequence[Tuple[float, float]] | Sequence[Tuple[float, float, float]],
    interval_s: int,
) -> pd.DataFrame:
    """Build an OHLC DataFrame from `(ts, price[, volume])` tuples.

    Args:
        ticks: iterable of 2- or 3-tuples `(ts, price)` or `(ts, price, volume)`.
            Timestamps are seconds since epoch. Order does not matter — the
            function sorts internally.
        interval_s: bar width in seconds (e.g. 60 for 1m, 14400 for 4h).

    Returns:
        DataFrame indexed by bar-start UTC timestamps with columns
        `open, high, low, close, volume`. When tick volume is missing, the
        volume column is filled with the tick count per bar (0 never occurs
        since empty buckets are dropped).

    Raises:
        ValueError: if `interval_s <= 0`.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")

    if not ticks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    arr = np.asarray(list(ticks), dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("ticks must be 2- or 3-tuples of (ts, price[, volume])")

    has_volume = arr.shape[1] == 3
    ts = arr[:, 0]
    price = arr[:, 1]
    volume = arr[:, 2] if has_volume else np.ones_like(ts)

    # Bucket by floor(ts / interval_s) * interval_s so each bar has stable edges.
    bar_ts = (ts // interval_s).astype(np.int64) * interval_s

    df = pd.DataFrame({
        "bar_ts": bar_ts,
        "ts": ts,
        "price": price,
        "volume": volume,
    }).sort_values(["bar_ts", "ts"])

    grouped = df.groupby("bar_ts", sort=True)
    ohlc = pd.DataFrame({
        "open": grouped["price"].first(),
        "high": grouped["price"].max(),
        "low": grouped["price"].min(),
        "close": grouped["price"].last(),
        "volume": grouped["volume"].sum(),
    })
    ohlc.index = pd.to_datetime(ohlc.index, unit="s", utc=True)
    ohlc.index.name = "bar_start_utc"
    return ohlc


def aggregate_ohlc_from_1s_ohlc(
    df_1s: pd.DataFrame,
    interval_s: int,
) -> pd.DataFrame:
    """Aggregate an already-bucketed 1-second OHLCV DataFrame into larger bars.

    Faster than going through `aggregate_to_ohlc` when the source is already
    1s OHLCV (e.g. `btc_live_1s.csv`). Input must contain
    `timestamp, open, high, low, close, volume` columns.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df_1s.columns)
    if missing:
        raise ValueError(f"df_1s missing columns: {sorted(missing)}")

    bar_ts = (df_1s["timestamp"].astype(np.int64) // interval_s) * interval_s
    grouped = df_1s.assign(bar_ts=bar_ts).groupby("bar_ts", sort=True)
    out = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "volume": grouped["volume"].sum(),
    })
    out.index = pd.to_datetime(out.index, unit="s", utc=True)
    out.index.name = "bar_start_utc"
    return out
