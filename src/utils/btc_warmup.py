"""Ensure the BTC price series covers enough history for multi-TF indicators.

Used by both the training pipeline (batch) and the live bot (startup).
Multi-timeframe indicator warmup requires ≥2 days of BTC data for the 4h
bar to produce meaningful setup/countdown values. This module closes the
gap between "how much we have locally" and "how much we need" by pulling
1-second klines from Binance REST.

Design choice: we fetch **1s** klines rather than 1m because the multi-TF
feature computer aggregates from 1s-granularity input — passing pre-aggregated
1m bars would give degenerate OHLC (open==close==high==low) for the 1m TF.
The tradeoff is ~3 min of startup time for 2 days of coverage (173 REST
calls at ~1s each), which only happens once per session or training run.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd

from .binance_historical import fetch_btc_klines


# 2 days is the minimum; 3 gives comfortable headroom for 4h TD Sequential.
DEFAULT_WARMUP_DAYS = 3
WARMUP_INTERVAL = "1s"


def warmup_btc_history(
    existing: Optional[pd.DataFrame],
    need_start_ts: float,
    need_end_ts: float,
    interval: str = WARMUP_INTERVAL,
    symbol: str = "BTCUSDT",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Return a BTC series guaranteed to cover ``[need_start_ts, need_end_ts]``.

    Strategy:
      1. Use ``existing`` as the base (may be ``None`` or an empty DataFrame).
      2. If the base doesn't cover ``need_start_ts`` at the head, fetch a
         left-side warmup chunk from Binance.
      3. If the base doesn't cover ``need_end_ts`` at the tail, fetch a
         right-side tail chunk from Binance.
      4. Merge, de-duplicate on timestamp, sort ascending.

    Args:
        existing: DataFrame with columns ``timestamp, open, high, low, close,
            volume`` (same schema Binance returns). ``None`` treated as empty.
        need_start_ts, need_end_ts: UTC seconds window that must be covered.
        interval: Binance kline interval token (default ``"1s"``).
        symbol: Binance symbol.
        logger: optional logger.

    Returns:
        DataFrame covering the requested window, sorted by timestamp.
    """
    log = logger or logging.getLogger(__name__)
    if need_end_ts <= need_start_ts:
        raise ValueError("need_end_ts must be > need_start_ts")

    base = existing if existing is not None and not existing.empty else _empty_frame()
    _validate_schema(base)

    chunks: List[pd.DataFrame] = [base]

    if base.empty:
        head_gap = (need_start_ts, need_end_ts)
        tail_gap = None
    else:
        base_start = float(base["timestamp"].iloc[0])
        base_end = float(base["timestamp"].iloc[-1])
        head_gap = (need_start_ts, min(base_start, need_end_ts)) if base_start > need_start_ts else None
        tail_gap = (max(base_end, need_start_ts), need_end_ts) if base_end < need_end_ts else None

    if head_gap is not None:
        log.info("warmup_btc_history: fetching head gap %s..%s", *head_gap)
        chunks.append(fetch_btc_klines(head_gap[0], head_gap[1], interval=interval, symbol=symbol, logger=log))
    if tail_gap is not None:
        log.info("warmup_btc_history: fetching tail gap %s..%s", *tail_gap)
        chunks.append(fetch_btc_klines(tail_gap[0], tail_gap[1], interval=interval, symbol=symbol, logger=log))

    merged = pd.concat([c for c in chunks if not c.empty], ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def btc_dataframe_to_tuples(df: pd.DataFrame) -> List[Tuple[float, float, float]]:
    """Convert a BTC OHLCV frame to the (ts, close, volume) tuples that
    feature_builder.build_live_features expects as ``btc_prices``."""
    if df.empty:
        return []
    return [
        (float(r.timestamp), float(r.close), float(r.volume))
        for r in df.itertuples(index=False)
    ]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _validate_schema(df: pd.DataFrame) -> None:
    if df.empty:
        return
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BTC frame missing columns: {sorted(missing)}")
