"""Binance REST historical kline fetcher.

Used for warmup of multi-timeframe indicator features — both at bot startup
(live inference) and before training dataset construction (batch).

Why 1m klines by default
------------------------
The smallest timeframe we compute indicators on is 1 minute, so 1s resolution
is unnecessary for multi-TF warmup. 1m klines shrink a 2-day warmup from
~173 REST calls (1s) to 3 (1m), at identical indicator output.

No auth is required — Binance's `/api/v3/klines` is a public endpoint. We
keep the client dependency-light (stdlib only) so it runs in CI / constrained
environments where `requests` may not be installed.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

import pandas as pd


_BINANCE_API_BASE = "https://api.binance.com"
_KLINES_PATH = "/api/v3/klines"
_MAX_LIMIT = 1000  # per-call cap documented by Binance
# Supported intervals. Keep the string tokens Binance expects.
_SUPPORTED_INTERVALS = (
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
)
_INTERVAL_SECONDS = {
    "1s": 1, "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
    "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
}


def fetch_btc_klines(
    start_ts: float,
    end_ts: float,
    interval: str = "1m",
    symbol: str = "BTCUSDT",
    session_opener=None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Fetch OHLCV klines covering [start_ts, end_ts] from Binance REST.

    Args:
        start_ts, end_ts: UTC seconds. Binance expects milliseconds internally
            but this function converts for you.
        interval: Binance interval token ("1s", "1m", "5m", "1h", ...).
        symbol: trading pair, default "BTCUSDT".
        session_opener: optional ``urllib.request.OpenerDirector`` for tests.
        max_retries: number of retries per paged request on transient errors.
        retry_backoff_s: base backoff between retries (exponential).

    Returns:
        DataFrame with columns ``timestamp, open, high, low, close, volume``
        where ``timestamp`` is the bar-open time in UTC seconds. Sorted
        ascending and deduplicated.

    Raises:
        ValueError: bad args.
        RuntimeError: server returned non-JSON or HTTP error after retries.
    """
    if interval not in _SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval!r}; valid: {_SUPPORTED_INTERVALS}")
    if end_ts <= start_ts:
        raise ValueError("end_ts must be > start_ts")

    log = logger or logging.getLogger(__name__)
    interval_s = _INTERVAL_SECONDS[interval]
    opener = session_opener or urllib.request.build_opener()

    rows: List[List] = []
    # Binance returns at most 1000 klines per call — step through in chunks.
    cursor_ms = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)
    step_ms = _MAX_LIMIT * interval_s * 1000

    while cursor_ms < end_ms:
        chunk_end_ms = min(cursor_ms + step_ms, end_ms)
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor_ms,
            "endTime": chunk_end_ms,
            "limit": _MAX_LIMIT,
        }
        url = f"{_BINANCE_API_BASE}{_KLINES_PATH}?{urllib.parse.urlencode(params)}"
        chunk = _request_with_retry(opener, url, max_retries, retry_backoff_s, log)

        if not chunk:
            # No data in this window (e.g. future timestamps) — advance and try next.
            cursor_ms = chunk_end_ms
            continue
        rows.extend(chunk)
        # Advance past the last returned open-time + one interval so we don't
        # re-fetch the same bar on the next pass.
        last_open_ms = int(chunk[-1][0])
        cursor_ms = last_open_ms + interval_s * 1000

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=[
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ])
    df = df.astype({
        "open_time_ms": "int64", "open": "float64", "high": "float64",
        "low": "float64", "close": "float64", "volume": "float64",
    })
    df["timestamp"] = (df["open_time_ms"] // 1000).astype("int64")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _request_with_retry(
    opener,
    url: str,
    max_retries: int,
    base_backoff_s: float,
    log: logging.Logger,
) -> list:
    """GET `url` and return the parsed JSON list, retrying on transient errors."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            with opener.open(url, timeout=30.0) as resp:
                body = resp.read()
            parsed = json.loads(body)
            if not isinstance(parsed, list):
                raise RuntimeError(f"Binance klines: expected list, got {type(parsed).__name__}")
            return parsed
        except (urllib.error.URLError, json.JSONDecodeError, RuntimeError) as e:
            last_err = e
            log.warning(
                "Binance klines request failed (attempt %d/%d): %s", attempt + 1, max_retries, e,
            )
            time.sleep(base_backoff_s * (2 ** attempt))
    raise RuntimeError(f"Binance klines: all {max_retries} retries failed; last={last_err}")
