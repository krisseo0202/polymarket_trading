"""Tests for the Binance historical REST client — no live HTTP."""

import json
from unittest.mock import MagicMock

import pytest

from src.utils.binance_historical import fetch_btc_klines


def _klines_chunk(start_ms: int, n: int, interval_s: int, base_close: float = 100_000.0) -> list:
    """Build a list of Binance-style kline arrays [open_time_ms, o, h, l, c, v, ...]."""
    rows = []
    for i in range(n):
        ot = start_ms + i * interval_s * 1000
        close = str(base_close + i)
        rows.append([
            ot, "99999.0", "100100.0", "99800.0", close, "10.5",
            ot + interval_s * 1000 - 1, "0", 0, "0", "0", "0",
        ])
    return rows


def _mock_opener(responses: list):
    """Build a fake urllib opener that returns queued JSON payloads."""
    opener = MagicMock()
    call_iter = iter(responses)

    def _open(url, timeout=30.0):
        body = next(call_iter)
        resp = MagicMock()
        resp.read.return_value = json.dumps(body).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    opener.open.side_effect = _open
    return opener


def test_fetch_single_chunk_returns_dataframe():
    interval_s = 60
    chunk = _klines_chunk(start_ms=1776556800 * 1000, n=10, interval_s=interval_s)
    opener = _mock_opener([chunk])

    df = fetch_btc_klines(
        start_ts=1776556800,
        end_ts=1776556800 + 10 * interval_s,
        interval="1m",
        session_opener=opener,
    )

    assert len(df) == 10
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].iloc[0] == 1776556800
    assert df["timestamp"].iloc[-1] == 1776556800 + 9 * interval_s
    assert df["close"].iloc[0] == 100_000.0


def test_fetch_pages_through_multiple_chunks():
    interval_s = 60
    # 2500 bars total → 3 paged requests (1000 + 1000 + 500).
    chunk1 = _klines_chunk(1776556800 * 1000, n=1000, interval_s=interval_s)
    chunk2 = _klines_chunk((1776556800 + 1000 * 60) * 1000, n=1000, interval_s=interval_s)
    chunk3 = _klines_chunk((1776556800 + 2000 * 60) * 1000, n=500, interval_s=interval_s)
    opener = _mock_opener([chunk1, chunk2, chunk3])

    df = fetch_btc_klines(
        start_ts=1776556800,
        end_ts=1776556800 + 2500 * interval_s,
        interval="1m",
        session_opener=opener,
    )
    # Dedup keeps 2500 unique bars.
    assert len(df) == 2500
    assert opener.open.call_count == 3


def test_fetch_deduplicates_overlapping_pages():
    # Binance's windowing can return the same bar in two consecutive pages
    # when the endTime of page N and startTime of page N+1 align on a bar edge.
    interval_s = 60
    chunk1 = _klines_chunk(1776556800 * 1000, n=5, interval_s=interval_s)
    chunk2 = _klines_chunk((1776556800 + 4 * 60) * 1000, n=5, interval_s=interval_s)  # overlaps last bar of chunk1
    opener = _mock_opener([chunk1, chunk2])

    df = fetch_btc_klines(
        start_ts=1776556800,
        end_ts=1776556800 + 9 * interval_s,
        interval="1m",
        session_opener=opener,
    )
    # 5 + 5 raw rows → 9 unique after dedup on timestamp.
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].is_unique


def test_fetch_empty_window_returns_empty_frame():
    opener = _mock_opener([[]])
    df = fetch_btc_klines(
        start_ts=1776556800,
        end_ts=1776556800 + 60,
        interval="1m",
        session_opener=opener,
    )
    assert len(df) == 0
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_rejects_bad_interval():
    with pytest.raises(ValueError):
        fetch_btc_klines(1, 2, interval="bogus")


def test_fetch_rejects_end_before_start():
    with pytest.raises(ValueError):
        fetch_btc_klines(100, 50, interval="1m")
