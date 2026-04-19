"""Tests for BtcPriceFeed.warmup_from_binance — no real HTTP."""

from unittest.mock import patch

import pandas as pd

from src.utils.btc_feed import BtcPriceFeed


def _klines(start_ts: int, n: int) -> pd.DataFrame:
    rows = [{
        "timestamp": start_ts + i,
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0 + i * 0.01, "volume": 1.0,
    } for i in range(n)]
    return pd.DataFrame(rows)


def test_warmup_populates_buffer():
    feed = BtcPriceFeed(symbol="BTC-USD", exchange="coinbase", buffer_s=3 * 86400)
    # Mock Binance to return 60 bars ending ~10s ago.
    import time
    now = time.time()
    fake_df = _klines(int(now) - 60, n=60)
    with patch("src.utils.binance_historical.fetch_btc_klines", return_value=fake_df):
        loaded = feed.warmup_from_binance(days=3)
    assert loaded == 60
    recent = feed.get_recent_prices(window_s=120)
    assert len(recent) == 60
    # Timestamps ascending.
    ts = [t for t, _, _ in recent]
    assert ts == sorted(ts)


def test_warmup_drops_bars_older_than_buffer_window():
    feed = BtcPriceFeed(symbol="BTC-USD", exchange="coinbase", buffer_s=60)
    import time
    now = time.time()
    # 200 bars over 200 seconds — only last ~60 should land in the buffer.
    fake_df = _klines(int(now) - 200, n=200)
    with patch("src.utils.binance_historical.fetch_btc_klines", return_value=fake_df):
        feed.warmup_from_binance(days=3)
    recent = feed.get_recent_prices(window_s=200)
    # Only the bars within buffer_s=60 of "now" should remain.
    assert len(recent) <= 61  # allow 1-tick slack for the boundary
    assert len(recent) >= 55


def test_warmup_empty_result_is_noop():
    feed = BtcPriceFeed(symbol="BTC-USD", exchange="coinbase", buffer_s=3600)
    with patch("src.utils.binance_historical.fetch_btc_klines",
               return_value=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])):
        loaded = feed.warmup_from_binance(days=3)
    assert loaded == 0
    assert feed.get_recent_prices(window_s=3600) == []
