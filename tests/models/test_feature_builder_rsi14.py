"""Tests for the single-TF RSI-14 feature."""

import time

from src.api.types import OrderBook, OrderBookEntry
from src.models.feature_builder import build_live_features
from src.models.schema import FEATURE_COLUMNS


def _book(token: str) -> OrderBook:
    return OrderBook(
        market_id="mkt", token_id=token,
        bids=[OrderBookEntry(price=0.50, size=100.0)],
        asks=[OrderBookEntry(price=0.51, size=100.0)],
        tick_size=0.001,
    )


def _snapshot(prices: list) -> dict:
    now = prices[-1][0]
    return {
        "btc_prices": prices,
        "yes_book": _book("yes"),
        "no_book": _book("no"),
        "yes_history": [],
        "no_history": [],
        "question": "Will BTC be above $100,000 at 12:05 PM ET?",
        "slot_expiry_ts": now + 120,
        "now_ts": now,
    }


def test_rsi_14_in_schema():
    assert "rsi_14" in FEATURE_COLUMNS


def test_rsi_defaults_to_50_when_insufficient_history():
    now = time.time()
    # Only 10 ticks — below 15 required
    prices = [(now - 9 + i, 100_000.0 + i) for i in range(10)]
    f = build_live_features(_snapshot(prices)).features
    assert f["rsi_14"] == 50.0


def test_rsi_approaches_100_on_monotonic_rally():
    now = time.time()
    # 20 monotonically rising ticks
    prices = [(now - 19 + i, 100_000.0 + i * 50) for i in range(20)]
    f = build_live_features(_snapshot(prices)).features
    assert f["rsi_14"] == 100.0  # no losses -> saturates


def test_rsi_approaches_zero_on_monotonic_selloff():
    now = time.time()
    prices = [(now - 19 + i, 100_000.0 - i * 50) for i in range(20)]
    f = build_live_features(_snapshot(prices)).features
    # All losses, no gains -> gains=0 -> rs=0 -> rsi = 100 - 100/1 = 0
    assert f["rsi_14"] == 0.0


def test_rsi_near_50_on_balanced_moves():
    now = time.time()
    # Alternating ±$50 ticks -> gains == losses -> rs = 1 -> rsi = 50
    prices = []
    base = 100_000.0
    for i in range(20):
        price = base + (50 if i % 2 == 0 else 0)
        prices.append((now - 19 + i, price))
    f = build_live_features(_snapshot(prices)).features
    # Gains == losses over last 14 diffs -> rsi = 50
    assert abs(f["rsi_14"] - 50.0) < 1.0


def test_rsi_stays_in_0_100_range():
    import random
    rng = random.Random(42)
    now = time.time()
    prices = [(now - 99 + i, 100_000.0 + rng.gauss(0, 100)) for i in range(100)]
    f = build_live_features(_snapshot(prices)).features
    assert 0.0 <= f["rsi_14"] <= 100.0
