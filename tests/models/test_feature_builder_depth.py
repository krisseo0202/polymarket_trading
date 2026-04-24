"""Tests for Family A (full-depth book) features in feature_builder."""

import time

import pytest

from src.api.types import OrderBook, OrderBookEntry
from src.models.feature_builder import build_live_features


def _multi_book(
    token_id: str,
    bids: list,
    asks: list,
) -> OrderBook:
    """Build an OrderBook from (price, size) tuples.

    Input bids must already be descending, asks ascending (matches the
    convention that live ClobClient enforces on its books).
    """
    return OrderBook(
        market_id="mkt",
        token_id=token_id,
        bids=[OrderBookEntry(price=p, size=s) for p, s in bids],
        asks=[OrderBookEntry(price=p, size=s) for p, s in asks],
        tick_size=0.001,
    )


def _snapshot_with_books(yes_book: OrderBook, no_book: OrderBook) -> dict:
    now = time.time()
    btc_prices = [(now - 70 + idx, 100_000.0 + idx * 2.0) for idx in range(71)]
    return {
        "btc_prices": btc_prices,
        "yes_book": yes_book,
        "no_book": no_book,
        "yes_history": [],
        "no_history": [],
        "question": "Will BTC be above $100,050 at 12:05 PM ET?",
        "slot_expiry_ts": now + 120,
        "now_ts": now,
    }


def test_microprice_skews_toward_heavier_side():
    yes_book = _multi_book("yes", bids=[(0.52, 100.0)], asks=[(0.54, 90.0)])
    no_book = _multi_book("no", bids=[(0.46, 50.0)], asks=[(0.48, 50.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    expected_yes = (100.0 * 0.54 + 90.0 * 0.52) / (100.0 + 90.0)
    expected_no = (50.0 * 0.48 + 50.0 * 0.46) / (50.0 + 50.0)

    assert result.features["yes_microprice"] == pytest.approx(expected_yes, abs=1e-9)
    assert result.features["no_microprice"] == pytest.approx(expected_no, abs=1e-9)
    assert result.features["yes_microprice"] > 0.53  # skewed toward ask
    assert result.features["no_microprice"] == pytest.approx(0.47)


def test_depth_slope_accumulates_with_price_distance():
    bids = [(0.52, 100.0), (0.51, 50.0), (0.50, 50.0)]
    asks = [(0.54, 100.0), (0.55, 50.0), (0.56, 50.0)]
    yes_book = _multi_book("yes", bids=bids, asks=asks)
    no_book = _multi_book("no", bids=[(0.46, 10.0)], asks=[(0.48, 10.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    # mid = 0.53. Each side has x=[0.01,0.02,0.03], cum y=[size1, size1+size2, ...]
    # Both sides symmetric → bid_slope == ask_slope, average equals either.
    # cov(x,y)/var(x) with N=3 and ddof=0 works out to 5000.
    assert result.features["yes_depth_slope"] == pytest.approx(5000.0, rel=1e-6)
    # Single-level NO book → slope is 0.0 (degenerate fit).
    assert result.features["no_depth_slope"] == 0.0


def test_depth_concentration_ratio():
    yes_book = _multi_book(
        "yes",
        bids=[(0.52, 100.0), (0.51, 50.0), (0.50, 50.0)],
        asks=[(0.54, 90.0), (0.55, 50.0), (0.56, 50.0)],
    )
    no_book = _multi_book("no", bids=[(0.46, 50.0)], asks=[(0.48, 50.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    l1 = 100.0 + 90.0
    total = (100.0 + 50.0 + 50.0) + (90.0 + 50.0 + 50.0)
    assert result.features["yes_depth_concentration"] == pytest.approx(l1 / total)
    # Single-level book: L1 is the whole book, so concentration = 1.0.
    assert result.features["no_depth_concentration"] == pytest.approx(1.0)


def test_empty_book_leaves_depth_features_at_default():
    yes_book = OrderBook(market_id="", token_id="yes", bids=[], asks=[], tick_size=0.001)
    no_book = _multi_book("no", bids=[(0.46, 50.0)], asks=[(0.48, 50.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    # All Family A features for YES default to 0.0 when the book is empty.
    assert result.features["yes_microprice"] == 0.0
    assert result.features["yes_depth_slope"] == 0.0
    assert result.features["yes_depth_concentration"] == 0.0


def test_schema_contains_family_a_columns():
    from src.models.schema import FEATURE_COLUMNS

    for col in (
        "yes_microprice",
        "yes_depth_slope",
        "yes_depth_concentration",
        "no_microprice",
        "no_depth_slope",
        "no_depth_concentration",
    ):
        assert col in FEATURE_COLUMNS
