"""Tests for Family A+ (top-N depth, side-split slopes) and Family E
(closed-form fair-value residual)."""

import math
import time

from src.api.types import OrderBook, OrderBookEntry
from src.models.feature_builder import build_live_features
from src.models.schema import FEATURE_COLUMNS


def _book(token_id: str, bids: list, asks: list) -> OrderBook:
    return OrderBook(
        market_id="mkt",
        token_id=token_id,
        bids=[OrderBookEntry(price=p, size=s) for p, s in bids],
        asks=[OrderBookEntry(price=p, size=s) for p, s in asks],
        tick_size=0.001,
    )


def _snapshot(yes_book: OrderBook, no_book: OrderBook,
              btc_start: float = 100_000.0, btc_end: float = 100_050.0,
              strike: float = 100_000.0, tte: float = 200.0) -> dict:
    now = time.time()
    prices = []
    for idx in range(71):
        frac = idx / 70
        prices.append((now - 70 + idx, btc_start + frac * (btc_end - btc_start)))
    return {
        "btc_prices": prices,
        "yes_book": yes_book,
        "no_book": no_book,
        "yes_history": [],
        "no_history": [],
        "question": f"Will BTC be above ${strike:,.0f} at 12:05 PM ET?",
        "slot_expiry_ts": now + tte,
        "now_ts": now,
    }


# ---------------------------------------------------------------------------
# Family A+ — top-N depth + side-split slopes
# ---------------------------------------------------------------------------

def test_topn_depth_columns_in_schema():
    for name in (
        "yes_top3_bid_depth", "yes_top3_ask_depth", "yes_top3_imbalance",
        "yes_bid_slope", "yes_ask_slope",
        "no_top3_bid_depth", "no_top3_ask_depth", "no_top3_imbalance",
        "no_bid_slope", "no_ask_slope",
    ):
        assert name in FEATURE_COLUMNS


def test_top3_sums_first_three_levels():
    yes_book = _book("yes",
        bids=[(0.55, 100.0), (0.54, 50.0), (0.53, 25.0), (0.52, 1000.0)],
        asks=[(0.57, 80.0), (0.58, 40.0), (0.59, 20.0), (0.60, 5000.0)])
    no_book  = _book("no", bids=[(0.43, 100.0)], asks=[(0.45, 100.0)])
    f = build_live_features(_snapshot(yes_book, no_book)).features
    assert f["yes_top3_bid_depth"] == 100.0 + 50.0 + 25.0
    assert f["yes_top3_ask_depth"] == 80.0 + 40.0 + 20.0
    # The 4th-level whale (1000/5000) must NOT leak into top-3.
    assert f["yes_top3_bid_depth"] < 200
    assert f["yes_top3_ask_depth"] < 200


def test_top3_imbalance_signs_with_bid_dominance():
    yes_book = _book("yes",
        bids=[(0.55, 100.0), (0.54, 100.0), (0.53, 100.0)],
        asks=[(0.57, 30.0),  (0.58, 30.0),  (0.59, 30.0)])
    no_book  = _book("no", bids=[(0.43, 1.0)], asks=[(0.45, 1.0)])
    f = build_live_features(_snapshot(yes_book, no_book)).features
    # 300 vs 90 -> (300-90)/390 ≈ +0.538
    assert f["yes_top3_imbalance"] > 0.5


def test_side_split_slopes_differ_when_book_asymmetric():
    # Steep bid side (depth grows fast away from touch), flat ask side.
    yes_book = _book("yes",
        bids=[(0.55, 10.0), (0.54, 100.0), (0.53, 500.0), (0.52, 1000.0)],
        asks=[(0.57, 20.0), (0.58, 22.0),  (0.59, 25.0),  (0.60, 28.0)])
    no_book  = _book("no", bids=[(0.43, 1.0)], asks=[(0.45, 1.0)])
    f = build_live_features(_snapshot(yes_book, no_book)).features
    # Bid slope should be much larger than ask slope.
    assert f["yes_bid_slope"] > f["yes_ask_slope"] * 5


# ---------------------------------------------------------------------------
# Family E — fair-value residuals
# ---------------------------------------------------------------------------

def test_fair_value_columns_in_schema():
    for name in ("fair_value_p_up", "yes_bid_residual", "microprice_residual"):
        assert name in FEATURE_COLUMNS


def test_fair_value_defaults_to_half_when_vol_is_zero():
    # Flat BTC -> btc_vol_30s = 0 -> fair must default to 0.5.
    yes_book = _book("yes", [(0.55, 100.0)], [(0.57, 100.0)])
    no_book  = _book("no",  [(0.43, 100.0)], [(0.45, 100.0)])
    f = build_live_features(
        _snapshot(yes_book, no_book, btc_start=100_000.0, btc_end=100_000.0)
    ).features
    assert f["btc_vol_30s"] == 0.0
    assert f["fair_value_p_up"] == 0.5
    assert math.isclose(f["yes_bid_residual"], f["yes_bid"] - 0.5)


def test_fair_value_above_half_when_btc_above_strike_with_vol():
    # BTC drifting upward + non-zero realized vol + positive moneyness ->
    # fair should be > 0.5.
    yes_book = _book("yes", [(0.55, 100.0)], [(0.57, 100.0)])
    no_book  = _book("no",  [(0.43, 100.0)], [(0.45, 100.0)])
    f = build_live_features(_snapshot(
        yes_book, no_book,
        btc_start=100_000.0, btc_end=100_500.0, strike=100_000.0, tte=200.0,
    )).features
    assert f["moneyness"] > 0
    assert f["btc_vol_30s"] > 0
    assert f["fair_value_p_up"] > 0.5
    # And residual should equal yes_bid minus fair.
    assert math.isclose(f["yes_bid_residual"], f["yes_bid"] - f["fair_value_p_up"])
    assert math.isclose(f["microprice_residual"], f["yes_microprice"] - f["fair_value_p_up"])


def test_fair_value_below_half_when_btc_below_strike():
    yes_book = _book("yes", [(0.45, 100.0)], [(0.47, 100.0)])
    no_book  = _book("no",  [(0.53, 100.0)], [(0.55, 100.0)])
    f = build_live_features(_snapshot(
        yes_book, no_book,
        btc_start=100_000.0, btc_end=99_500.0, strike=100_000.0, tte=200.0,
    )).features
    assert f["moneyness"] < 0
    assert f["fair_value_p_up"] < 0.5
