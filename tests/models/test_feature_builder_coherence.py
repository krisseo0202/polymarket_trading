"""Tests for Family B (YES/NO coherence) features."""

import time

import pytest

from src.api.types import OrderBook, OrderBookEntry
from src.models.feature_builder import build_live_features


def _multi_book(token_id: str, bids: list, asks: list) -> OrderBook:
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


def test_mid_sum_residual_positive_when_overpriced():
    # YES mid = 0.53, NO mid = 0.50 -> sum = 1.03 -> residual = +0.03
    yes_book = _multi_book("yes", [(0.52, 100.0)], [(0.54, 100.0)])
    no_book = _multi_book("no", [(0.49, 100.0)], [(0.51, 100.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    assert result.features["mid_sum_residual"] == pytest.approx(0.03, abs=1e-9)
    assert result.features["mid_sum_residual_abs"] == pytest.approx(0.03, abs=1e-9)


def test_mid_sum_residual_negative_when_underpriced():
    # YES mid = 0.45, NO mid = 0.50 -> sum = 0.95 -> residual = -0.05
    yes_book = _multi_book("yes", [(0.44, 100.0)], [(0.46, 100.0)])
    no_book = _multi_book("no", [(0.49, 100.0)], [(0.51, 100.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    assert result.features["mid_sum_residual"] == pytest.approx(-0.05, abs=1e-9)
    assert result.features["mid_sum_residual_abs"] == pytest.approx(0.05, abs=1e-9)


def test_spread_asymmetry_sign_matches_wider_side():
    # YES spread 0.04, NO spread 0.02 -> (0.04 - 0.02)/0.06 = +0.333
    yes_book = _multi_book("yes", [(0.48, 100.0)], [(0.52, 100.0)])
    no_book = _multi_book("no", [(0.49, 100.0)], [(0.51, 100.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    assert result.features["spread_asymmetry"] == pytest.approx((0.04 - 0.02) / 0.06)
    # NO side wider should flip the sign.
    flipped = build_live_features(_snapshot_with_books(no_book, yes_book))
    assert flipped.features["spread_asymmetry"] == pytest.approx((0.02 - 0.04) / 0.06)


def test_depth_asymmetry_bounded_and_signed():
    # YES total = 300, NO total = 100 -> (300-100)/400 = +0.5
    yes_book = _multi_book("yes", [(0.52, 100.0), (0.51, 50.0)], [(0.54, 100.0), (0.55, 50.0)])
    no_book = _multi_book("no", [(0.48, 50.0)], [(0.50, 50.0)])

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    assert result.features["depth_asymmetry"] == pytest.approx(0.5)
    assert -1.0 <= result.features["depth_asymmetry"] <= 1.0


def test_coherence_defaults_to_zero_when_one_side_empty():
    yes_book = _multi_book("yes", [(0.52, 100.0)], [(0.54, 100.0)])
    no_book = OrderBook(market_id="", token_id="no", bids=[], asks=[], tick_size=0.001)

    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    # no_mid == 0 (empty book), so residual features stay at default 0.0.
    assert result.features["mid_sum_residual"] == 0.0
    assert result.features["mid_sum_residual_abs"] == 0.0
    # Only YES has spread, so spread_asymmetry == +1.0.
    assert result.features["spread_asymmetry"] == pytest.approx(1.0)
    # Only YES has depth, so depth_asymmetry == +1.0.
    assert result.features["depth_asymmetry"] == pytest.approx(1.0)


def test_slot_path_features_flow_through_feature_builder():
    yes_book = _multi_book("yes", [(0.52, 100.0)], [(0.54, 100.0)])
    no_book = _multi_book("no", [(0.48, 100.0)], [(0.50, 100.0)])
    snapshot = _snapshot_with_books(yes_book, no_book)
    snapshot["slot_path_features"] = {
        "slot_high_excursion_bps": 42.5,
        "slot_low_excursion_bps": -17.0,
        "slot_drift_bps": 3.25,
        "slot_time_above_strike_pct": 0.8,
        "slot_strike_crosses": 2.0,
    }

    result = build_live_features(snapshot)

    assert result.features["slot_high_excursion_bps"] == 42.5
    assert result.features["slot_low_excursion_bps"] == -17.0
    assert result.features["slot_drift_bps"] == 3.25
    assert result.features["slot_time_above_strike_pct"] == 0.8
    assert result.features["slot_strike_crosses"] == 2.0


def test_slot_path_features_default_to_zero_when_absent():
    yes_book = _multi_book("yes", [(0.52, 100.0)], [(0.54, 100.0)])
    no_book = _multi_book("no", [(0.48, 100.0)], [(0.50, 100.0)])
    result = build_live_features(_snapshot_with_books(yes_book, no_book))

    assert result.features["slot_high_excursion_bps"] == 0.0
    assert result.features["slot_drift_bps"] == 0.0
    assert result.features["slot_strike_crosses"] == 0.0


def test_schema_contains_family_b_columns():
    from src.models.schema import FEATURE_COLUMNS

    for col in (
        "mid_sum_residual",
        "mid_sum_residual_abs",
        "spread_asymmetry",
        "depth_asymmetry",
    ):
        assert col in FEATURE_COLUMNS
