"""Tests for Family D (multiplicative interaction) features."""

import time

from src.api.types import OrderBook, OrderBookEntry
from src.models.feature_builder import build_live_features
from src.models.schema import FEATURE_COLUMNS


def _book(token_id: str, bid: float, ask: float, size: float = 100.0) -> OrderBook:
    return OrderBook(
        market_id="mkt",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=size)],
        asks=[OrderBookEntry(price=ask, size=size)],
        tick_size=0.001,
    )


def _snapshot(
    btc_start: float = 100_000.0,
    btc_end: float = 100_050.0,
    strike: float = 100_000.0,
    tte: float = 200.0,
    slot_path_features: dict | None = None,
) -> dict:
    now = time.time()
    # 70 seconds of BTC history interpolating from start to end so
    # btc_vol_30s is non-trivial and moneyness reflects btc_end vs strike.
    prices = []
    for idx in range(71):
        frac = idx / 70
        price = btc_start + frac * (btc_end - btc_start)
        prices.append((now - 70 + idx, price))
    return {
        "btc_prices": prices,
        "yes_book": _book("yes", 0.55, 0.57),
        "no_book": _book("no", 0.43, 0.45),
        "yes_history": [],
        "no_history": [],
        "question": f"Will BTC be above ${strike:,.0f} at 12:05 PM ET?",
        "slot_expiry_ts": now + tte,
        "now_ts": now,
        "slot_path_features": slot_path_features,
    }


def test_interaction_columns_in_schema():
    for name in ("moneyness_x_tte", "microprice_x_tte", "strike_crosses_x_vol"):
        assert name in FEATURE_COLUMNS


def test_moneyness_x_tte_matches_product():
    result = build_live_features(_snapshot(btc_end=100_080.0, tte=250.0))
    f = result.features
    assert f["moneyness_x_tte"] == f["moneyness"] * f["seconds_to_expiry"]
    assert f["moneyness"] > 0  # sanity: BTC above strike
    assert f["moneyness_x_tte"] > 0


def test_microprice_x_tte_matches_product():
    result = build_live_features(_snapshot(tte=150.0))
    f = result.features
    assert f["microprice_x_tte"] == f["yes_microprice"] * f["seconds_to_expiry"]
    assert f["yes_microprice"] > 0


def test_strike_crosses_x_vol_uses_family_c_state():
    # With no slot_path_features supplied, crosses defaults to 0 -> product is 0.
    result_zero = build_live_features(_snapshot())
    assert result_zero.features["strike_crosses_x_vol"] == 0.0

    # When Family C supplies crosses, the interaction equals crosses * vol_30s.
    slot_state = {"slot_strike_crosses": 7.0}
    result = build_live_features(_snapshot(slot_path_features=slot_state))
    f = result.features
    assert f["slot_strike_crosses"] == 7.0
    assert f["strike_crosses_x_vol"] == f["slot_strike_crosses"] * f["btc_vol_30s"]


def test_interactions_zero_when_inputs_zero():
    # Flat price history -> vol_30s = 0, moneyness = 0 -> all interactions 0.
    snap = _snapshot(btc_start=100_000.0, btc_end=100_000.0, strike=100_000.0)
    f = build_live_features(snap).features
    assert f["btc_vol_30s"] == 0.0
    assert f["moneyness"] == 0.0
    assert f["moneyness_x_tte"] == 0.0
    assert f["strike_crosses_x_vol"] == 0.0
