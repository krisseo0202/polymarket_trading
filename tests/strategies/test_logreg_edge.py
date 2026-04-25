"""Tests for LogRegEdgeStrategy."""

import time
from unittest.mock import MagicMock


from src.api.types import OrderBook, OrderBookEntry
from src.models.logreg_model import LogRegModel
from src.models.prediction import PredictionResult
from src.strategies.logreg_edge import LogRegEdgeStrategy


def _make_book(token_id, bid, ask):
    return OrderBook(
        market_id="test",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=100.0)],
        asks=[OrderBookEntry(price=ask, size=100.0)],
        tick_size=0.01,
    )


def _mock_model(prob_yes: float):
    """Create a mock model that always returns the given prob_yes."""
    model = MagicMock(spec=LogRegModel)
    model.ready = True
    model.model_version = "test_v1"
    model.predict.return_value = PredictionResult(
        prob_yes=prob_yes,
        model_version="test_v1",
        feature_status="ready",
    )
    return model


def _make_strategy(model, delta=0.05, balance=1000.0):
    config = {
        "delta": delta,
        "position_size_usdc": 30.0,
        "kelly_fraction": 0.15,
        "max_spread_pct": 0.10,
        "min_seconds_to_expiry": 10.0,
        "max_seconds_to_expiry": 290.0,
    }
    strategy = LogRegEdgeStrategy(config=config, model_service=model)
    strategy.set_tokens("market_1", "yes_token", "no_token")
    return strategy


def _market_data(yes_bid, yes_ask, no_bid, no_ask, tte=150.0):
    now = time.time()
    return {
        "order_books": {
            "yes_token": _make_book("yes_token", yes_bid, yes_ask),
            "no_token": _make_book("no_token", no_bid, no_ask),
        },
        "positions": [],
        "balance": 1000.0,
        "slot_expiry_ts": now + tte,
        "question": "Bitcoin Up or Down $50,000",
        "strike_price": 50000.0,
    }


def test_buy_yes_when_model_thinks_up():
    """Model predicts high P(Up), market prices Up low → BUY YES."""
    model = _mock_model(prob_yes=0.70)
    strategy = _make_strategy(model, delta=0.05)

    # Market: yes_mid = 0.50, c_t = 0.01
    # edge_yes = 0.70 - 0.50 - 0.01 = 0.19 > delta=0.05 → BUY YES
    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51)
    signals = strategy.analyze(data)

    assert len(signals) == 1
    assert signals[0].outcome == "YES"
    assert signals[0].action == "BUY"
    assert "logreg" in signals[0].reason
    assert "edge_yes" in signals[0].reason


def test_buy_no_when_model_thinks_down():
    """Model predicts low P(Up), market prices Up high → BUY NO."""
    model = _mock_model(prob_yes=0.30)
    strategy = _make_strategy(model, delta=0.05)

    # Market: yes_mid = 0.50, c_t = 0.01
    # edge_no = 0.50 - 0.30 - 0.01 = 0.19 > delta=0.05 → BUY NO
    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51)
    signals = strategy.analyze(data)

    assert len(signals) == 1
    assert signals[0].outcome == "NO"
    assert signals[0].action == "BUY"


def test_no_trade_when_edge_below_delta():
    """Model agrees with market — no trade."""
    model = _mock_model(prob_yes=0.51)
    strategy = _make_strategy(model, delta=0.05)

    # edge_yes = 0.51 - 0.50 - 0.01 = 0.00, edge_no = 0.50 - 0.51 - 0.01 = -0.02
    # Neither exceeds delta=0.05
    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51)
    signals = strategy.analyze(data)
    assert len(signals) == 0


def test_no_trade_when_already_in_position():
    """Once holding, no new signals (hold to expiry)."""
    from src.api.types import Position

    model = _mock_model(prob_yes=0.70)
    strategy = _make_strategy(model, delta=0.05)
    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51)

    # First call: enters
    signals = strategy.analyze(data)
    assert len(signals) == 1
    sig = signals[0]

    # Simulate the fill — populate `positions` with what would be there after
    # the order lands. Without this the strategy's _auto_recover_position
    # correctly clears the optimistic in-flight state and re-evaluates entry.
    token_id = "yes_token" if sig.outcome == "YES" else "no_token"
    data["positions"] = [
        Position(
            market_id=strategy._market_id,
            token_id=token_id,
            outcome=sig.outcome,
            size=sig.size,
            average_price=sig.price,
        )
    ]

    # Second call: already holding (per inventory) → no signals
    signals = strategy.analyze(data)
    assert len(signals) == 0


def test_no_trade_near_expiry():
    """Skip when too close to expiry."""
    model = _mock_model(prob_yes=0.80)
    strategy = _make_strategy(model, delta=0.05)

    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51, tte=5.0)
    signals = strategy.analyze(data)
    assert len(signals) == 0


def test_no_trade_when_model_not_ready():
    """No signals when model isn't loaded."""
    model = MagicMock(spec=LogRegModel)
    model.ready = False
    strategy = _make_strategy(model, delta=0.05)

    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51)
    signals = strategy.analyze(data)
    assert len(signals) == 0


def test_observable_state():
    """Check that observable state is populated after analyze."""
    model = _mock_model(prob_yes=0.70)
    strategy = _make_strategy(model, delta=0.05)
    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51)

    strategy.analyze(data)

    assert strategy.last_prob_yes == 0.70
    assert strategy.last_q_t is not None
    assert strategy.last_c_t is not None
    assert strategy.last_edge_yes is not None
    assert strategy.last_edge_no is not None
    assert strategy.last_tte_seconds is not None


def test_wide_spread_rejected():
    """Wide spread exceeding max_spread_pct should block entry."""
    model = _mock_model(prob_yes=0.80)
    strategy = _make_strategy(model, delta=0.05)
    # max_spread_pct = 0.10, spread here = 0.20/0.40 = 50%
    data = _market_data(yes_bid=0.20, yes_ask=0.40, no_bid=0.60, no_ask=0.80)
    signals = strategy.analyze(data)
    assert len(signals) == 0


# ──────────────────────────────────────────────────────────────────────────
# Kelly sizing — clip(kelly_notional, min, max) when edge > 0
# ──────────────────────────────────────────────────────────────────────────
#
# Regression context: an earlier rework briefly removed the strategy-level
# USDC ceiling entirely, on the (wrong) assumption that RiskManager's size
# check would catch unbounded Kelly. RiskManager's size check is share-
# denominated, so the ceiling must live here. These tests pin that.


def _sizing_strategy(
    kelly_fraction: float = 0.10,
    min_position_size_usdc: float = 10.0,
    max_position_size_usdc: float = 50.0,
    legacy_cap: float = None,
) -> LogRegEdgeStrategy:
    """Strategy wired only for _kelly_size() exercise — no market data."""
    model = _mock_model(prob_yes=0.60)
    config = {
        "delta": 0.03,
        "kelly_fraction": kelly_fraction,
        "min_position_size_usdc": min_position_size_usdc,
        "max_position_size_usdc": max_position_size_usdc,
        "max_spread_pct": 0.10,
        "min_seconds_to_expiry": 10.0,
        "max_seconds_to_expiry": 290.0,
    }
    if legacy_cap is not None:
        config.pop("max_position_size_usdc")
        config["position_size_usdc"] = legacy_cap
    return LogRegEdgeStrategy(config=config, model_service=model)


def test_kelly_size_no_edge_returns_zero():
    """prob <= price → zero-Kelly → return 0, no floor applied."""
    s = _sizing_strategy()
    assert s._kelly_size(prob=0.50, price=0.55, balance=1000.0) == 0.0
    assert s._kelly_size(prob=0.55, price=0.55, balance=1000.0) == 0.0


def test_kelly_size_zero_balance_returns_floor():
    """Unknown/zero balance → fall back to the configured floor."""
    s = _sizing_strategy(min_position_size_usdc=10.0)
    # balance=0 hits the early return before Kelly math
    assert s._kelly_size(prob=0.60, price=0.55, balance=0.0) == 10.0


def test_kelly_size_kelly_above_max_returns_max():
    """Huge edge + big balance → Kelly exceeds ceiling → clamped to max."""
    s = _sizing_strategy(max_position_size_usdc=50.0)
    # f_full = 0.15/0.45 = 0.333; f_used = 0.1 * 0.333 = 0.0333
    # usdc = 0.0333 * 5000 = $166.67 → clamped to 50
    result = s._kelly_size(prob=0.70, price=0.55, balance=5000.0)
    assert result == 50.0


def test_kelly_size_kelly_below_min_returns_floor():
    """REGRESSION: small edge + small balance → Kelly below floor → floor."""
    s = _sizing_strategy(min_position_size_usdc=10.0)
    # f_full = 0.03/0.45 = 0.0667; f_used = 0.1 * 0.0667 = 0.00667
    # usdc = 0.00667 * 300 = $2.00 → floored to 10
    result = s._kelly_size(prob=0.58, price=0.55, balance=300.0)
    assert result == 10.0


def test_kelly_size_kelly_in_band_returns_kelly():
    """Mid-edge, mid-balance → raw Kelly lands inside band → unchanged."""
    s = _sizing_strategy(min_position_size_usdc=10.0, max_position_size_usdc=50.0)
    # f_full = 0.05/0.45 = 0.111; f_used = 0.1 * 0.111 = 0.01111
    # usdc = 0.01111 * 1400 = $15.56 → in band
    result = s._kelly_size(prob=0.60, price=0.55, balance=1400.0)
    assert 15.0 < result < 16.0
    # Importantly, NOT clamped to the min or max.
    assert result != 10.0
    assert result != 50.0


def test_kelly_size_legacy_config_without_min_returns_capped_kelly():
    """Backcompat: old config with only position_size_usdc still caps at max,
    and min==0 means the floor is skipped (raw Kelly wins)."""
    s = _sizing_strategy(
        min_position_size_usdc=0.0,
        max_position_size_usdc=None,  # placeholder, overridden below
        legacy_cap=30.0,
    )
    # Ceiling = $30 (from legacy key), no floor
    # f_full = 0.15/0.45 = 0.333; f_used = 0.1 * 0.333 = 0.0333
    # usdc = 0.0333 * 1000 = $33.33 → capped at 30
    result = s._kelly_size(prob=0.70, price=0.55, balance=1000.0)
    assert result == 30.0

    # Tiny Kelly → NOT floored (min=0) — stays at the raw value
    small = s._kelly_size(prob=0.58, price=0.55, balance=100.0)
    # $0.667 raw, no floor → returned as-is
    assert 0.5 < small < 1.0


def test_kelly_size_invalid_price_returns_zero():
    """Edge case: price <= 0 or price >= 1 is malformed → 0."""
    s = _sizing_strategy()
    assert s._kelly_size(prob=0.60, price=0.0, balance=1000.0) == 0.0
    assert s._kelly_size(prob=0.60, price=1.0, balance=1000.0) == 0.0
    assert s._kelly_size(prob=0.60, price=-0.1, balance=1000.0) == 0.0
