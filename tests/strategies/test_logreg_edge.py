"""Tests for LogRegEdgeStrategy."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.types import OrderBook, OrderBookEntry, Position
from src.models.logreg_model import LogRegModel, LR_FEATURES
from src.models.xgb_model import PredictionResult
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
    model = _mock_model(prob_yes=0.70)
    strategy = _make_strategy(model, delta=0.05)
    data = _market_data(yes_bid=0.49, yes_ask=0.51, no_bid=0.49, no_ask=0.51)

    # First call: enters
    signals = strategy.analyze(data)
    assert len(signals) == 1

    # Second call: already holding → no signals
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
