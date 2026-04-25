"""Regression test for the phantom-position-state self-heal.

Bug: ``LogRegEdgeStrategy._check_entry`` optimistically sets
``active_token_id``, ``entry_price``, ``entry_size``, ``entry_timestamp``
*before* the signal is sent to ``execute_signals``. If the order is
rejected downstream (disabled strategy, risk-manager clip-to-zero,
exception during ``place_order``), no fill ever lands. The strategy keeps
the in-flight state forever, blocking all future entries on
``if self.active_token_id is not None: return []`` until a slot rollover.

Fix: ``_auto_recover_position`` now reconciles in both directions —
sets state from inventory if missing, AND clears state if it's set but
inventory doesn't back it up. This test locks in the clear path.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from src.api.types import OrderBook, OrderBookEntry, Position
from src.models.logreg_model import LogRegModel
from src.models.prediction import PredictionResult
from src.strategies.logreg_edge import LogRegEdgeStrategy


def _mock_model(prob_yes: float):
    class _M(LogRegModel):
        def __init__(self):
            pass
        ready = True
        model_version = "test"
        feature_names = []
        thresholds = {"min_edge": 0.05, "min_prob_yes": 0.55, "max_prob_yes_for_no": 0.45,
                      "max_spread_pct": 0.05, "exit_edge": -0.01,
                      "min_seconds_to_expiry": 30, "max_seconds_to_expiry": 280}

        def predict(self, snapshot):
            return PredictionResult(prob_yes=prob_yes, model_version="test", feature_status="ready")

    return _M()


def _make_book(token_id, bid, ask, size=1000.0):
    return OrderBook(
        market_id="market_1",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=size)],
        asks=[OrderBookEntry(price=ask, size=size)],
        tick_size=0.01,
    )


def _market_data(positions=None):
    return {
        "order_books": {
            "yes_token": _make_book("yes_token", 0.49, 0.51),
            "no_token": _make_book("no_token", 0.49, 0.51),
        },
        "positions": positions or [],
        "balance": 1000.0,
        "slot_expiry_ts": time.time() + 150.0,
        "question": "Bitcoin Up or Down $50,000",
        "strike_price": 50000.0,
    }


def _make_strategy(model):
    config = {
        "yes_token_id": "yes_token", "no_token_id": "no_token",
        "market_id": "market_1", "delta": 0.05, "min_confidence": 0.0,
        "max_spread_pct": 0.05, "min_seconds_to_expiry": 30,
        "max_seconds_to_expiry": 280, "min_entry_price": 0.05,
        "max_entry_price": 0.95, "stop_loss_pct": 0.5, "profit_target_pct": 0.5,
        "max_hold_seconds": 280, "kelly_fraction": 0.10,
        "min_position_size_usdc": 5.0, "max_position_size_usdc": 50.0,
    }
    s = LogRegEdgeStrategy(config=config, btc_feed=None, model_service=model)
    s.set_tokens("market_1", "yes_token", "no_token")
    return s


def test_phantom_position_self_heals_when_order_was_never_filled():
    """If state was set but inventory is empty, next analyze() must clear it."""
    model = _mock_model(prob_yes=0.70)
    strategy = _make_strategy(model)

    # Cycle 1: strategy decides BUY → optimistically sets state.
    signals = strategy.analyze(_market_data(positions=[]))
    assert len(signals) == 1
    assert strategy.active_token_id is not None  # in-flight guard set

    # Simulate: order rejected downstream (disabled, risk clip, exception).
    # Inventory NEVER updates. Phantom state would block forever without
    # the reconciliation fix.

    # Cycle 2: empty positions again. Reconciliation must clear the phantom
    # so entry can re-evaluate. Without the fix, the strategy would short-
    # circuit on `if self.active_token_id is not None: return []` and emit
    # zero signals. With the fix, the phantom clears and a new signal fires.
    signals = strategy.analyze(_market_data(positions=[]))
    assert len(signals) == 1, (
        "Phantom state must self-heal so re-entry is possible. "
        "Got 0 signals — phantom block still in effect."
    )


def test_real_position_is_preserved_when_inventory_backs_it_up():
    """If state is set AND inventory shows the position, do not clear."""
    model = _mock_model(prob_yes=0.70)
    strategy = _make_strategy(model)

    # Cycle 1: enter.
    signals = strategy.analyze(_market_data(positions=[]))
    assert len(signals) == 1
    sig = signals[0]
    token_id = "yes_token" if sig.outcome == "YES" else "no_token"

    # Simulate: order filled. Inventory now holds the position.
    real_pos = Position(
        market_id="market_1", token_id=token_id, outcome=sig.outcome,
        size=sig.size, average_price=sig.price,
    )

    # Cycle 2: inventory backs the state → must NOT clear.
    signals = strategy.analyze(_market_data(positions=[real_pos]))
    assert strategy.active_token_id == token_id
    assert len(signals) == 0  # in_position → no new entries


def test_state_synced_from_inventory_when_strategy_is_unaware():
    """Bot restart scenario: state is None but inventory has a position."""
    model = _mock_model(prob_yes=0.70)
    strategy = _make_strategy(model)

    # Strategy starts fresh (post-restart). Inventory has a leftover position.
    leftover = Position(
        market_id="market_1", token_id="yes_token", outcome="YES",
        size=10.0, average_price=0.40,
    )
    assert strategy.active_token_id is None

    signals = strategy.analyze(_market_data(positions=[leftover]))

    # _auto_recover_position should have populated state from inventory.
    assert strategy.active_token_id == "yes_token"
    assert strategy.entry_price == 0.40
    assert strategy.entry_size == 10.0
    # And the strategy is now "in_position" → no new signals.
    assert len(signals) == 0
