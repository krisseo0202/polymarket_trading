"""Regression test for the tightened NO-side gate on XGBFBModel.

Background: the 2026-04-25 paper run (131 trades, 44% win rate, -$414
PnL) showed the model's NO predictions were anti-calibrated by ~0.19 in
the upper p_hat range. Only the bucket where p_hat ≤ 0.32 was profitable
on NO. We tightened ``XGBFBModel.thresholds["max_prob_yes_for_no"]``
from 0.46 → 0.32 to gate out the bleeding region.

This test locks in the new threshold and asserts the strategy honors it
when an XGBFBModel is wired in.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.api.types import OrderBook, OrderBookEntry
from src.models.prediction import PredictionResult
from src.models.xgb_fb_model import XGBFBModel
from src.strategies.logreg_edge import LogRegEdgeStrategy


def test_xgb_fb_threshold_locked_at_032():
    """The threshold change is the entire fix — keep it pinned."""
    assert XGBFBModel.thresholds.fget(XGBFBModel.__new__(XGBFBModel))["max_prob_yes_for_no"] == 0.32


def _make_book(token_id, bid, ask, size=1000.0):
    return OrderBook(
        market_id="market_1", token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=size)],
        asks=[OrderBookEntry(price=ask, size=size)],
        tick_size=0.01,
    )


def _market_data():
    """Setup that would WANT to bet NO if not for the threshold:
    market mid 0.50, model says p_yes=0.40 (so p_no=0.60, edge_no = 0.10)."""
    return {
        "order_books": {
            "yes_token": _make_book("yes_token", 0.49, 0.51),
            "no_token": _make_book("no_token", 0.49, 0.51),
        },
        "positions": [],
        "balance": 1000.0,
        "slot_expiry_ts": time.time() + 150.0,
        "question": "Bitcoin Up or Down $50,000",
        "strike_price": 50000.0,
    }


def _fake_xgb_model(p_yes: float):
    """A model that always returns the given p_yes and exposes the real
    XGBFBModel thresholds dict. We don't need a fitted booster because the
    strategy reads thresholds via ``model_service.thresholds``."""
    m = MagicMock()
    m.ready = True
    m.model_version = "fake-xgb"
    m.feature_names = []
    # Real XGBFBModel.thresholds is a property returning a dict copy.
    # The strategy code requires a real dict (rejects MagicMock thresholds).
    inst = XGBFBModel(model=object(), feature_names=["f"])
    m.thresholds = inst.thresholds
    m.predict.return_value = PredictionResult(
        prob_yes=p_yes, model_version="fake-xgb", feature_status="ready"
    )
    return m


def _strategy_with_model(model):
    config = {
        "yes_token_id": "yes_token", "no_token_id": "no_token",
        "market_id": "market_1", "delta": 0.01, "min_confidence": 0.0,
        "max_spread_pct": 0.10, "min_seconds_to_expiry": 30,
        "max_seconds_to_expiry": 280, "min_entry_price": 0.05,
        "max_entry_price": 0.95, "stop_loss_pct": 0.5, "profit_target_pct": 0.5,
        "max_hold_seconds": 280, "kelly_fraction": 0.10,
        "min_position_size_usdc": 5.0, "max_position_size_usdc": 50.0,
    }
    s = LogRegEdgeStrategy(config=config, btc_feed=None, model_service=model)
    s.set_tokens("market_1", "yes_token", "no_token")
    return s


def test_no_trade_blocked_when_p_hat_above_032():
    """p_hat=0.40 → would have traded NO under old 0.46 threshold; must skip now."""
    strategy = _strategy_with_model(_fake_xgb_model(p_yes=0.40))
    signals = strategy.analyze(_market_data())
    assert signals == []
    assert "max_prob_yes_for_no" in strategy.last_skip_reason


def test_no_trade_blocked_at_old_threshold_boundary():
    """p_hat=0.45 was inside the bleeding region (gap −0.21). Must skip."""
    strategy = _strategy_with_model(_fake_xgb_model(p_yes=0.45))
    signals = strategy.analyze(_market_data())
    assert signals == []


def test_no_trade_allowed_when_p_hat_below_032():
    """p_hat=0.25 → p_chosen=0.75. This is the profitable NO bucket; allow."""
    strategy = _strategy_with_model(_fake_xgb_model(p_yes=0.25))
    signals = strategy.analyze(_market_data())
    assert len(signals) == 1
    assert signals[0].outcome == "NO"


def test_yes_threshold_unchanged_at_054():
    """YES-side calibration was clean; min_prob_yes must stay at 0.54."""
    inst = XGBFBModel(model=object(), feature_names=["f"])
    assert inst.thresholds["min_prob_yes"] == 0.54
