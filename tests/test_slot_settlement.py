"""Tests for _settle_expiring_positions and _fetch_slot_outcome in bot.py."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.engine.slot_state import (
    settle_expiring_positions as _settle_expiring_positions,
    fetch_slot_outcome as _fetch_slot_outcome,
)
from src.engine.inventory import InventoryState
from src.engine.state_store import BotState


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_state() -> BotState:
    s = BotState()
    s.daily_realized_pnl = 0.0
    s.slot_realized_pnl = 0.0
    s.session_wins = 0
    s.session_losses = 0
    return s


def _make_inv(token_id: str, position: float, avg_cost: float) -> InventoryState:
    inv = InventoryState(token_id=token_id)
    inv.position = position
    inv.avg_cost = avg_cost
    return inv


def _make_risk_manager():
    rm = MagicMock()
    rm.record_trade = MagicMock()
    return rm


def _gamma_response(closed: bool, up_price: str, down_price: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [{
        "markets": [{
            "closed": closed,
            "outcomePrices": json.dumps([up_price, down_price]),
        }]
    }]
    return resp


_SLOT_TS = 1_699_999_800
_YES_TID  = "yes_token_abc"
_NO_TID   = "no_token_xyz"
_LOGGER   = MagicMock()


# ── _fetch_slot_outcome ───────────────────────────────────────────────────────

class TestFetchSlotOutcome:
    def test_returns_up_when_yes_resolves(self):
        resp = _gamma_response(closed=True, up_price="0.99", down_price="0.01")
        with patch("requests.get", return_value=resp):
            assert _fetch_slot_outcome(_SLOT_TS, _LOGGER) == "Up"

    def test_returns_down_when_no_resolves(self):
        resp = _gamma_response(closed=True, up_price="0.01", down_price="0.99")
        with patch("requests.get", return_value=resp):
            assert _fetch_slot_outcome(_SLOT_TS, _LOGGER) == "Down"

    def test_returns_none_when_not_closed(self):
        resp = _gamma_response(closed=False, up_price="0.50", down_price="0.50")
        with patch("requests.get", return_value=resp):
            assert _fetch_slot_outcome(_SLOT_TS, _LOGGER) is None

    def test_returns_none_on_api_exception(self):
        with patch("requests.get", side_effect=Exception("timeout")):
            assert _fetch_slot_outcome(_SLOT_TS, _LOGGER) is None

    def test_returns_none_when_no_events(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = []
        with patch("requests.get", return_value=resp):
            assert _fetch_slot_outcome(_SLOT_TS, _LOGGER) is None


# ── _settle_expiring_positions ────────────────────────────────────────────────

class TestSettleExpiringPositions:
    def _call(self, inventories, outcome, paper_trading=True):
        state = _make_state()
        rm = _make_risk_manager()
        resp = _gamma_response(
            closed=(outcome is not None),
            up_price="0.99" if outcome == "Up" else "0.01",
            down_price="0.99" if outcome == "Down" else "0.01",
        )
        with patch("requests.get", return_value=resp):
            _settle_expiring_positions(
                yes_token_id=_YES_TID,
                no_token_id=_NO_TID,
                slot_ts=_SLOT_TS,
                inventories=inventories,
                state=state,
                risk_manager=rm,
                paper_trading=paper_trading,
                logger=_LOGGER,
            )
        return state, inventories

    def test_yes_win_realizes_positive_pnl(self):
        inv = _make_inv(_YES_TID, position=100.0, avg_cost=0.29)
        state, _ = self._call({_YES_TID: inv, _NO_TID: _make_inv(_NO_TID, 0, 0)}, "Up")
        expected = (0.99 - 0.29) * 100.0
        assert state.daily_realized_pnl == pytest.approx(expected, abs=1e-6)
        assert state.slot_realized_pnl == pytest.approx(expected, abs=1e-6)
        assert state.session_wins == 1
        assert state.session_losses == 0

    def test_yes_loss_realizes_negative_pnl(self):
        inv = _make_inv(_YES_TID, position=100.0, avg_cost=0.65)
        state, _ = self._call({_YES_TID: inv, _NO_TID: _make_inv(_NO_TID, 0, 0)}, "Down")
        expected = (0.01 - 0.65) * 100.0
        assert state.daily_realized_pnl == pytest.approx(expected, abs=1e-6)
        assert state.session_losses == 1
        assert state.session_wins == 0

    def test_no_position_skips_api_call(self):
        invs = {
            _YES_TID: _make_inv(_YES_TID, 0, 0),
            _NO_TID:  _make_inv(_NO_TID, 0, 0),
        }
        with patch("requests.get") as mock_get:
            _settle_expiring_positions(
                _YES_TID, _NO_TID, _SLOT_TS, invs,
                _make_state(), _make_risk_manager(), True, _LOGGER,
            )
        mock_get.assert_not_called()

    def test_unresolved_outcome_leaves_positions_unchanged(self):
        inv = _make_inv(_YES_TID, position=100.0, avg_cost=0.29)
        invs = {_YES_TID: inv, _NO_TID: _make_inv(_NO_TID, 0, 0)}
        state = _make_state()
        resp = _gamma_response(closed=False, up_price="0.50", down_price="0.50")
        with patch("requests.get", return_value=resp):
            _settle_expiring_positions(
                _YES_TID, _NO_TID, _SLOT_TS, invs,
                state, _make_risk_manager(), True, _LOGGER,
            )
        assert state.slot_realized_pnl == 0.0
        assert invs[_YES_TID].position == 100.0  # unchanged

    def test_both_tokens_settled(self):
        """Edge case: positions in both YES and NO (shouldn't happen but handle safely)."""
        yes_inv = _make_inv(_YES_TID, position=50.0, avg_cost=0.40)
        no_inv  = _make_inv(_NO_TID,  position=30.0, avg_cost=0.55)
        invs = {_YES_TID: yes_inv, _NO_TID: no_inv}
        state, _ = self._call(invs, "Up")
        # YES wins: (0.99 - 0.40) * 50 = +29.5
        # NO loses: (0.01 - 0.55) * 30 = -16.2
        # Total = +13.3
        assert state.daily_realized_pnl == pytest.approx((0.99 - 0.40) * 50 + (0.01 - 0.55) * 30, abs=1e-5)
        assert state.session_wins == 1
        assert state.session_losses == 1

    def test_slot_pnl_nonzero_before_reset(self):
        inv = _make_inv(_YES_TID, position=103.5, avg_cost=0.29)
        state, _ = self._call({_YES_TID: inv, _NO_TID: _make_inv(_NO_TID, 0, 0)}, "Up")
        # slot_realized_pnl should be positive (72.45) BEFORE the caller resets it to 0
        assert state.slot_realized_pnl > 0

    def test_position_zeroed_after_settlement(self):
        inv = _make_inv(_YES_TID, position=100.0, avg_cost=0.29)
        invs = {_YES_TID: inv}
        state, result_invs = self._call(invs, "Up")
        assert result_invs[_YES_TID].position == pytest.approx(0.0, abs=1e-9)

    def test_gamma_api_failure_no_crash(self):
        inv = _make_inv(_YES_TID, position=100.0, avg_cost=0.29)
        invs = {_YES_TID: inv}
        state = _make_state()
        with patch("requests.get", side_effect=Exception("network error")):
            _settle_expiring_positions(
                _YES_TID, _NO_TID, _SLOT_TS, invs,
                state, _make_risk_manager(), True, _LOGGER,
            )
        # No crash, no fill applied
        assert state.slot_realized_pnl == 0.0
        assert invs[_YES_TID].position == 100.0
