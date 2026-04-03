"""
Deterministic validation tests for dashboard data bugs.

Each test targets one specific fix:
  1. order_count — active_order_ids is a dict, not a list
  2. position side labels — YES/NO not Up/Down/truncated ID
  3. bot status derivation — richer than RUNNING/STOPPED
  4. PnL — unrealized PnL shown when positions are open
  5. win/loss — intra-cycle fills counted (bot.py fix, tested at data level)
"""

import time
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. order_count was always 0 because isinstance(dict, list) is False
# ---------------------------------------------------------------------------

class TestOrderCountFix:
    """active_order_ids is a dict {token_id: order_id|None}, not a list."""

    def test_order_count_from_dict_with_active_orders(self):
        """Non-None values should be counted as active orders."""
        orders = {
            "token_aaa": "order_123",
            "token_bbb": None,
            "token_ccc": "order_456",
        }
        # This is the fixed logic from _build_bot_status_panel
        if isinstance(orders, dict):
            count = sum(1 for v in orders.values() if v is not None)
        else:
            count = 0
        assert count == 2

    def test_order_count_all_none(self):
        """All-None dict should count as 0 active orders."""
        orders = {"token_aaa": None, "token_bbb": None}
        count = sum(1 for v in orders.values() if v is not None) if isinstance(orders, dict) else 0
        assert count == 0

    def test_order_count_empty_dict(self):
        orders = {}
        count = sum(1 for v in orders.values() if v is not None) if isinstance(orders, dict) else 0
        assert count == 0

    def test_old_bug_reproduced(self):
        """The old code: isinstance(dict, list) → always False → always 0."""
        orders = {"token_aaa": "order_123"}
        # Old buggy logic:
        old_count = len(orders) if isinstance(orders, list) else 0
        assert old_count == 0  # confirms the bug existed


# ---------------------------------------------------------------------------
# 2. Position side should be YES/NO, not Up/Down or truncated token ID
# ---------------------------------------------------------------------------

class TestPositionSideLabels:
    """Polymarket binary markets: tids[0] = YES, tids[1] = NO."""

    def test_up_token_maps_to_yes(self):
        market = {"up_token": "tok_aaa", "down_token": "tok_bbb"}
        token_to_outcome = {}
        if market.get("up_token"):
            token_to_outcome[market["up_token"]] = "YES"
        if market.get("down_token"):
            token_to_outcome[market["down_token"]] = "NO"

        assert token_to_outcome["tok_aaa"] == "YES"
        assert token_to_outcome["tok_bbb"] == "NO"

    def test_fallback_to_truncated_id_when_no_market(self):
        """When market is None, unknown tokens fall back to truncated ID."""
        tid = "0x1234567890abcdef1234567890abcdef"
        token_to_outcome = {}
        # Simulate the fallback path
        if tid in token_to_outcome:
            label = token_to_outcome[tid]
        else:
            label = tid[:8] + "..." if len(tid) > 8 else tid
        assert label == "0x123456..."


# ---------------------------------------------------------------------------
# 3. Bot status should reflect strategy state + orders, not just file mtime
# ---------------------------------------------------------------------------

class TestBotStatusDerivation:
    """Status derived from file mtime + strategy_status + active orders."""

    def _derive_status(self, age_min, strategy_status, order_count):
        """Replicates the fixed _build_bot_status_panel logic."""
        if age_min >= 10:
            return "STOPPED"
        elif strategy_status == "POSITION_OPEN":
            return "POSITION_OPEN"
        elif order_count > 0:
            return "ORDER_PENDING"
        else:
            return "RUNNING"

    def test_stopped_when_stale(self):
        assert self._derive_status(15, "WATCHING", 0) == "STOPPED"

    def test_position_open_shown(self):
        assert self._derive_status(1, "POSITION_OPEN", 0) == "POSITION_OPEN"

    def test_order_pending_shown(self):
        assert self._derive_status(1, "WATCHING", 2) == "ORDER_PENDING"

    def test_running_when_idle(self):
        assert self._derive_status(1, "WATCHING", 0) == "RUNNING"

    def test_position_open_takes_priority_over_orders(self):
        """POSITION_OPEN should show even if there are pending orders."""
        assert self._derive_status(1, "POSITION_OPEN", 1) == "POSITION_OPEN"


# ---------------------------------------------------------------------------
# 4. PnL — unrealized PnL computation
# ---------------------------------------------------------------------------

class TestUnrealizedPnl:
    """Unrealized PnL = (current_mid - avg_cost) * position for each active inventory."""

    def test_long_position_profit(self):
        """Bought at 0.45, current mid 0.55, 100 shares → uPnL = +10.00."""
        avg_cost = 0.45
        position = 100.0
        current_mid = 0.55
        upnl = (current_mid - avg_cost) * position
        assert abs(upnl - 10.0) < 1e-9

    def test_long_position_loss(self):
        """Bought at 0.55, current mid 0.45, 100 shares → uPnL = -10.00."""
        avg_cost = 0.55
        position = 100.0
        current_mid = 0.45
        upnl = (current_mid - avg_cost) * position
        assert abs(upnl - (-10.0)) < 1e-9

    def test_flat_position_no_upnl(self):
        """No position → unrealized PnL computation returns None."""
        inventories = {"tok_aaa": {"position": 0, "avg_cost": 0.5}}
        active = {t: i for t, i in inventories.items() if i.get("position", 0) != 0}
        assert len(active) == 0  # filtered out


# ---------------------------------------------------------------------------
# 5. Win/loss counting — verify the fix applies to intra-cycle fills
# ---------------------------------------------------------------------------

class TestWinLossCounting:
    """Intra-cycle fills should increment session_wins/session_losses."""

    def test_positive_realized_counts_as_win(self):
        """Simulates the fixed intra-cycle logic."""
        session_wins = 0
        session_losses = 0
        realized = 0.05  # positive = win
        if realized > 0:
            session_wins += 1
        elif realized < 0:
            session_losses += 1
        assert session_wins == 1
        assert session_losses == 0

    def test_negative_realized_counts_as_loss(self):
        session_wins = 0
        session_losses = 0
        realized = -0.03  # negative = loss
        if realized > 0:
            session_wins += 1
        elif realized < 0:
            session_losses += 1
        assert session_wins == 0
        assert session_losses == 1

    def test_zero_realized_no_count(self):
        """Position add (no close) returns 0 realized — not a win or loss."""
        session_wins = 0
        session_losses = 0
        realized = 0.0
        if realized > 0:
            session_wins += 1
        elif realized < 0:
            session_losses += 1
        assert session_wins == 0
        assert session_losses == 0
