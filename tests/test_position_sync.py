"""Tests for position sync (strategy ↔ inventory) and paper trading fixes."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.api.client import PolymarketClient
from src.api.types import Position
from src.engine.execution import ExecutionTracker
from src.engine.inventory import InventoryState
from src.strategies.base import Strategy, Signal
from src.strategies.btc_updown import BTCUpDownStrategy, Bias


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MinimalStrategy(Strategy):
    """Concrete subclass for testing base-class sync behaviour."""

    def __init__(self):
        super().__init__(name="test", config={})
        self._outcome_map = {"tok_yes": "YES", "tok_no": "NO"}

    def analyze(self, market_data):
        return []

    def should_enter(self, signal):
        return True


def _make_paper_client() -> PolymarketClient:
    """Create a paper-trading client (no real API calls)."""
    return PolymarketClient(paper_trading=True)


# ===========================================================================
# Part A — sync_position_from_inventory
# ===========================================================================

class TestSyncPositionFromInventory:

    def test_sync_sets_strategy_position_from_inventory(self):
        """Inventory has a position → strategy state matches."""
        s = MinimalStrategy()
        s.sync_position_from_inventory("tok_yes", 50.0, 0.45)

        assert s.active_token_id == "tok_yes"
        assert s.entry_price == 0.45
        assert s.entry_size == 50.0
        assert s.entry_timestamp is not None

    def test_sync_clears_strategy_when_flat(self):
        """Inventory is zero → strategy resets to flat."""
        s = MinimalStrategy()
        # First, set a position
        s.active_token_id = "tok_yes"
        s.entry_price = 0.45
        s.entry_size = 50.0
        s.entry_timestamp = time.monotonic()

        # Sync with flat inventory (token_id=None means no position found)
        s.sync_position_from_inventory(None, 0.0, 0.0)

        assert s.active_token_id is None
        assert s.entry_price is None
        assert s.entry_size is None

    def test_sync_corrects_stale_token(self):
        """Strategy thinks it holds YES, inventory says NO → strategy updates."""
        s = MinimalStrategy()
        s.active_token_id = "tok_yes"
        s.entry_price = 0.45
        s.entry_size = 50.0

        s.sync_position_from_inventory("tok_no", 30.0, 0.55)

        assert s.active_token_id == "tok_no"
        assert s.entry_price == 0.55
        assert s.entry_size == 30.0

    def test_sync_noop_when_already_correct(self):
        """Strategy and inventory agree → no change to entry_timestamp."""
        s = MinimalStrategy()
        ts = time.monotonic() - 100
        s.active_token_id = "tok_yes"
        s.entry_price = 0.45
        s.entry_size = 50.0
        s.entry_timestamp = ts

        # Same token — should not overwrite
        s.sync_position_from_inventory("tok_yes", 50.0, 0.45)

        assert s.active_token_id == "tok_yes"
        assert s.entry_timestamp == ts  # unchanged

    def test_sync_clears_when_specific_token_goes_flat(self):
        """Strategy holds tok_yes, inventory says tok_yes is flat → reset."""
        s = MinimalStrategy()
        s.active_token_id = "tok_yes"
        s.entry_price = 0.45
        s.entry_size = 50.0
        s.entry_timestamp = time.monotonic()

        s.sync_position_from_inventory("tok_yes", 0.0, 0.0)

        assert s.active_token_id is None
        assert s.entry_price is None


# ===========================================================================
# Part C/D — Paper trading fixes
# ===========================================================================

class TestPaperTradingFixes:

    def test_paper_position_keyed_by_token_id(self):
        """Place paper order → get_positions() returns position findable by token_id."""
        client = _make_paper_client()
        client.place_order(
            market_id="mkt_1",
            token_id="tok_abc",
            outcome="YES",
            side="BUY",
            price=0.50,
            size=10.0,
        )
        # get_positions settles orders and returns positions
        positions = client.get_positions()
        by_token = {p.token_id: p for p in positions}

        assert "tok_abc" in by_token
        assert by_token["tok_abc"].size == 10.0
        assert by_token["tok_abc"].average_price == 0.50

    def test_paper_order_starts_pending(self):
        """Place paper order → order status is PENDING before settlement."""
        client = _make_paper_client()
        order = client.place_order(
            market_id="mkt_1",
            token_id="tok_abc",
            outcome="YES",
            side="BUY",
            price=0.50,
            size=10.0,
        )
        assert order.status == "PENDING"

    def test_paper_order_settles_on_get_positions(self):
        """Place paper order → call get_positions() → order is FILLED, position exists."""
        client = _make_paper_client()
        order = client.place_order(
            market_id="mkt_1",
            token_id="tok_abc",
            outcome="YES",
            side="BUY",
            price=0.50,
            size=10.0,
        )
        assert order.status == "PENDING"

        positions = client.get_positions()
        # Order should now be filled
        assert order.status == "FILLED"
        assert len(positions) == 1
        assert positions[0].token_id == "tok_abc"

    def test_paper_sell_reduces_position(self):
        """BUY then SELL → position reduced correctly."""
        client = _make_paper_client()
        client.place_order("mkt_1", "tok_abc", "YES", "BUY", 0.50, 10.0)
        client.get_positions()  # settle
        client.place_order("mkt_1", "tok_abc", "YES", "SELL", 0.55, 10.0)
        positions = client.get_positions()  # settle sell

        # Position should be gone (size <= 0)
        by_token = {p.token_id: p for p in positions}
        assert "tok_abc" not in by_token

    def test_paper_open_orders_empty_after_settle(self):
        """After settlement, get_open_orders returns nothing."""
        client = _make_paper_client()
        client.place_order("mkt_1", "tok_abc", "YES", "BUY", 0.50, 10.0)
        # get_open_orders settles first, then filters for PENDING
        open_orders = client.get_open_orders()
        assert len(open_orders) == 0


# ===========================================================================
# Integration: rollover does not orphan position
# ===========================================================================

class TestRolloverDoesNotOrphan:

    def test_rollover_sync_clears_strategy_for_old_tokens(self):
        """Hold position → rollover to new tokens → sync → strategy is flat."""
        s = BTCUpDownStrategy(config={"default_bias": "LONG"})
        s.set_tokens("mkt_1", "old_yes", "old_no")

        # Simulate holding a position
        s.active_token_id = "old_yes"
        s.entry_price = 0.50
        s.entry_size = 20.0
        s.entry_timestamp = time.monotonic()

        # Rollover: new tokens
        s.set_tokens("mkt_2", "new_yes", "new_no")

        # After rollover, set_tokens resets position state
        assert s.active_token_id is None

        # Sync with empty inventory for new tokens confirms flat
        s.sync_position_from_inventory(None, 0.0, 0.0)
        assert s.active_token_id is None
        assert s.entry_price is None

    def test_execution_tracker_cleanup_on_rollover(self):
        """Execution tracker state for old tokens is cleaned up on rollover."""
        tracker = ExecutionTracker()
        from src.api.types import Order
        from datetime import datetime

        # Simulate active order for old token
        old_order = Order(
            order_id="order_1",
            market_id="mkt_1",
            token_id="old_yes",
            outcome="YES",
            side="BUY",
            price=0.50,
            size=10.0,
            status="PENDING",
            timestamp=datetime.now(),
        )
        tracker.active_orders["order_1"] = old_order
        tracker._last_positions_by_token["old_yes"] = (10.0, 0.50)
        tracker._last_positions_by_token["old_no"] = (0.0, 0.0)

        # Simulate rollover cleanup (same logic as bot.py)
        for old_tid in ("old_yes", "old_no"):
            tracker.active_orders = {
                oid: o for oid, o in tracker.active_orders.items()
                if o.token_id != old_tid
            }
            tracker._last_positions_by_token.pop(old_tid, None)

        assert len(tracker.active_orders) == 0
        assert "old_yes" not in tracker._last_positions_by_token
        assert "old_no" not in tracker._last_positions_by_token
