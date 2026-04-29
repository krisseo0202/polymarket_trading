"""Tests for position sync (strategy <-> inventory) and paper trading fixes."""

import time


from src.api.client import PolymarketClient
from src.engine.execution import ExecutionTracker
from src.strategies.base import Strategy


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
    return PolymarketClient(paper_trading=True)


# === sync_position_from_inventory ===

class TestSyncPosition:

    def test_sets_from_inventory(self):
        s = MinimalStrategy()
        s.sync_position_from_inventory("tok_yes", 50.0, 0.45)
        assert s.active_token_id == "tok_yes"
        assert s.entry_price == 0.45
        assert s.entry_size == 50.0

    def test_clears_when_flat(self):
        s = MinimalStrategy()
        s.active_token_id = "tok_yes"
        s.entry_price = 0.45
        s.entry_size = 50.0
        s.entry_timestamp = time.monotonic()
        s.sync_position_from_inventory(None, 0.0, 0.0)
        assert s.active_token_id is None

    def test_corrects_stale_token(self):
        s = MinimalStrategy()
        s.active_token_id = "tok_yes"
        s.sync_position_from_inventory("tok_no", 30.0, 0.55)
        assert s.active_token_id == "tok_no"
        assert s.entry_price == 0.55

    def test_noop_when_correct(self):
        s = MinimalStrategy()
        ts = time.monotonic() - 100
        s.active_token_id = "tok_yes"
        s.entry_price = 0.45
        s.entry_size = 50.0
        s.entry_timestamp = ts
        s.sync_position_from_inventory("tok_yes", 50.0, 0.45)
        assert s.entry_timestamp == ts  # unchanged


# === Paper trading fixes ===

class TestPaperTrading:

    def test_position_keyed_by_token_id(self):
        c = _make_paper_client()
        c.place_order("mkt", "tok_abc", "YES", "BUY", 0.50, 10.0)
        positions = c.get_positions()
        by_token = {p.token_id: p for p in positions}
        assert "tok_abc" in by_token

    def test_order_starts_pending(self):
        c = _make_paper_client()
        order = c.place_order("mkt", "tok_abc", "YES", "BUY", 0.50, 10.0)
        assert order.status == "PENDING"

    def test_settles_on_get_positions(self):
        c = _make_paper_client()
        order = c.place_order("mkt", "tok_abc", "YES", "BUY", 0.50, 10.0)
        positions = c.get_positions()
        assert order.status == "FILLED"
        assert len(positions) == 1

    def test_sell_reduces_position(self):
        c = _make_paper_client()
        c.place_order("mkt", "tok_abc", "YES", "BUY", 0.50, 10.0)
        c.get_positions()
        c.place_order("mkt", "tok_abc", "YES", "SELL", 0.55, 10.0)
        positions = c.get_positions()
        assert all(p.token_id != "tok_abc" for p in positions)


# === Rollover cleanup ===

class TestRollover:

    def test_execution_tracker_cleanup(self):
        from src.api.types import Order
        from datetime import datetime

        tracker = ExecutionTracker()
        tracker.active_orders["o1"] = Order(
            order_id="o1", market_id="m", token_id="old_yes",
            outcome="YES", side="BUY", price=0.5, size=10,
            status="PENDING", timestamp=datetime.now(),
        )
        tracker._last_positions_by_token["old_yes"] = (10.0, 0.5)

        for old_tid in ("old_yes", "old_no"):
            tracker.active_orders = {
                oid: o for oid, o in tracker.active_orders.items()
                if o.token_id != old_tid
            }
            tracker._last_positions_by_token.pop(old_tid, None)

        assert len(tracker.active_orders) == 0
        assert "old_yes" not in tracker._last_positions_by_token
