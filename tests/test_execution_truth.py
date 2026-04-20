"""
Tests for execution truth: position, side, fills, realized PnL, unrealized PnL, avg cost.

Covers two bugs that existed before the fix:
  Bug 1 (live) — _apply_positions_snapshot() overwrote inventory before apply_fill_to_state()
                  ran, so realized PnL was never recorded.
  Bug 2 (paper) — execute_signals() applied fills immediately AND reconcile() re-applied
                  them → double PnL.
"""

from datetime import datetime
from typing import Dict


from src.api.types import Fill, Order, Position
from src.engine.execution import ExecutionTracker
from src.engine.inventory import InventoryState, apply_fill_to_state


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _FakeState:
    def __init__(self):
        self.daily_realized_pnl = 0.0
        self.slot_realized_pnl = 0.0
        self.session_wins = 0
        self.session_losses = 0


class _FakeRiskManager:
    def __init__(self):
        self.trades = []

    def record_trade(self, pnl):
        self.trades.append(pnl)


class _FakeClient:
    """Returns caller-provided positions and open orders."""

    def __init__(self, positions=None, open_orders_by_token=None):
        self._positions = positions or []
        self._open_orders = open_orders_by_token or {}

    def get_positions(self):
        return self._positions

    def get_open_orders(self, token_id=None):
        return self._open_orders.get(token_id, [])

    def get_recent_fills(self, token_id, since_ts=None):
        return []


def _make_order(order_id, token_id, side, price, size, status="PENDING") -> Order:
    return Order(
        order_id=order_id,
        market_id="mkt",
        token_id=token_id,
        outcome="YES" if "yes" in token_id else "NO",
        side=side,
        price=price,
        size=size,
        status=status,
        timestamp=datetime.utcnow(),
    )


def _make_position(token_id, size, avg_price) -> Position:
    return Position(
        market_id="mkt",
        token_id=token_id,
        outcome="YES" if "yes" in token_id else "NO",
        size=size,
        average_price=avg_price,
    )


# ---------------------------------------------------------------------------
# ExecutionTracker._compute_realized_pnl
# ---------------------------------------------------------------------------

class TestComputeRealizedPnl:

    def _tracker(self):
        return ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)

    def _fill(self, token_id, side, price, size) -> Fill:
        return Fill(
            order_id="o1",
            side=side,
            price=price,
            size=size,
            timestamp=datetime.utcnow(),
            token_id=token_id,
        )

    def test_sell_close_long_pnl(self):
        """SELL that closes a long position: realized = (sell - buy) * size."""
        tracker = self._tracker()
        inv = InventoryState(token_id="tok_yes", position=10.0, avg_cost=0.50)
        fill = self._fill("tok_yes", "SELL", 0.60, 10.0)
        pnl = tracker._compute_realized_pnl([fill], {"tok_yes": inv})
        assert abs(pnl - 1.0) < 1e-9   # (0.60 - 0.50) * 10

    def test_sell_partial_close_pnl(self):
        """Partial SELL: only the closed portion counts."""
        tracker = self._tracker()
        inv = InventoryState(token_id="tok_yes", position=10.0, avg_cost=0.50)
        fill = self._fill("tok_yes", "SELL", 0.60, 4.0)
        pnl = tracker._compute_realized_pnl([fill], {"tok_yes": inv})
        assert abs(pnl - 0.40) < 1e-9  # (0.60 - 0.50) * 4

    def test_buy_open_has_no_realized_pnl(self):
        """A BUY that opens a new position has zero realized PnL."""
        tracker = self._tracker()
        inv = InventoryState(token_id="tok_yes", position=0.0, avg_cost=0.0)
        fill = self._fill("tok_yes", "BUY", 0.50, 10.0)
        pnl = tracker._compute_realized_pnl([fill], {"tok_yes": inv})
        assert pnl == 0.0

    def test_flat_inventory_yields_zero(self):
        """No position → no PnL regardless of fill."""
        tracker = self._tracker()
        inv = InventoryState(token_id="tok_yes", position=0.0, avg_cost=0.0)
        fill = self._fill("tok_yes", "SELL", 0.60, 5.0)
        pnl = tracker._compute_realized_pnl([fill], {"tok_yes": inv})
        assert pnl == 0.0

    def test_sell_at_loss(self):
        """SELL below entry: negative realized PnL."""
        tracker = self._tracker()
        inv = InventoryState(token_id="tok_yes", position=10.0, avg_cost=0.60)
        fill = self._fill("tok_yes", "SELL", 0.50, 10.0)
        pnl = tracker._compute_realized_pnl([fill], {"tok_yes": inv})
        assert abs(pnl - (-1.0)) < 1e-9  # (0.50 - 0.60) * 10


# ---------------------------------------------------------------------------
# Full reconcile → realized PnL pipeline (Bug 1 fix)
# ---------------------------------------------------------------------------

class TestReconcilePnlPipeline:
    """
    Simulates live trading: an order was placed, filled, and the API now shows
    a closed position.  reconcile() must compute PnL from the pre-snapshot
    inventory state and store it in realized_pnl_from_fills.
    """

    def _setup(self, buy_price=0.50, sell_price=0.60, size=10.0):
        tracker = ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)
        token_id = "tok_yes"

        # Inventory reflects an open long position at buy_price
        inv = InventoryState(token_id=token_id, position=size, avg_cost=buy_price)
        inventories: Dict[str, InventoryState] = {token_id: inv}

        # tracker knows about the open SELL order
        sell_order = _make_order("o1", token_id, "SELL", sell_price, size, "PENDING")
        tracker.active_orders["o1"] = sell_order
        tracker._last_positions_by_token[token_id] = (size, buy_price)

        # API state: position is now flat (SELL was filled)
        client = _FakeClient(
            positions=[],  # flat
            open_orders_by_token={token_id: []},  # order gone
        )
        return tracker, inventories, client, token_id, sell_order

    def test_realized_pnl_correct_after_reconcile(self):
        buy_price, sell_price, size = 0.50, 0.60, 10.0
        tracker, inventories, client, token_id, _ = self._setup(buy_price, sell_price, size)

        tracker.reconcile(client, [token_id], inventories)

        expected_pnl = (sell_price - buy_price) * size  # 1.0
        assert abs(tracker.realized_pnl_from_fills - expected_pnl) < 1e-9

    def test_inventory_overwritten_to_api_truth(self):
        """After reconcile, inventory reflects API state (flat)."""
        tracker, inventories, client, token_id, _ = self._setup()
        tracker.reconcile(client, [token_id], inventories)
        assert inventories[token_id].position == 0.0

    def test_pnl_not_lost_when_inventory_zeroed(self):
        """The pre-fix bug: PnL was 0 because apply_fill ran on zeroed inventory.
        Now it must be non-zero."""
        buy_price, sell_price, size = 0.50, 0.65, 10.0
        tracker, inventories, client, token_id, _ = self._setup(buy_price, sell_price, size)

        tracker.reconcile(client, [token_id], inventories)
        # If buggy: realized_pnl_from_fills == 0.0 because inventory was already flat
        assert tracker.realized_pnl_from_fills > 0.0

    def test_realized_pnl_at_loss(self):
        buy_price, sell_price, size = 0.60, 0.45, 10.0
        tracker, inventories, client, token_id, _ = self._setup(buy_price, sell_price, size)

        tracker.reconcile(client, [token_id], inventories)

        expected_pnl = (sell_price - buy_price) * size  # -1.5
        assert abs(tracker.realized_pnl_from_fills - expected_pnl) < 1e-9

    def test_no_fill_no_pnl(self):
        """If no orders closed, realized_pnl_from_fills stays 0."""
        tracker = ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)
        token_id = "tok_yes"
        inv = InventoryState(token_id=token_id, position=10.0, avg_cost=0.50)
        inventories = {token_id: inv}
        tracker._last_positions_by_token[token_id] = (10.0, 0.50)

        open_order = _make_order("o1", token_id, "SELL", 0.60, 10.0)
        tracker.active_orders["o1"] = open_order

        # API: still open position, order still active
        client = _FakeClient(
            positions=[_make_position(token_id, 10.0, 0.50)],
            open_orders_by_token={token_id: [{"id": "o1", "side": "SELL", "price": "0.60",
                                              "size": "10.0", "token_id": token_id,
                                              "status": "PENDING"}]},
        )
        tracker.reconcile(client, [token_id], inventories)
        assert tracker.realized_pnl_from_fills == 0.0


# ---------------------------------------------------------------------------
# Paper trading: execute_signals applies fills immediately (no double count)
# ---------------------------------------------------------------------------

class TestPaperTradingNoDuplicate:
    """
    In paper mode, execute_signals() calls apply_fill_to_state() immediately.
    reconcile() should NOT apply PnL again (even though inferred_fills may be non-empty).
    The guard 'if not svc.paper_trading' in cycle_runner enforces this.

    We test it here by verifying that apply_fill_to_state() on an already-closed
    inventory (post-snapshot) yields zero PnL — the behaviour that would cause
    double-count if called again.
    """

    def test_apply_fill_on_flat_inventory_yields_zero_pnl(self):
        """
        After _apply_positions_snapshot zeros the inventory, calling apply_fill_to_state
        for a SELL would not yield the correct PnL — it opens a phantom short instead.
        This confirms the guard in cycle_runner is necessary.
        """
        state = _FakeState()
        rm = _FakeRiskManager()
        inv = InventoryState(token_id="tok_yes", position=0.0, avg_cost=0.0)

        # Simulate what happened before the fix: inventory already zeroed by snapshot,
        # then apply_fill_to_state called with the old fill.
        realized = apply_fill_to_state(inv, "SELL", 0.60, 10.0, state, rm)

        # Wrong: opens a phantom short, realized = 0 (no PnL recorded)
        assert realized == 0.0
        assert inv.position == -10.0  # phantom short — confirms inventory corruption

    def test_correct_pnl_via_pre_snapshot_compute(self):
        """
        The fix: compute PnL from _compute_realized_pnl() BEFORE snapshot runs.
        """
        tracker = ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)
        inv = InventoryState(token_id="tok_yes", position=10.0, avg_cost=0.50)
        fill = Fill(order_id="o1", side="SELL", price=0.60, size=10.0,
                    timestamp=datetime.utcnow(), token_id="tok_yes")

        pnl = tracker._compute_realized_pnl([fill], {"tok_yes": inv})

        assert abs(pnl - 1.0) < 1e-9  # (0.60 - 0.50) * 10
        assert inv.position == 10.0   # inventory NOT mutated


# ---------------------------------------------------------------------------
# Paper mode: win/loss counting on mid-slot closes
# ---------------------------------------------------------------------------

class TestPaperWinLossCounting:
    """
    Paper-mode strategy exits (stop-loss, take-profit, edge-reprice) route
    through execute_signals → apply_fill_to_state. Before the fix, only
    daily/slot PnL updated — session_wins/session_losses stayed 0, making
    the dashboard's win-rate row show "—" no matter how many trades closed.
    Hold-to-expiry settlements (settle_expiring_positions) count correctly
    via the same helper now.
    """

    def test_winning_close_increments_session_wins(self):
        state = _FakeState()
        rm = _FakeRiskManager()
        inv = InventoryState(token_id="tok_yes", position=10.0, avg_cost=0.50)

        realized = apply_fill_to_state(inv, "SELL", 0.60, 10.0, state, rm)

        assert realized > 0
        assert state.session_wins == 1
        assert state.session_losses == 0

    def test_losing_close_increments_session_losses(self):
        state = _FakeState()
        rm = _FakeRiskManager()
        inv = InventoryState(token_id="tok_yes", position=10.0, avg_cost=0.50)

        realized = apply_fill_to_state(inv, "SELL", 0.40, 10.0, state, rm)

        assert realized < 0
        assert state.session_wins == 0
        assert state.session_losses == 1

    def test_open_fill_does_not_touch_counters(self):
        """A pure BUY open (no realized PnL) must not bump win/loss."""
        state = _FakeState()
        rm = _FakeRiskManager()
        inv = InventoryState(token_id="tok_yes", position=0.0, avg_cost=0.0)

        apply_fill_to_state(inv, "BUY", 0.50, 10.0, state, rm)

        assert state.session_wins == 0
        assert state.session_losses == 0


# ---------------------------------------------------------------------------
# Held side tracking
# ---------------------------------------------------------------------------

class TestHeldSide:

    def test_yes_position_gives_yes(self):
        from src.engine.inventory import InventoryState

        inv_yes = InventoryState(token_id="tok_yes", position=10.0, avg_cost=0.50)
        inv_no = InventoryState(token_id="tok_no", position=0.0, avg_cost=0.0)
        inventories = {"tok_yes": inv_yes, "tok_no": inv_no}

        snap = _build_snap(inventories)
        assert snap.held_side == "YES"

    def test_no_position_gives_no(self):
        inv_yes = InventoryState(token_id="tok_yes", position=0.0, avg_cost=0.0)
        inv_no = InventoryState(token_id="tok_no", position=10.0, avg_cost=0.50)
        inventories = {"tok_yes": inv_yes, "tok_no": inv_no}

        snap = _build_snap(inventories)
        assert snap.held_side == "NO"

    def test_flat_gives_flat(self):
        inventories = {}
        snap = _build_snap(inventories)
        assert snap.held_side == "FLAT"


# ---------------------------------------------------------------------------
# Helpers for TestHeldSide
# ---------------------------------------------------------------------------

def _build_snap(inventories):
    from src.engine.cycle_snapshot import build_cycle_snapshot

    class _FakeExecTracker:
        active_orders = {}

    class _FakeRM:
        circuit_breaker_active = False

    class _FakeState2:
        daily_realized_pnl = 0.0
        slot_realized_pnl = 0.0
        cycle_count = 0

    class _FakeSlotCtx:
        slot_start_ts = 1_700_000_000
        slot_end_ts = 1_700_000_300
        strike_price = 84_000.0

    return build_cycle_snapshot(
        market_id="mkt",
        question="Will BTC go up?",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        slot_ctx=_FakeSlotCtx(),
        btc_now=84_000.0,
        yes_book=None,
        no_book=None,
        inventories=inventories,
        execution_tracker=_FakeExecTracker(),
        risk_manager=_FakeRM(),
        state=_FakeState2(),
        paper_trading=True,
        now=1_700_000_150.0,
    )
