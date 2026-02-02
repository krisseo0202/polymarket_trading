from datetime import datetime

from src.engine.execution import ExecutionTracker
from src.engine.inventory import InventoryState
from src.api.types import Position


def assert_close(a: float, b: float, eps: float = 1e-9) -> None:
    assert abs(a - b) <= eps, f"Expected {a} ~= {b}"

class FakeClient:
    def __init__(self) -> None:
        self._open_orders_by_token = {}
        self._positions = []

    def set_state(self, open_orders_by_token, positions) -> None:
        self._open_orders_by_token = open_orders_by_token
        self._positions = positions

    def get_open_orders(self, token_id=None):
        if token_id is None:
            combined = []
            for orders in self._open_orders_by_token.values():
                combined.extend(orders)
            return combined
        return list(self._open_orders_by_token.get(token_id, []))

    def get_positions(self):
        return list(self._positions)

    def get_recent_fills(self, token_id, since_ts=None):
        return []

def test_execution_tracker_filled_partial_canceled():
    client = FakeClient()
    token = "t1"

    # Filled
    tracker = ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)
    client.set_state(
        {token: [{"id": "o1", "side": "BUY", "price": 0.2, "size": 2, "token_id": token}]},
        [Position(market_id="m1", token_id=token, outcome="YES", size=0.0, average_price=0.0)],
    )
    tracker.reconcile(client, [token], inventories={token: InventoryState(token_id=token)})

    client.set_state(
        {token: []},
        [Position(market_id="m1", token_id=token, outcome="YES", size=2.0, average_price=0.2)],
    )
    closed = tracker.reconcile(client, [token], inventories={token: InventoryState(token_id=token)})
    assert closed[0].status == "FILLED"
    assert_close(closed[0].filled_qty, 2.0)
    assert_close(closed[0].avg_fill_price, 0.2)

    # Partial
    tracker_partial = ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)
    client.set_state(
        {token: [{"id": "o2", "side": "BUY", "price": 0.2, "size": 2, "token_id": token}]},
        [Position(market_id="m1", token_id=token, outcome="YES", size=0.0, average_price=0.0)],
    )
    tracker_partial.reconcile(client, [token], inventories={token: InventoryState(token_id=token)})
    client.set_state(
        {token: []},
        [Position(market_id="m1", token_id=token, outcome="YES", size=1.0, average_price=0.2)],
    )
    closed = tracker_partial.reconcile(client, [token], inventories={token: InventoryState(token_id=token)})
    assert closed[0].status == "PARTIALLY_FILLED"
    assert_close(closed[0].filled_qty, 1.0)

    # Canceled
    tracker_cancel = ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)
    client.set_state(
        {token: [{"id": "o3", "side": "SELL", "price": 0.2, "size": 2, "token_id": token}]},
        [Position(market_id="m1", token_id=token, outcome="YES", size=0.0, average_price=0.0)],
    )
    tracker_cancel.reconcile(client, [token], inventories={token: InventoryState(token_id=token)})
    client.set_state(
        {token: []},
        [Position(market_id="m1", token_id=token, outcome="YES", size=0.0, average_price=0.0)],
    )
    closed = tracker_cancel.reconcile(client, [token], inventories={token: InventoryState(token_id=token)})
    assert closed[0].status == "CANCELED"
    assert_close(closed[0].filled_qty, 0.0)