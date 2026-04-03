from datetime import datetime

from src.engine.inventory import InventoryState
from src.engine.pnl import PnLTracker
from src.api.types import Fill

def assert_close(a: float, b: float, eps: float = 1e-9) -> None:
    assert abs(a - b) <= eps, f"Expected {a} ~= {b}"

def test_inventory_and_pnl_basic():
    inv = InventoryState(token_id="t1")
    pnl = PnLTracker()

    fill = Fill(
        order_id="f1",
        side="BUY",
        price=0.2,
        size=10.0,
        timestamp=datetime.utcnow(),
        token_id="t1",
    )
    pnl.apply_fill(fill, inv)
    assert_close(inv.position, 10.0)
    assert_close(inv.avg_cost, 0.2)

    mid = 0.25
    pnl.mark(mid, inv)
    assert_close(pnl.unrealized, (mid - 0.2) * 10.0)