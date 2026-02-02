import os

import pytest

from src.api.client import PolymarketClient
from src.engine.execution import ExecutionTracker
from src.engine.inventory import InventoryState


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        pytest.skip(f"Missing env var: {var_name}")
    return value


def _build_live_client() -> PolymarketClient:
    return PolymarketClient(
        private_key=_require_env("PRIVATE_KEY"),
        funder_address=_require_env("PROXY_FUNDER"),
        host=os.getenv("HOST", "https://clob.polymarket.com"),
        chain_id=int(os.getenv("CHAIN_ID", "137")),
        paper_trading=False,
    )


def test_live_reconcile_positions(live_mode, live_env_enabled, yes_token_id, no_token_id):
    if not live_mode or not live_env_enabled:
        pytest.skip("Live test disabled. Use --live and POLY_LIVE_TEST=1.")

    if not yes_token_id or not no_token_id:
        pytest.skip("Missing YES/NO token IDs. Use --yes-token/--no-token or env vars.")

    client = _build_live_client()
    tracker = ExecutionTracker(orders_sync_interval_s=0, positions_sync_interval_s=0)

    inventories = {
        yes_token_id: InventoryState(token_id=yes_token_id),
        no_token_id: InventoryState(token_id=no_token_id),
    }

    closed = tracker.reconcile(client, [yes_token_id, no_token_id], inventories=inventories)
    assert isinstance(closed, list)

    positions = {p.token_id: p for p in client.get_positions()}
    for token_id in [yes_token_id, no_token_id]:
        inv = inventories[token_id]
        pos = positions.get(token_id)
        if pos is None:
            assert inv.position == 0.0
            assert inv.avg_cost == 0.0
        else:
            assert inv.position == float(pos.size)
            assert inv.avg_cost == float(pos.average_price)