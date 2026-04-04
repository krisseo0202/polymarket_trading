"""Execution tracking for market making"""

from typing import Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

from ..api.client import PolymarketClient
from ..api.types import Fill, Order, Position
from ..utils.logger import setup_logger
from .inventory import InventoryState

logger = setup_logger(level="INFO")


class ExecutionTracker:
    """
    Minimal fill poller with deduplication.
    Dedupe key: (order_id, timestamp_iso)
    """
    def __init__(
        self,
        orders_sync_interval_s: float = 5.0,
        positions_sync_interval_s: float = 30.0,
    ):
        self.last_fill_ts: Optional[datetime] = None
        self.seen: set = set()
        self.active_orders: Dict[str, Order] = {}
        self.closed_orders: Dict[str, Order] = {}
        self.inferred_fills: List[Fill] = []
        self.realized_pnl_from_fills: float = 0.0  # computed before snapshot overwrite
        self._fills_by_order_id: Dict[str, List[Fill]] = {}
        self._last_positions_by_token: Dict[str, Tuple[float, float]] = {}
        self._orders_sync_interval_s = orders_sync_interval_s
        self._positions_sync_interval_s = positions_sync_interval_s
        self._last_orders_sync_monotonic: Optional[float] = None
        self._last_positions_sync_monotonic: Optional[float] = None

    def poll(self, client: PolymarketClient, token_id: str) -> List[Fill]:
        """
        Poll for recent fills and return new ones.
        
        Args:
            client: Polymarket API client
            token_id: Token ID to poll fills for
            
        Returns:
            List of new Fill objects, sorted by timestamp
        """
        try:
            data = client.get_recent_fills(token_id, since_ts=self.last_fill_ts)
        except Exception as e:
            logger.warning(f"poll fills failed: {e}")
            return []

        out: List[Fill] = []
        for f in data:
            order_id = f.get("order_id") or f.get("id") or ""
            ts_raw = f.get("timestamp") or f.get("ts")

            # minimal parsing
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.utcnow()
            elif isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(ts_raw)
            else:
                ts = datetime.utcnow()

            key = (order_id, ts.isoformat())
            if key in self.seen:
                continue
            self.seen.add(key)

            fill = Fill(
                order_id=order_id,
                side=str(f.get("side", "BUY")).upper(),   # "BUY"/"SELL"
                price=float(f.get("price", 0.0)),
                size=float(f.get("size", 0.0)),
                timestamp=ts,
                token_id=token_id,
            )
            out.append(fill)
            if order_id:
                self._fills_by_order_id.setdefault(order_id, []).append(fill)

            if self.last_fill_ts is None or ts > self.last_fill_ts:
                self.last_fill_ts = ts

        out.sort(key=lambda x: x.timestamp)
        return out

    def reconcile_open_orders(self, client: PolymarketClient, token_id: str) -> List[Order]:
        """
        Reconcile open orders for a token and update active_orders.
        Any previously active order that disappears is classified as FILLED,
        PARTIALLY_FILLED, or CANCELED using positions and fills.
        """
        if not token_id:
            return []
        return self.reconcile(client, [token_id], inventories={})

    def reconcile_inventory_from_positions(
        self,
        client: PolymarketClient,
        token_ids: Optional[Iterable[str]],
        inventories: Dict[str, InventoryState],
    ) -> Dict[str, Position]:
        """
        Reconcile inventory state from positions (source of truth).
        """
        if inventories is None:
            return {}

        if not self._should_sync(self._last_positions_sync_monotonic, self._positions_sync_interval_s):
            return {}

        token_list = list(token_ids) if token_ids else list(inventories.keys())
        if not token_list:
            return {}

        try:
            positions = client.get_positions()
        except Exception as e:
            logger.warning(f"poll positions failed: {e}")
            return {}

        positions_by_token = {pos.token_id: pos for pos in positions}
        self._apply_positions_snapshot(token_list, positions_by_token, inventories)
        self._update_last_positions(token_list, positions_by_token)
        self._last_positions_sync_monotonic = time.monotonic()
        return positions_by_token

    def reconcile(
        self,
        client: PolymarketClient,
        token_ids: Iterable[str],
        inventories: Dict[str, InventoryState],
    ) -> List[Order]:
        """
        Run a reconciliation cycle for orders and inventory.
        Returns orders that disappeared from the open-order set.
        Inferred fills are stored on self.inferred_fills.
        """
        token_list = list(token_ids)
        closed_orders: List[Order] = []
        self.inferred_fills = []
        self.realized_pnl_from_fills = 0.0

        if not token_list:
            return closed_orders

        open_orders_by_token: Dict[str, Dict[str, Order]] = {}
        if self._should_sync(self._last_orders_sync_monotonic, self._orders_sync_interval_s):
            with ThreadPoolExecutor(max_workers=len(token_list)) as pool:
                futures = {pool.submit(client.get_open_orders, tid): tid for tid in token_list}
                for future in as_completed(futures):
                    tid = futures[future]
                    try:
                        open_orders_raw = future.result()
                    except Exception as e:
                        logger.warning(f"poll open orders failed: {e}")
                        open_orders_raw = []
                    open_orders_by_token[tid] = self._parse_open_orders(open_orders_raw, tid)

            next_active: Dict[str, Order] = {
                order_id: order
                for order_id, order in self.active_orders.items()
                if order.token_id not in token_list
            }
            for token_id in token_list:
                open_orders = open_orders_by_token.get(token_id, {})
                for order_id, order in self.active_orders.items():
                    if order.token_id == token_id and order_id not in open_orders:
                        closed_orders.append(order)
                        self.closed_orders[order_id] = order
                next_active.update(open_orders)
            self.active_orders = next_active
            self._last_orders_sync_monotonic = time.monotonic()

        should_sync_positions = self._should_sync(
            self._last_positions_sync_monotonic, self._positions_sync_interval_s
        )
        if should_sync_positions or closed_orders:
            positions_ok = True
            try:
                positions = client.get_positions()
            except Exception as e:
                logger.warning(f"poll positions failed: {e}")
                positions = []
                positions_ok = False
            positions_by_token = {pos.token_id: pos for pos in positions}
            deltas = (
                self._compute_position_deltas(token_list, positions_by_token)
                if positions_ok
                else {token_id: 0.0 for token_id in token_list}
            )

            if closed_orders:
                self._resolve_closed_orders(closed_orders, deltas)
                self.inferred_fills = self._build_inferred_fills(closed_orders)
                # Compute PnL using pre-snapshot inventory (avg_cost is still correct here).
                # Must run BEFORE _apply_positions_snapshot overwrites inv.avg_cost/position.
                if inventories:
                    self.realized_pnl_from_fills = self._compute_realized_pnl(
                        self.inferred_fills, inventories
                    )

            if inventories is not None and positions_ok:
                self._apply_positions_snapshot(token_list, positions_by_token, inventories)

            if positions_ok:
                self._last_positions_sync_monotonic = time.monotonic()
                self._update_last_positions(token_list, positions_by_token)

        return closed_orders

    def _should_sync(self, last_ts: Optional[float], interval_s: float) -> bool:
        if interval_s <= 0:
            return True
        return last_ts is None or (time.monotonic() - last_ts) >= interval_s

    def _parse_open_orders(self, raw: List[Dict[str, object]], token_id: str) -> Dict[str, Order]:
        parsed: Dict[str, Order] = {}
        for data in raw:
            order = self._parse_open_order(data, token_id)
            if order.order_id:
                parsed[order.order_id] = order
        return parsed

    def _parse_open_order(self, data: Dict[str, object], token_id: str) -> Order:
        order_id = (
            str(data.get("id") or data.get("order_id") or data.get("orderId") or "")
        )
        side = str(data.get("side", "")).upper()
        price = float(data.get("price", 0.0) or 0.0)
        size = float(data.get("size", 0.0) or 0.0)
        status = str(data.get("status", "PENDING")).upper()
        market_id = str(data.get("market_id") or data.get("marketId") or "")
        outcome = str(data.get("outcome") or "")
        return Order(
            order_id=order_id,
            market_id=market_id,
            token_id=str(data.get("token_id") or token_id or ""),
            outcome=outcome,
            side=side,
            price=price,
            size=size,
            status=status,
            timestamp=datetime.utcnow(),
        )

    def _compute_position_deltas(
        self,
        token_list: List[str],
        positions_by_token: Dict[str, Position],
    ) -> Dict[str, float]:
        deltas: Dict[str, float] = {}
        for token_id in token_list:
            prev = self._last_positions_by_token.get(token_id, (0.0, 0.0))
            prev_size = float(prev[0])
            curr = positions_by_token.get(token_id)
            curr_size = float(curr.size) if curr else 0.0
            deltas[token_id] = curr_size - prev_size
        return deltas

    def _resolve_closed_orders(self, closed_orders: List[Order], deltas: Dict[str, float]) -> None:
        grouped: Dict[str, List[Order]] = {}
        for order in closed_orders:
            order.filled_qty = 0.0
            order.avg_fill_price = None
            grouped.setdefault(order.token_id, []).append(order)

        for token_id, orders in grouped.items():
            delta = float(deltas.get(token_id, 0.0))
            fill_adjustment = 0.0
            for order in orders:
                fill_qty, avg_price = self._get_fill_stats(order.order_id)
                if fill_qty > 0:
                    side_sign = 1.0 if order.side == "BUY" else -1.0
                    fill_adjustment += side_sign * fill_qty
                    order.filled_qty = min(fill_qty, order.size)
                    order.avg_fill_price = avg_price

            remaining = delta - fill_adjustment
            for order in sorted(orders, key=self._order_sort_key):
                if order.filled_qty > 0:
                    continue
                if order.side == "BUY" and remaining <= 0:
                    continue
                if order.side == "SELL" and remaining >= 0:
                    continue
                alloc = min(order.size, abs(remaining))
                if alloc <= 0:
                    continue
                order.filled_qty = alloc
                order.avg_fill_price = order.price if order.price > 0 else None
                remaining += -alloc if order.side == "BUY" else alloc

            if abs(remaining) > 1e-9:
                logger.warning(
                    f"position delta mismatch for {token_id}: remaining {remaining:.6f}"
                )

            for order in orders:
                self._finalize_order_state(order)

    def _compute_realized_pnl(
        self, fills: List[Fill], inventories: Dict[str, InventoryState]
    ) -> float:
        """Compute realized PnL from fills using the current (pre-snapshot) inventory.

        Must be called BEFORE _apply_positions_snapshot() overwrites avg_cost/position.
        Mirrors the avg-cost accounting in InventoryState.apply_fill() without mutating state.
        """
        realized = 0.0
        for fill in fills:
            inv = inventories.get(fill.token_id)
            if inv is None or inv.position == 0:
                continue
            if fill.side == "SELL" and inv.position > 0:
                closing = min(fill.size, inv.position)
                realized += (fill.price - inv.avg_cost) * closing
            elif fill.side == "BUY" and inv.position < 0:
                closing = min(fill.size, abs(inv.position))
                realized += (inv.avg_cost - fill.price) * closing
        return realized

    def _build_inferred_fills(self, closed_orders: List[Order]) -> List[Fill]:
        inferred: List[Fill] = []
        for order in closed_orders:
            if order.filled_qty <= 0:
                continue
            price = order.avg_fill_price if order.avg_fill_price is not None else order.price
            if price <= 0:
                continue
            inferred.append(
                Fill(
                    order_id=order.order_id or "",
                    side=order.side,
                    price=price,
                    size=order.filled_qty,
                    timestamp=order.timestamp or datetime.utcnow(),
                    token_id=order.token_id,
                )
            )
        return inferred

    def _get_fill_stats(self, order_id: Optional[str]) -> Tuple[float, Optional[float]]:
        if not order_id:
            return 0.0, None
        fills = self._fills_by_order_id.get(order_id, [])
        if not fills:
            return 0.0, None
        total_qty = sum(f.size for f in fills)
        if total_qty <= 0:
            return 0.0, None
        avg_price = sum(f.size * f.price for f in fills) / total_qty
        return total_qty, avg_price

    def _apply_positions_snapshot(
        self,
        token_list: List[str],
        positions_by_token: Dict[str, Position],
        inventories: Dict[str, InventoryState],
    ) -> None:
        for token_id in token_list:
            if not token_id:
                continue
            inv = inventories.get(token_id)
            if inv is None:
                inv = InventoryState(token_id=token_id)
                inventories[token_id] = inv
            pos = positions_by_token.get(token_id)
            if pos:
                inv.position = float(pos.size)
                inv.avg_cost = float(pos.average_price)
            else:
                inv.position = 0.0
                inv.avg_cost = 0.0
            inv._normalize()

    def _update_last_positions(
        self,
        token_list: List[str],
        positions_by_token: Dict[str, Position],
    ) -> None:
        for token_id in token_list:
            pos = positions_by_token.get(token_id)
            if pos:
                self._last_positions_by_token[token_id] = (
                    float(pos.size),
                    float(pos.average_price),
                )
            else:
                self._last_positions_by_token[token_id] = (0.0, 0.0)

    def _finalize_order_state(self, order: Order) -> None:
        eps = 1e-9
        if order.filled_qty >= max(order.size - eps, 0.0):
            order.status = "FILLED"
        elif order.filled_qty > eps:
            order.status = "PARTIALLY_FILLED"
        else:
            order.status = "CANCELED"

    def _order_sort_key(self, order: Order) -> Tuple[int, float]:
        ts = order.timestamp.timestamp() if order.timestamp else 0.0
        return (0 if order.side == "BUY" else 1, ts)
