"""Profit and Loss tracking for market making"""

from dataclasses import dataclass, field
from typing import List
from ..api.types import Fill
from .inventory import InventoryState


@dataclass
class PnLTracker:
    """
    Minimal PnL tracking:
    - realized from fills
    - unrealized = (mid - avg_cost) * position
    """
    realized: float = 0.0
    unrealized: float = 0.0
    total: float = 0.0
    trade_count: int = 0
    fills: List[Fill] = field(default_factory=list)

    def apply_fill(self, fill: Fill, inv: InventoryState) -> None:
        """
        Apply a fill and update realized PnL.
        
        Args:
            fill: The fill event
            inv: Inventory state to update
        """
        realized_delta = inv.apply_fill(fill.side, fill.price, fill.size)
        self.realized += realized_delta
        self.trade_count += 1
        self.fills.append(fill)
        self._recompute()

    def mark(self, mid: float, inv: InventoryState) -> None:
        """
        Mark-to-market unrealized PnL.
        
        Args:
            mid: Current mid price
            inv: Current inventory state
        """
        self.unrealized = (mid - inv.avg_cost) * inv.position if inv.position != 0 else 0.0
        self._recompute()

    def _recompute(self) -> None:
        """Recalculate total PnL"""
        self.total = self.realized + self.unrealized
