"""Inventory management for market making"""

from dataclasses import dataclass
from typing import Literal
from ..api.types import Side


@dataclass
class InventoryState:
    """
    Minimal average-cost inventory tracking.
    Correct on flips and partial closes, but no extra diagnostics.
    """
    token_id: str
    position: float = 0.0  # >0 long, <0 short
    avg_cost: float = 0.0  # avg entry price for OPEN position, 0 if flat

    def _normalize(self) -> None:
        """Normalize near-zero positions to zero"""
        if abs(self.position) < 1e-12:
            self.position = 0.0
            self.avg_cost = 0.0

    def apply_fill(self, side: Side, price: float, size: float) -> float:
        """
        Apply fill and return realized PnL (average-cost).
        
        Args:
            side: "BUY" or "SELL"
            price: Fill price
            size: Fill size
            
        Returns:
            Realized PnL from this fill
        """
        if size <= 0:
            return 0.0

        realized = 0.0
        pos0 = self.position
        avg0 = self.avg_cost

        if side == "BUY":
            if pos0 >= 0:
                # add / open long
                new_pos = pos0 + size
                self.avg_cost = (avg0 * pos0 + price * size) / new_pos if pos0 > 0 else price
                self.position = new_pos
            else:
                # close short (maybe flip)
                closing = min(size, abs(pos0))
                realized += (avg0 - price) * closing  # short: entry(avg0) - exit(price)
                remaining = size - closing
                if remaining > 0:
                    self.position = remaining
                    self.avg_cost = price
                else:
                    self.position = pos0 + size  # less negative
                    self.avg_cost = avg0

        elif side == "SELL":
            if pos0 <= 0:
                # add / open short
                new_pos = pos0 - size
                short0 = abs(pos0)
                short_new = abs(new_pos)
                self.avg_cost = (avg0 * short0 + price * size) / short_new if short0 > 0 else price
                self.position = new_pos
            else:
                # close long (maybe flip)
                closing = min(size, pos0)
                realized += (price - avg0) * closing  # long: exit(price) - entry(avg0)
                remaining = size - closing
                if remaining > 0:
                    self.position = -remaining
                    self.avg_cost = price
                else:
                    self.position = pos0 - size
                    self.avg_cost = avg0

        self._normalize()
        return realized
