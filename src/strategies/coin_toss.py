"""
Coin Toss baseline strategy — pure infrastructure test with no edge.

On each cycle, flips a fair coin and buys YES or NO with equal probability.
Holds until take-profit, stop-loss, or max_hold_seconds elapses.

Purpose: verify the full pipeline (order placement, fills, P&L) works correctly
and establish a random baseline. Any real strategy must beat this in backtest.
"""

import random
import time
from typing import Any, Dict, List, Optional

from .base import Signal, Strategy
from ..api.types import OrderBook, Position
from ..utils.market_utils import get_mid_price, round_to_tick


class CoinTossStrategy(Strategy):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="coin_toss", config=config)

        self.profit_target_pct: float  = config.get("profit_target_pct", 0.10)
        self.stop_loss_pct: float      = config.get("stop_loss_pct", 0.05)
        self.max_hold_seconds: int     = int(config.get("max_hold_seconds", 240))
        self.position_size_usdc: float = config.get("position_size_usdc", 5.0)

        # Token context — set by set_tokens() before analyze() is called
        self._market_id: str = ""
        self._yes_token_id: str = ""
        self._no_token_id: str = ""
        self._outcome_map: Dict[str, str] = {}

        # Open position state
        self.active_token_id: Optional[str]  = None
        self.entry_price: Optional[float]    = None
        self.entry_timestamp: Optional[float] = None  # time.monotonic()
        self.entry_size: Optional[float]     = None

        # Saved just before reset so _execute_signals can compute realized PnL
        self._pending_exit_entry_price: Optional[float] = None
        self._pending_exit_entry_size: Optional[float]  = None

    # ------------------------------------------------------------------
    # Public setters (same interface as BTCUpDownStrategy)
    # ------------------------------------------------------------------

    def record_price(self, token_id: str, mid: float, ts: float = None) -> None:
        """No-op — coin toss needs no price history."""
        pass

    def set_tokens(self, market_id: str, yes_token_id: str, no_token_id: str) -> None:
        """Register current market tokens. Resets position state on rollover."""
        if (yes_token_id != self._yes_token_id or no_token_id != self._no_token_id) \
                and self._yes_token_id:
            self._reset_position_state()
        self._market_id    = market_id
        self._yes_token_id = yes_token_id
        self._no_token_id  = no_token_id
        self._outcome_map  = {yes_token_id: "YES", no_token_id: "NO"}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Steps each cycle:
          1. Auto-recover position state from live positions.
          2. Check exit conditions → emit SELL if triggered.
          3. If flat → flip coin → emit BUY for YES or NO.
        """
        if not self._yes_token_id or not self._no_token_id:
            return []

        order_books: Dict[str, OrderBook] = market_data.get("order_books", {})
        positions: List[Position]         = market_data.get("positions", [])
        by_token: Dict[str, Position]     = {p.token_id: p for p in positions}
        now_ts = time.monotonic()

        # 1. Auto-recover state after restart
        if self.active_token_id is None:
            for tid in (self._yes_token_id, self._no_token_id):
                pos = by_token.get(tid)
                if pos and pos.size > 0:
                    self.active_token_id = tid
                    self.entry_price     = pos.average_price
                    self.entry_timestamp = now_ts
                    self.entry_size      = pos.size
                    break

        # 2. Exit check — always before entry
        exit_sig = self.check_exit(order_books, by_token, now_ts)
        if exit_sig:
            self._pending_exit_entry_price = self.entry_price
            self._pending_exit_entry_size  = self.entry_size
            self._reset_position_state()
            return [exit_sig]

        # 3. Entry — only when flat
        if self.active_token_id is None:
            entry_sig = self._coin_flip_entry(order_books, now_ts)
            if entry_sig:
                return [entry_sig]

        return []

    def should_enter(self, signal: Signal) -> bool:
        return self.validate_signal(signal)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _coin_flip_entry(
        self,
        order_books: Dict[str, OrderBook],
        now_ts: float,
    ) -> Optional[Signal]:
        # Flip coin: heads → YES, tails → NO
        outcome  = "YES" if random.random() < 0.5 else "NO"
        token_id = self._yes_token_id if outcome == "YES" else self._no_token_id

        book = order_books.get(token_id)
        if book is None:
            return None
        best_ask = book.asks[0].price if book.asks else get_mid_price(book)
        if not best_ask or best_ask <= 0:
            return None

        tick = book.tick_size or 0.001
        size = round(self.position_size_usdc / best_ask, 2)

        # Optimistically record position state
        self.active_token_id = token_id
        self.entry_price     = best_ask
        self.entry_timestamp = now_ts
        self.entry_size      = size

        return Signal(
            market_id=self._market_id,
            outcome=outcome,
            action="BUY",
            confidence=0.5,
            price=round_to_tick(best_ask, tick),
            size=size,
            reason=f"coin_toss: bought {outcome}",
        )

