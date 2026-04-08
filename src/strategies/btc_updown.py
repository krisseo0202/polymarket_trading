"""
Stateful directional momentum strategy for BTC Up/Down 5-minute binary markets.

Bias behaviour: LONG → buy YES, SHORT → buy NO, NONE → stay flat.
An external caller sets the view via set_bias(); config key `default_bias`
seeds an initial value so the bot works without a live caller.

Momentum confirmation (deterministic, no indicators):
  return_pct = (mid_now - mid_prev) / mid_prev  >=  confirmation_pct
where mid_prev is the closest recorded price ≥ confirmation_window_seconds ago.
Mid-price history is accumulated internally — bot need not populate price_history.

Position lifecycle:
  Entry  → bias set + no open position + momentum confirmed.
  Exit   → take-profit | stop-loss | max-hold time (checked before entry).

Market-rollover safety: set_tokens() resets state when 5-min tokens change.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time

from .base import Signal, Strategy
from ..api.types import OrderBook, Position
from ..utils.market_utils import get_mid_price, round_to_tick


class Bias(Enum):
    LONG  = "LONG"   # BTC expected up   → trade YES token
    SHORT = "SHORT"  # BTC expected down → trade NO  token
    NONE  = "NONE"   # no view           → stay flat


class BTCUpDownStrategy(Strategy):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="btc_updown", config=config)

        self.confirmation_pct: float        = config.get("confirmation_pct", 0.02)
        self.confirmation_window_seconds: int = int(config.get("confirmation_window_seconds", 60))
        self.profit_target_pct: float       = config.get("profit_target_pct", 0.04)
        self.stop_loss_pct: float           = config.get("stop_loss_pct", 0.12)
        self.max_hold_seconds: int          = int(config.get("max_hold_seconds", 240))
        self.position_size_usdc: float      = config.get("position_size_usdc", 20.0)

        self.current_bias: Bias = Bias[config.get("default_bias", "NONE").upper()]

        self._history_max_age = max(self.confirmation_window_seconds * 3, 300.0)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_bias(self, bias: Bias) -> None:
        """
        Set directional view from an external source (user, model, rule).
        LONG / SHORT selects which token to enter; NONE keeps the bot flat.
        """
        self.current_bias = bias

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Called every cycle. Steps:
          1. Append current mid-prices to internal history buffer.
          2. Auto-recover lost position state from live positions.
          3. Check exit conditions → emit SELL if triggered.
          4. Check entry conditions → emit BUY if bias + momentum align.
        """
        if not self._yes_token_id or not self._no_token_id:
            return []

        order_books: Dict[str, OrderBook] = market_data.get("order_books", {})
        positions: List[Position]         = market_data.get("positions", [])
        by_token: Dict[str, Position]     = {p.token_id: p for p in positions}
        now_ts = time.monotonic()

        # 1. Record mid-prices
        for tid in (self._yes_token_id, self._no_token_id):
            book = order_books.get(tid)
            if book is None:
                continue
            mid = get_mid_price(book)
            if mid is None:
                continue
            buf = self._price_history.setdefault(tid, [])
            buf.append((now_ts, mid))
            cutoff = now_ts - self._history_max_age
            self._price_history[tid] = [(ts, p) for ts, p in buf if ts >= cutoff]

        self._auto_recover_position(by_token, now_ts)

        # 3. Exit check — always before entry
        exit_sig = self.check_exit(order_books, by_token, now_ts)
        if exit_sig:
            self._pending_exit_entry_price = self.entry_price
            self._pending_exit_entry_size  = self.entry_size
            self._reset_position_state()
            return [exit_sig]

        # 4. Entry check
        if self.current_bias != Bias.NONE and self.active_token_id is None:
            entry_sig = self._check_entry(order_books, by_token, now_ts)
            if entry_sig:
                return [entry_sig]

        return []

    def should_enter(self, signal: Signal) -> bool:
        return self.validate_signal(signal) and signal.confidence >= self.min_confidence

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_entry(
        self,
        order_books: Dict[str, OrderBook],
        by_token: Dict[str, Position],
        now_ts: float,
    ) -> Optional[Signal]:
        """
        Entry conditions:
          - Bias selects YES (LONG) or NO (SHORT) token.
          - No existing position for that token.
          - Momentum measured on YES token return for both directions:
              LONG:  YES_return_pct >= +confirmation_pct  → buy YES
              SHORT: YES_return_pct <= -confirmation_pct  → buy NO
            where YES_return_pct = (yes_mid_now - yes_mid_prev) / yes_mid_prev
            and yes_mid_prev is from at least confirmation_window_seconds ago.
        On confirmation, state is set optimistically so exit logic
        fires on the very next cycle.
        """
        # Determine target token for position/book lookup
        token_id = self._yes_token_id if self.current_bias == Bias.LONG \
                   else self._no_token_id

        if not self.is_flat(by_token):
            return None   # already holding — no pyramiding

        # Always measure momentum on YES token
        yes_book = order_books.get(self._yes_token_id)
        yes_mid_now = get_mid_price(yes_book) if yes_book else None
        yes_mid_prev = self._lookback_mid(self._yes_token_id, now_ts, self.confirmation_window_seconds)
        if yes_mid_now is None or yes_mid_prev is None or yes_mid_prev <= 0:
            return None   # insufficient history or no book

        return_pct = (yes_mid_now - yes_mid_prev) / yes_mid_prev

        if self.current_bias == Bias.LONG:
            if return_pct < self.confirmation_pct:
                return None   # YES not rising fast enough
        elif self.current_bias == Bias.SHORT:
            if return_pct > -self.confirmation_pct:
                return None   # YES not falling fast enough

        book = order_books.get(token_id)
        if book is None:
            return None
        mid_now = get_mid_price(book)
        if mid_now is None:
            return None

        tick     = book.tick_size or 0.001
        best_ask = book.asks[0].price if book.asks else mid_now
        if best_ask <= 0:
            return None
        if best_ask > self.max_entry_price or best_ask < self.min_entry_price:
            return None
        size       = round(self.position_size_usdc / best_ask, 2)
        confidence = min(0.5 + (abs(return_pct) / self.confirmation_pct) * 0.5, 1.0)

        # Optimistically record position state
        self.active_token_id  = token_id
        self.entry_price      = best_ask
        self.entry_timestamp  = now_ts
        self.entry_size       = size

        direction = "rising" if self.current_bias == Bias.LONG else "falling"
        return Signal(
            market_id=self._market_id,
            outcome=self._outcome_map.get(token_id, ""),
            action="BUY",
            confidence=confidence,
            price=round_to_tick(best_ask, tick),
            size=size,
            reason=(
                f"momentum confirmed: YES {direction} return={return_pct:.2%} "
                f"threshold={self.confirmation_pct:.2%} "
                f"over {self.confirmation_window_seconds}s | "
                f"bias={self.current_bias.value}"
            ),
        )

    def _lookback_mid(
        self, token_id: str, now_ts: float, lookback_seconds: int
    ) -> Optional[float]:
        """
        Return the most-recent mid recorded at least `lookback_seconds` ago.
        History is chronological; we walk forward and keep the last entry
        that falls before the target timestamp.
        """
        target_ts = now_ts - lookback_seconds
        result: Optional[float] = None
        for ts, mid in self._price_history.get(token_id, []):
            if ts <= target_ts:
                result = mid
            else:
                break
        return result

