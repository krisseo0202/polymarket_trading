"""
Stateful volatility mean-reversion strategy for BTC Up/Down 5-minute binary markets.

In highly volatile BTC markets, prediction market prices overreact to short-term spikes.
When BTC drops 1% in 30s, traders panic-sell YES tokens (pushing price to 30-35¢), but
a single spike has ~50% chance of reversing within the same 5-min window. This creates
a mean-reversion edge: fade the overreaction.

Volatility signal (deterministic, no indicators):
  z = (current_yes_price - rolling_mean) / rolling_std  over 90s window
  Entry when |z| >= zscore_threshold (default 1.8)
    z < -1.8  → YES is cheap (bearish overreaction) → BUY YES
    z > +1.8  → YES is expensive (bullish overreaction) → BUY NO

Position lifecycle:
  Entry  → z-score signal confirmed + no open position.
  Exit   → take-profit | stop-loss | max-hold time (checked before entry).

Market-rollover safety: set_tokens() resets state when 5-min tokens change.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math
import time

from .base import Signal, Strategy
from ..api.types import OrderBook, Position
from ..utils.market_utils import get_mid_price, round_to_tick


class BTCVolatilityReversionStrategy(Strategy):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="btc_vol_reversion", config=config)

        self.window_seconds: int        = int(config.get("window_seconds", 90))
        self.min_samples: int           = int(config.get("min_samples", 20))
        self.zscore_threshold: float    = config.get("zscore_threshold", 1.8)
        self.min_vol_threshold: float   = config.get("min_vol_threshold", 0.005)
        self.profit_target_pct: float   = config.get("profit_target_pct", 0.03)
        self.stop_loss_pct: float       = config.get("stop_loss_pct", 0.07)
        self.max_hold_seconds: int      = int(config.get("max_hold_seconds", 90))
        self.position_size_usdc: float  = config.get("position_size_usdc", 20.0)

        # Token context — refreshed each cycle by set_tokens()
        self._market_id: str = ""
        self._yes_token_id: str = ""
        self._no_token_id: str = ""
        self._outcome_map: Dict[str, str] = {}

        # Open position state
        self.active_token_id: Optional[str]   = None
        self.entry_price: Optional[float]      = None
        self.entry_timestamp: Optional[float]  = None  # time.monotonic()
        self.entry_size: Optional[float]       = None

        # Saved just before reset so _execute_signals can compute realized PnL
        self._pending_exit_entry_price: Optional[float] = None
        self._pending_exit_entry_size: Optional[float]  = None

        # Mid-price history: token_id → [(monotonic_ts, mid), ...]
        self._price_history: Dict[str, List[Tuple[float, float]]] = {}
        self._history_max_age: float = max(self.window_seconds * 3, 300.0)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_tokens(self, market_id: str, yes_token_id: str, no_token_id: str) -> None:
        """Register current market tokens. Resets position state on rollover."""
        if (yes_token_id != self._yes_token_id or no_token_id != self._no_token_id) \
                and self._yes_token_id:
            self._reset_position_state()
        self._market_id   = market_id
        self._yes_token_id = yes_token_id
        self._no_token_id  = no_token_id
        self._outcome_map  = {yes_token_id: "YES", no_token_id: "NO"}

    def record_price(self, token_id: str, mid: float, ts: Optional[float] = None) -> None:
        """Feed a mid-price observation into the internal history buffer.
        Called by the ticker every 1s so the rolling window has dense samples
        for accurate z-score computation.
        """
        now = ts if ts is not None else time.monotonic()
        buf = self._price_history.setdefault(token_id, [])
        buf.append((now, mid))
        cutoff = now - self._history_max_age
        self._price_history[token_id] = [(t, p) for t, p in buf if t >= cutoff]

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Called every cycle. Steps:
          1. Append current mid-prices to internal history buffer.
          2. Auto-recover lost position state from live positions.
          3. Check exit conditions → emit SELL if triggered.
          4. Check entry conditions → emit BUY if z-score signals mean-reversion.
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

        # 2. Auto-recover state after restart (positions are source of truth)
        if self.active_token_id is None:
            for tid in (self._yes_token_id, self._no_token_id):
                pos = by_token.get(tid)
                if pos and pos.size > 0:
                    self.active_token_id  = tid
                    self.entry_price      = pos.average_price
                    self.entry_timestamp  = now_ts   # conservative: treat as just entered
                    self.entry_size       = pos.size
                    break

        # 3. Exit check — always before entry
        exit_sig = self.check_exit(order_books, by_token, now_ts)
        if exit_sig:
            self._pending_exit_entry_price = self.entry_price
            self._pending_exit_entry_size  = self.entry_size
            self._reset_position_state()
            return [exit_sig]

        # 4. Entry check
        if self.active_token_id is None:
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
          - Z-score on YES token detects mean-reversion signal.
          - No existing position.
          - Z-score triggers when |z| >= threshold:
              z < -threshold  → YES is cheap (bearish overreaction) → buy YES
              z > +threshold  → YES is expensive (bullish overreaction) → buy NO
        On confirmation, state is set optimistically so exit logic
        fires on the very next cycle.
        """
        # Compute z-score on YES token
        yes_zscore = self._compute_zscore(self._yes_token_id, now_ts)
        if yes_zscore is None:
            return None  # insufficient history, warmup, or flat market

        # Check entry threshold
        if abs(yes_zscore) < self.zscore_threshold:
            return None  # z-score not extreme enough

        # Determine target token based on z-score sign
        if yes_zscore < -self.zscore_threshold:
            # YES is cheap → bullish reversal expected → buy YES
            token_id = self._yes_token_id
            direction = "bullish_reversion"
        else:
            # YES is expensive → bearish reversal expected → buy NO
            token_id = self._no_token_id
            direction = "bearish_reversion"

        pos = by_token.get(token_id)
        if pos and pos.size > 0:
            return None   # already holding — no pyramiding

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
        size       = round(self.position_size_usdc / best_ask, 2)
        # Confidence scales with z-score magnitude: baseline 0.5, up to 1.0
        confidence = min(0.5 + (abs(yes_zscore) / self.zscore_threshold) * 0.5, 1.0)

        # Optimistically record position state
        self.active_token_id  = token_id
        self.entry_price      = best_ask
        self.entry_timestamp  = now_ts
        self.entry_size       = size

        return Signal(
            market_id=self._market_id,
            outcome=self._outcome_map.get(token_id, ""),
            action="BUY",
            confidence=confidence,
            price=round_to_tick(best_ask, tick),
            size=size,
            reason=(
                f"vol_reversion fading: {direction} zscore={yes_zscore:.2f} "
                f"threshold={self.zscore_threshold:.2f} "
                f"over {self.window_seconds}s"
            ),
        )

    def _compute_zscore(self, token_id: str, now_ts: float) -> Optional[float]:
        """
        Compute z-score of current price within rolling window.

        Returns z-score = (current_price - mean) / std over the window,
        or None if:
          - Insufficient history (< min_samples)
          - Price volatility too low (std < min_vol_threshold)
          - No current price in order book
        """
        history = self._price_history.get(token_id, [])

        # Extract prices within window
        window_start = now_ts - self.window_seconds
        prices_in_window = [mid for ts, mid in history if ts >= window_start]

        # Warmup guard
        if len(prices_in_window) < self.min_samples:
            return None

        # Compute mean and std
        mean = sum(prices_in_window) / len(prices_in_window)
        variance = sum((p - mean) ** 2 for p in prices_in_window) / len(prices_in_window)
        std = math.sqrt(variance)

        # Guard: skip if market is too flat (no volatility edge)
        if std < self.min_vol_threshold:
            return None

        # Z-score of current (latest) price
        current_price = prices_in_window[-1]
        zscore = (current_price - mean) / std

        return zscore

