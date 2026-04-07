"""Logistic regression edge strategy — hold to expiry.

Entry rule:
    BUY YES  when  p_hat > q_t + c_t + delta   (market underprices Up)
    BUY NO   when  p_hat < q_t - c_t - delta   (market overprices Up)

    where:
      p_hat = LogRegModel predicted P(Up)
      q_t   = market mid price for Up token
      c_t   = half-spread cost = (up_ask - up_bid) / 2
      delta = minimum margin of safety (configurable)

Exit rule:
    Hold to expiry — binary resolution.

Position sizing: fractional Kelly, capped at position_size_usdc.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .base import Signal, Strategy
from ..api.types import OrderBook, Position
from ..models.logreg_model import LogRegModel
from ..utils.market_utils import get_mid_price, round_to_tick, spread_pct


class LogRegEdgeStrategy(Strategy):
    """Trade when the LR model disagrees with the market by more than delta + costs."""

    def __init__(
        self,
        config: Dict[str, Any],
        btc_feed=None,
        model_service: Optional[LogRegModel] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="logreg_edge", config=config)
        self.logger = logger or logging.getLogger(__name__)
        self.btc_feed = btc_feed
        self.model_service = model_service

        # Entry filters
        self.delta: float = float(config.get("delta", 0.05))
        self.max_spread_pct: float = float(config.get("max_spread_pct", 0.06))
        self.min_seconds_to_expiry: float = float(config.get("min_seconds_to_expiry", 10.0))
        self.max_seconds_to_expiry: float = float(config.get("max_seconds_to_expiry", 295.0))

        # Exit: hold to expiry
        self.exit_rule = "hold_to_expiry"
        self.stop_loss_pct: float = float(config.get("stop_loss_pct", 999.0))
        self.max_hold_seconds: int = int(config.get("max_hold_seconds", 300))
        self.profit_target_pct: float = float(config.get("profit_target_pct", 999.0))

        # Sizing
        self.kelly_fraction: float = float(config.get("kelly_fraction", 0.15))
        self.position_size_usdc: float = float(config.get("position_size_usdc", 30.0))

        # Observable state
        self.last_prob_yes: Optional[float] = None
        self.last_edge_yes: Optional[float] = None
        self.last_edge_no: Optional[float] = None
        self.last_net_edge_yes: Optional[float] = None
        self.last_net_edge_no: Optional[float] = None
        self.last_q_t: Optional[float] = None
        self.last_c_t: Optional[float] = None
        self.last_tte_seconds: Optional[float] = None
        self.last_feature_status: str = ""
        self.last_model_version: str = ""

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        if not self._yes_token_id or not self._no_token_id:
            return []

        self.last_skip_reason = ""

        if self.model_service is None or not self.model_service.ready:
            self.last_skip_reason = "model_not_ready"
            return []

        order_books: Dict[str, OrderBook] = market_data.get("order_books", {})
        positions: List[Position] = market_data.get("positions", [])
        by_token: Dict[str, Position] = {p.token_id: p for p in positions}
        balance: float = float(market_data.get("balance") or 0.0)
        now_mono = time.monotonic()
        now_wall = time.time()

        # Record price history
        for tid in (self._yes_token_id, self._no_token_id):
            book = order_books.get(tid)
            if book:
                mid = get_mid_price(book)
                if mid:
                    self.record_price(tid, mid, now_mono)

        self._auto_recover_position(by_token, now_mono)

        # If already in a position, hold to expiry (no exit logic)
        if self.active_token_id is not None:
            self.last_skip_reason = "in_position"
            return []

        return self._check_entry(order_books, by_token, market_data, now_wall, balance)

    def should_enter(self, signal: Signal) -> bool:
        return self.validate_signal(signal) and signal.confidence >= self.min_confidence

    # ------------------------------------------------------------------
    # Entry logic: p_hat vs q_t +/- c_t +/- delta
    # ------------------------------------------------------------------

    def _check_entry(
        self,
        order_books: Dict[str, OrderBook],
        by_token: Dict[str, Position],
        market_data: Dict[str, Any],
        now_wall: float,
        balance: float,
    ) -> List[Signal]:
        # Time-to-expiry gate
        slot_expiry_ts = float(market_data.get("slot_expiry_ts") or 0.0)
        tte = max(0.0, slot_expiry_ts - now_wall)
        self.last_tte_seconds = tte
        if tte < self.min_seconds_to_expiry or tte > self.max_seconds_to_expiry:
            self.last_skip_reason = f"tte: {tte:.0f}s (range {self.min_seconds_to_expiry:.0f}-{self.max_seconds_to_expiry:.0f})"
            return []

        yes_book = order_books.get(self._yes_token_id)
        no_book = order_books.get(self._no_token_id)
        if yes_book is None or no_book is None:
            self.last_skip_reason = "missing_orderbook"
            return []

        yes_bid = float(yes_book.bids[0].price) if yes_book.bids else 0.0
        yes_ask = float(yes_book.asks[0].price) if yes_book.asks else 0.0
        no_bid = float(no_book.bids[0].price) if no_book.bids else 0.0
        no_ask = float(no_book.asks[0].price) if no_book.asks else 0.0

        if yes_ask <= 0 or no_ask <= 0:
            self.last_skip_reason = "invalid_prices"
            return []

        # Market-implied probability and half-spread cost
        q_t = (yes_bid + yes_ask) / 2.0  # up mid
        c_t = (yes_ask - yes_bid) / 2.0  # half-spread

        self.last_q_t = q_t
        self.last_c_t = c_t

        # Predict
        prediction = self._predict(order_books, market_data, now_wall)
        if prediction is None or prediction.prob_yes is None:
            self.last_skip_reason = "no_prediction"
            return []

        p_hat = prediction.prob_yes
        self.last_prob_yes = p_hat

        # Edge computation
        edge_yes = p_hat - q_t - c_t  # model thinks Up is more likely than market
        edge_no = q_t - p_hat - c_t   # model thinks Down is more likely than market
        self.last_edge_yes = edge_yes
        self.last_edge_no = edge_no
        self.last_net_edge_yes = edge_yes
        self.last_net_edge_no = edge_no

        # Entry decision: pick the side with higher edge, but must exceed delta
        if edge_yes >= edge_no and edge_yes >= self.delta:
            side = "YES"
            edge = edge_yes
            entry_ask = yes_ask
            book = yes_book
            prob = p_hat
        elif edge_no > edge_yes and edge_no >= self.delta:
            side = "NO"
            edge = edge_no
            entry_ask = no_ask
            book = no_book
            prob = 1.0 - p_hat
        else:
            best = max(edge_yes, edge_no)
            self.last_skip_reason = f"edge_low: best={best:+.3f} < delta={self.delta}"
            return []

        # Spread filter
        if side == "YES" and spread_pct(yes_bid, yes_ask) > self.max_spread_pct:
            self.last_skip_reason = f"spread_wide: YES {spread_pct(yes_bid, yes_ask)*100:.1f}% > {self.max_spread_pct*100:.0f}%"
            return []
        if side == "NO" and spread_pct(no_bid, no_ask) > self.max_spread_pct:
            self.last_skip_reason = f"spread_wide: NO {spread_pct(no_bid, no_ask)*100:.1f}% > {self.max_spread_pct*100:.0f}%"
            return []

        # Price sanity
        if entry_ask < self.min_entry_price or entry_ask > self.max_entry_price:
            self.last_skip_reason = "price_out_of_range"
            return []

        # Must be flat
        if not self.is_flat(by_token):
            self.last_skip_reason = "in_position"
            return []

        # Kelly sizing
        size_usdc = self._kelly_size(prob, entry_ask, balance)
        if size_usdc <= 0:
            self.last_skip_reason = "kelly_size_zero"
            return []

        shares = round(size_usdc / entry_ask, 2)
        if shares <= 0:
            self.last_skip_reason = "kelly_size_zero"
            return []

        tick = book.tick_size or 0.001
        confidence = min(1.0, max(self.min_confidence, prob))
        token_id = self._yes_token_id if side == "YES" else self._no_token_id

        self.active_token_id = token_id
        self.entry_price = entry_ask
        self.entry_timestamp = time.monotonic()
        self.entry_size = shares

        return [Signal(
            market_id=self._market_id,
            outcome=side,
            action="BUY",
            confidence=confidence,
            price=round_to_tick(entry_ask, tick),
            size=shares,
            reason=(
                f"logreg p_hat={p_hat:.3f} q_t={q_t:.3f} c_t={c_t:.3f} "
                f"edge_{side.lower()}={edge:+.3f} delta={self.delta} tte={tte:.0f}s"
            ),
        )]

    # ------------------------------------------------------------------
    # Prediction (delegates to model_service)
    # ------------------------------------------------------------------

    def _predict(
        self,
        order_books: Dict[str, OrderBook],
        market_data: Dict[str, Any],
        now_wall: float,
    ):
        self.last_feature_status = ""

        btc_prices = []
        if self.btc_feed is not None:
            btc_prices = getattr(self.btc_feed, "get_recent_prices", lambda w=300: [])(300)
            # Guard against trading on stale BTC data (feed dead >60s)
            if btc_prices and (now_wall - btc_prices[-1][0]) > 60:
                self.logger.warning("btc_feed stale (>60s), skipping prediction")
                self.last_feature_status = "stale_btc"
                return None

        # Pass monotonic yes_history directly — _safe_return uses relative
        # offsets so any consistent timestamp base works.
        yes_history = list(self._price_history.get(self._yes_token_id) or [])
        mono_now = time.monotonic() if yes_history else now_wall

        snapshot = {
            "btc_prices": btc_prices,
            "yes_book": order_books.get(self._yes_token_id),
            "no_book": order_books.get(self._no_token_id),
            "yes_history": yes_history,
            "question": market_data.get("question", ""),
            "strike_price": market_data.get("strike_price"),
            "slot_expiry_ts": market_data.get("slot_expiry_ts"),
            "now_ts": mono_now,
        }

        try:
            prediction = self.model_service.predict(snapshot)
        except Exception as exc:
            self.logger.warning("logreg_edge: predict failed: %s", exc)
            self.last_feature_status = "predict_error"
            return None

        self.last_feature_status = prediction.feature_status
        self.last_model_version = prediction.model_version
        return prediction

    # ------------------------------------------------------------------
    # Kelly sizing (same formula as ProbEdgeStrategy)
    # ------------------------------------------------------------------

    def _kelly_size(self, prob: float, price: float, balance: float) -> float:
        """Fractional Kelly notional for a binary contract.

        f_kelly = (p - x) / (1 - x)
        f_used  = kelly_fraction * f_kelly
        usdc    = f_used * balance, capped at position_size_usdc
        """
        if price <= 0 or price >= 1 or prob <= price:
            return 0.0
        f_full = (prob - price) / (1.0 - price)
        f_used = max(0.0, self.kelly_fraction * f_full)
        usdc = f_used * balance if balance > 0 else self.position_size_usdc
        return min(usdc, self.position_size_usdc)
