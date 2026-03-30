"""Probability-native edge-based strategy — the production strategy path.

Decision rule (no momentum, no directional bias):

    edge_yes = p_up  - yes_ask   →  BUY YES  if edge_yes >= min_edge
    edge_no  = p_no  - no_ask    →  BUY NO   if edge_no  >= min_edge
    otherwise: NO TRADE

If both sides clear the threshold the higher-edge side is taken.

Position sizing: fractional Kelly, capped at position_size_usdc.
Exit: stop-loss, time-limit, or edge-reprice (model reprices held side below exit_edge).

Model-agnostic: pass any object with predict(snapshot) -> PredictionResult
as model_service.  Defaults to BTCUpDownBaselineModel (GBM) when none is supplied.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import Signal, Strategy
from ..api.types import OrderBook, Position
from ..models.baseline_model import BTCUpDownBaselineModel
from ..utils.market_utils import get_mid_price, round_to_tick, spread_pct


class ProbEdgeStrategy(Strategy):
    """Trade only when the model-implied edge exceeds the market ask price.

    This is the canonical production strategy.  Subclasses or callers that
    want to swap in a different model (XGBoost, Poisson, NN …) should pass
    model_service.  The rest of the pipeline — entry filter, Kelly sizing,
    exit logic — stays fixed.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        btc_feed=None,
        model_service=None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="prob_edge", config=config)
        self.logger = logger or logging.getLogger(__name__)
        self.btc_feed = btc_feed
        self.model_service = model_service or BTCUpDownBaselineModel(logger=self.logger)

        # ── Entry filters ──────────────────────────────────────────────────────
        self.min_edge: float = float(config.get("min_edge", 0.04))
        self.max_spread_pct: float = float(config.get("max_spread_pct", 0.06))
        self.min_seconds_to_expiry: float = float(config.get("min_seconds_to_expiry", 30.0))
        self.max_seconds_to_expiry: float = float(config.get("max_seconds_to_expiry", 280.0))

        # ── Exit parameters ────────────────────────────────────────────────────
        self.exit_edge: float = float(config.get("exit_edge", -0.02))
        self.stop_loss_pct: float = float(config.get("stop_loss_pct", 0.15))
        self.max_hold_seconds: int = int(config.get("max_hold_seconds", 270))

        # ── Sizing ────────────────────────────────────────────────────────────
        self.kelly_fraction: float = float(config.get("kelly_fraction", 0.15))
        self.position_size_usdc: float = float(config.get("position_size_usdc", 30.0))

        # ── Observable state (written every cycle — read by dashboard) ─────────
        self.last_prob_yes: Optional[float] = None
        self.last_edge_yes: Optional[float] = None
        self.last_edge_no: Optional[float] = None
        self.last_feature_status: str = ""
        self.last_model_version: str = getattr(self.model_service, "model_version", "")
        self.last_score_breakdown: Optional[dict] = None
        self.last_required_edge: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Core interface
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        if not self._yes_token_id or not self._no_token_id:
            return []

        order_books: Dict[str, OrderBook] = market_data.get("order_books", {})
        positions: List[Position] = market_data.get("positions", [])
        by_token: Dict[str, Position] = {p.token_id: p for p in positions}
        balance: float = float(market_data.get("balance") or 0.0)
        now_mono = time.monotonic()
        now_wall = time.time()

        for tid in (self._yes_token_id, self._no_token_id):
            book = order_books.get(tid)
            if book:
                mid = get_mid_price(book)
                if mid:
                    self.record_price(tid, mid, now_mono)

        self._auto_recover_position(by_token, now_mono)

        prediction = self._predict(order_books, market_data, now_wall)

        exit_sig = self._check_exit(order_books, by_token, now_mono, prediction)
        if exit_sig:
            self._pending_exit_entry_price = self.entry_price
            self._pending_exit_entry_size = self.entry_size
            self._reset_position_state()
            return [exit_sig]

        if self.active_token_id is not None:
            return []

        return self._check_entry(
            order_books, by_token, market_data, now_mono, now_wall, prediction, balance
        )

    def should_enter(self, signal: Signal) -> bool:
        return self.validate_signal(signal) and signal.confidence >= self.min_confidence

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def _predict(
        self,
        order_books: Dict[str, OrderBook],
        market_data: Dict[str, Any],
        now_wall: float,
    ):
        """Build snapshot, call model.predict(), cache observables. Returns None on failure."""
        self.last_prob_yes = None
        self.last_edge_yes = None
        self.last_edge_no = None
        self.last_score_breakdown = None

        yes_book = order_books.get(self._yes_token_id)
        no_book = order_books.get(self._no_token_id)

        btc_prices = []
        if self.btc_feed is not None and getattr(self.btc_feed, "is_healthy", lambda: False)():
            btc_prices = getattr(self.btc_feed, "get_recent_prices", lambda w=300: [])(300)

        snapshot = {
            "btc_prices": btc_prices,
            "yes_book": yes_book,
            "no_book": no_book,
            "yes_history": self._price_history.get(self._yes_token_id, []),
            "no_history": self._price_history.get(self._no_token_id, []),
            "question": market_data.get("question", ""),
            "strike_price": market_data.get("strike_price"),
            "slot_expiry_ts": market_data.get("slot_expiry_ts"),
            "now_ts": now_wall,
        }

        try:
            prediction = self.model_service.predict(snapshot)
        except Exception as exc:
            self.logger.warning("prob_edge: model.predict() raised: %s", exc)
            self.last_feature_status = "model_error"
            return None

        self.last_feature_status = prediction.feature_status
        self.last_model_version = prediction.model_version

        if prediction.prob_yes is None:
            return None

        yes_ask = float(yes_book.asks[0].price) if yes_book and yes_book.asks else None
        no_ask = float(no_book.asks[0].price) if no_book and no_book.asks else None
        self.last_prob_yes = prediction.prob_yes
        if yes_ask:
            self.last_edge_yes = prediction.prob_yes - yes_ask
        if no_ask and prediction.prob_no is not None:
            self.last_edge_no = prediction.prob_no - no_ask
        self.last_score_breakdown = getattr(self.model_service, "last_breakdown", None)

        return prediction

    # ──────────────────────────────────────────────────────────────────────────
    # Entry
    # ──────────────────────────────────────────────────────────────────────────

    def _check_entry(
        self,
        order_books: Dict[str, OrderBook],
        by_token: Dict[str, Position],
        market_data: Dict[str, Any],
        now_mono: float,
        now_wall: float,
        prediction,
        balance: float,
    ) -> List[Signal]:
        if prediction is None or prediction.prob_yes is None or prediction.prob_no is None:
            return []

        slot_expiry_ts = float(market_data.get("slot_expiry_ts") or 0.0)
        tte = max(0.0, slot_expiry_ts - now_wall)
        if tte < self.min_seconds_to_expiry or tte > self.max_seconds_to_expiry:
            return []

        yes_book = order_books.get(self._yes_token_id)
        no_book = order_books.get(self._no_token_id)
        if yes_book is None or no_book is None:
            return []

        yes_bid = float(yes_book.bids[0].price) if yes_book.bids else 0.0
        yes_ask = float(yes_book.asks[0].price) if yes_book.asks else 0.0
        no_bid  = float(no_book.bids[0].price)  if no_book.bids  else 0.0
        no_ask  = float(no_book.asks[0].price)  if no_book.asks  else 0.0

        if yes_ask <= 0 or no_ask <= 0:
            return []

        edge_yes = prediction.prob_yes - yes_ask
        edge_no  = prediction.prob_no  - no_ask
        self.last_edge_yes = edge_yes
        self.last_edge_no  = edge_no

        required = self._required_edge(tte)
        self.last_required_edge = required

        candidates: List[Tuple[str, OrderBook, float, float]] = []
        if (
            edge_yes >= required
            and spread_pct(yes_bid, yes_ask) <= self.max_spread_pct
            and self.min_entry_price <= yes_ask <= self.max_entry_price
        ):
            candidates.append(("YES", yes_book, yes_ask, edge_yes))

        if (
            edge_no >= required
            and spread_pct(no_bid, no_ask) <= self.max_spread_pct
            and self.min_entry_price <= no_ask <= self.max_entry_price
        ):
            candidates.append(("NO", no_book, no_ask, edge_no))

        if not candidates:
            return []
        if not self.is_flat(by_token):
            self.logger.debug("prob_edge: edge found but already in position, skipping entry")
            return []

        outcome, book, best_ask, edge = max(candidates, key=lambda c: c[3])
        token_id = self._yes_token_id if outcome == "YES" else self._no_token_id
        prob = prediction.prob_yes if outcome == "YES" else prediction.prob_no

        size_usdc = self._kelly_size(prob, best_ask, balance)
        if size_usdc <= 0:
            return []

        tick = book.tick_size or 0.001
        size = round(size_usdc / best_ask, 2)
        confidence = min(1.0, max(self.min_confidence, prob))

        self.active_token_id = token_id
        self.entry_price = best_ask
        self.entry_timestamp = now_mono
        self.entry_size = size

        return [Signal(
            market_id=self._market_id,
            outcome=outcome,
            action="BUY",
            confidence=confidence,
            price=round_to_tick(best_ask, tick),
            size=size,
            reason=(
                f"prob={prob:.3f} edge={edge:+.3f} "
                f"tte={tte:.0f}s status={prediction.feature_status}"
            ),
        )]

    # ──────────────────────────────────────────────────────────────────────────
    # Exit
    # ──────────────────────────────────────────────────────────────────────────

    def _check_exit(
        self,
        order_books: Dict[str, OrderBook],
        by_token: Dict[str, Position],
        now_mono: float,
        prediction,
    ) -> Optional[Signal]:
        if self.active_token_id is None or self.entry_price is None:
            return None

        pos = by_token.get(self.active_token_id)
        held_size = pos.size if pos and pos.size > 0 else (
            self.entry_size if self.entry_size is not None else 0.0
        )
        if held_size <= 0:
            self._reset_position_state()
            return None

        book = order_books.get(self.active_token_id)
        if book is None:
            return None

        mid = get_mid_price(book)
        best_bid = float(book.bids[0].price) if book.bids else (mid or 0.0)
        if best_bid <= 0:
            return None

        pnl_pct   = (best_bid - self.entry_price) / self.entry_price
        time_held = now_mono - self.entry_timestamp if self.entry_timestamp else 0.0
        outcome   = self._outcome_map.get(self.active_token_id, "")

        reason: Optional[str] = None

        if -pnl_pct >= self.stop_loss_pct:
            reason = f"stop_loss loss={-pnl_pct:.2%} >= limit={self.stop_loss_pct:.2%}"

        elif time_held >= self.max_hold_seconds:
            reason = f"time_limit held={time_held:.0f}s >= max={self.max_hold_seconds}s"

        elif (
            prediction is not None
            and prediction.prob_yes is not None
            and prediction.prob_no is not None
        ):
            # Edge-reprice: re-evaluate the held side at current best bid.
            # If held_edge <= exit_edge the position has lost its original rationale.
            held_edge = (
                prediction.prob_yes - best_bid
                if outcome == "YES"
                else prediction.prob_no - best_bid
            )
            if held_edge <= self.exit_edge:
                reason = (
                    f"edge_reprice held_edge={held_edge:+.3f} "
                    f"<= exit={self.exit_edge:+.3f} pnl={pnl_pct:+.2%}"
                )

        if reason is None:
            return None

        tick = book.tick_size or 0.001
        return Signal(
            market_id=self._market_id,
            outcome=outcome,
            action="SELL",
            confidence=1.0,
            price=round_to_tick(best_bid, tick),
            size=held_size,
            reason=reason,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamic edge threshold
    # ──────────────────────────────────────────────────────────────────────────

    def _required_edge(self, tte: float) -> float:
        """Dynamic min-edge scaled by time remaining in the slot.

        At tte=max_seconds_to_expiry: required = 2 × min_edge  (most uncertain)
        At tte=0:                     required = 1 × min_edge  (least uncertain)

        Formula: min_edge × (1 + tte / max_seconds_to_expiry)
        """
        tte_clamped = max(0.0, min(tte, self.max_seconds_to_expiry))
        return self.min_edge * (1.0 + tte_clamped / self.max_seconds_to_expiry)

    # ──────────────────────────────────────────────────────────────────────────
    # Kelly sizing
    # ──────────────────────────────────────────────────────────────────────────

    def _kelly_size(self, prob: float, price: float, balance: float) -> float:
        """Fractional Kelly notional (USDC) for a binary contract.

        Full Kelly:    f = (p - x) / (1 - x)
        Fractional:    f_used = kelly_fraction * f
        Notional:      usdc = f_used * balance   (capped at position_size_usdc)
        """
        if price <= 0 or price >= 1 or prob <= price:
            return 0.0
        f_full = (prob - price) / (1.0 - price)
        f_used = max(0.0, self.kelly_fraction * f_full)
        usdc = f_used * balance if balance > 0 else self.position_size_usdc
        return min(usdc, self.position_size_usdc)
