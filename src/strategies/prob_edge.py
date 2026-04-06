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
        self.fee_enabled: bool = bool(config.get("fee_enabled", False))
        self.taker_fee_rate: float = float(config.get("taker_fee_rate", 0.072))
        self.taker_fee_exponent: float = float(config.get("taker_fee_exponent", 1.0))

        # ── Re-entry config ───────────────────────────────────────────────────
        self.allow_reentry: bool = bool(config.get("allow_reentry", True))
        self.max_trades_per_slot: int = int(config.get("max_trades_per_slot", 3))
        self.re_entry_edge_mult: float = float(config.get("re_entry_edge_mult", 1.5))
        self.re_entry_size_mult: float = float(config.get("re_entry_size_mult", 0.5))
        self.slot_loss_limit_usdc: float = float(config.get("slot_loss_limit_usdc", 15.0))

        # Per-slot counters — reset by reset_slot_state() on market rollover
        self._slot_trade_count: int = 0
        self._slot_realized_pnl: float = 0.0
        self._slot_blocked_direction: Optional[str] = None  # "YES" or "NO"

        # ── Observable state (written every cycle — read by dashboard) ─────────
        self.last_prob_yes: Optional[float] = None
        self.last_prob_no: Optional[float] = None
        self.last_edge_yes: Optional[float] = None
        self.last_edge_no: Optional[float] = None
        self.last_net_edge_yes: Optional[float] = None
        self.last_net_edge_no: Optional[float] = None
        self.last_expected_fill_yes: Optional[float] = None
        self.last_expected_fill_no: Optional[float] = None
        self.last_tte_seconds: Optional[float] = None
        self.last_distance_to_break_pct: Optional[float] = None
        self.last_distance_to_strike_bps: Optional[float] = None
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

        self.last_skip_reason = ""
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
            # Record slot-level accounting before resetting position state
            self._slot_trade_count += 1
            if self.entry_price is not None and self.entry_size is not None:
                trade_pnl = (exit_sig.price - self.entry_price) * self.entry_size
                self._slot_realized_pnl += trade_pnl
            if "stop_loss" in exit_sig.reason:
                self._slot_blocked_direction = exit_sig.outcome
            self._reset_position_state()
            return [exit_sig]

        if self.active_token_id is not None:
            self.last_skip_reason = "in_position"
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
        self.last_prob_no = None
        self.last_edge_yes = None
        self.last_edge_no = None
        self.last_net_edge_yes = None
        self.last_net_edge_no = None
        self.last_expected_fill_yes = None
        self.last_expected_fill_no = None
        self.last_tte_seconds = None
        self.last_distance_to_break_pct = None
        self.last_distance_to_strike_bps = None
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

        strike_price = snapshot.get("strike_price")
        if btc_prices:
            try:
                btc_now = float(btc_prices[-1][1])
            except (IndexError, TypeError, ValueError):
                btc_now = None
        else:
            btc_now = None
        try:
            strike = float(strike_price) if strike_price is not None else None
        except (TypeError, ValueError):
            strike = None
        try:
            slot_expiry_ts = float(snapshot.get("slot_expiry_ts") or 0.0)
        except (TypeError, ValueError):
            slot_expiry_ts = 0.0
        if btc_now is not None and strike is not None and strike > 0:
            distance_pct = (btc_now - strike) / strike
            self.last_distance_to_break_pct = distance_pct
            self.last_distance_to_strike_bps = distance_pct * 10_000.0

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
        self.last_prob_no = prediction.prob_no
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
            self.last_skip_reason = "no_prediction"
            return []

        # ── Re-entry guards ────────────────────────────────────────────────────
        is_reentry = self._slot_trade_count > 0
        if is_reentry:
            if not self.allow_reentry:
                self.last_skip_reason = "reentry_blocked"
                return []
            if self._slot_trade_count >= self.max_trades_per_slot:
                self.last_skip_reason = f"slot_trade_cap: {self._slot_trade_count}/{self.max_trades_per_slot}"
                self.logger.debug(
                    "prob_edge: slot trade cap reached (%d/%d), skipping entry",
                    self._slot_trade_count, self.max_trades_per_slot,
                )
                return []
            if self._slot_realized_pnl < -self.slot_loss_limit_usdc:
                self.last_skip_reason = f"slot_loss_cap: pnl=${self._slot_realized_pnl:.2f}"
                self.logger.debug(
                    "prob_edge: slot loss cap hit (pnl=%.2f), skipping entry",
                    self._slot_realized_pnl,
                )
                return []

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
        no_bid  = float(no_book.bids[0].price)  if no_book.bids  else 0.0
        no_ask  = float(no_book.asks[0].price)  if no_book.asks  else 0.0

        if yes_ask <= 0 or no_ask <= 0:
            self.last_skip_reason = "invalid_prices"
            return []

        # last_edge_yes/no already set in _predict(); raw edge not re-computed here

        # Elevate edge requirement on re-entries to compensate for within-slot correlation
        edge_mult = self.re_entry_edge_mult if is_reentry else 1.0
        required = self._required_edge(tte) * edge_mult
        self.last_required_edge = required

        tentative_yes_usdc = self._kelly_size(prediction.prob_yes, yes_ask, balance)
        tentative_no_usdc = self._kelly_size(prediction.prob_no, no_ask, balance)
        if is_reentry:
            tentative_yes_usdc *= self.re_entry_size_mult
            tentative_no_usdc *= self.re_entry_size_mult

        self.last_expected_fill_yes = self._estimate_buy_vwap(yes_book, tentative_yes_usdc)
        self.last_expected_fill_no = self._estimate_buy_vwap(no_book, tentative_no_usdc)
        net_edge_yes = self._net_edge(prediction.prob_yes, self.last_expected_fill_yes)
        net_edge_no = self._net_edge(prediction.prob_no, self.last_expected_fill_no)
        self.last_net_edge_yes = net_edge_yes
        self.last_net_edge_no = net_edge_no

        candidates: List[Tuple[str, OrderBook, float, float]] = []
        if (
            net_edge_yes >= required
            and spread_pct(yes_bid, yes_ask) <= self.max_spread_pct
            and self.min_entry_price <= yes_ask <= self.max_entry_price
        ):
            candidates.append(("YES", yes_book, yes_ask, net_edge_yes))

        if (
            net_edge_no >= required
            and spread_pct(no_bid, no_ask) <= self.max_spread_pct
            and self.min_entry_price <= no_ask <= self.max_entry_price
        ):
            candidates.append(("NO", no_book, no_ask, net_edge_no))

        # Remove direction blocked by a prior stop-loss exit this slot
        if self._slot_blocked_direction and candidates:
            candidates = [c for c in candidates if c[0] != self._slot_blocked_direction]
            if not candidates:
                self.last_skip_reason = f"direction_blocked: {self._slot_blocked_direction}"
                self.logger.debug(
                    "prob_edge: re-entry blocked on %s (stop-loss cooldown)",
                    self._slot_blocked_direction,
                )

        if not candidates:
            best_net = max(net_edge_yes, net_edge_no)
            yes_sprd = spread_pct(yes_bid, yes_ask) if yes_bid > 0 else float("inf")
            no_sprd = spread_pct(no_bid, no_ask) if no_bid > 0 else float("inf")
            best_sprd = min(yes_sprd, no_sprd)
            if not self.last_skip_reason:
                if best_sprd > self.max_spread_pct:
                    self.last_skip_reason = f"spread_wide: {best_sprd*100:.1f}% > {self.max_spread_pct*100:.0f}%"
                elif best_net < required:
                    self.last_skip_reason = f"edge_low: net={best_net:+.3f} < req={required:.3f}"
                else:
                    self.last_skip_reason = "price_out_of_range"
            return []
        if not self.is_flat(by_token):
            self.last_skip_reason = "in_position"
            self.logger.debug("prob_edge: edge found but already in position, skipping entry")
            return []

        outcome, book, best_ask, net_edge = max(candidates, key=lambda c: c[3])
        token_id = self._yes_token_id if outcome == "YES" else self._no_token_id
        prob = prediction.prob_yes if outcome == "YES" else prediction.prob_no

        # Reduce Kelly size on re-entries to account for within-slot correlation
        size_mult = self.re_entry_size_mult if is_reentry else 1.0
        size_usdc = self._kelly_size(prob, best_ask, balance) * size_mult
        if size_usdc <= 0:
            self.last_skip_reason = "kelly_size_zero"
            return []

        # Reuse VWAP already computed for this side above (same size_usdc)
        expected_fill = (
            self.last_expected_fill_yes if outcome == "YES" else self.last_expected_fill_no
        )
        if expected_fill is None:
            self.last_skip_reason = "fill_estimate_failed"
            return []
        size = round(size_usdc / expected_fill, 2)
        if size <= 0:
            self.last_skip_reason = "fill_estimate_failed"
            return []

        tick = book.tick_size or 0.001
        confidence = min(1.0, max(self.min_confidence, prob))

        self.active_token_id = token_id
        self.entry_price = expected_fill
        self.entry_timestamp = now_mono
        self.entry_size = size

        reentry_tag = f" reentry={self._slot_trade_count}" if is_reentry else ""
        return [Signal(
            market_id=self._market_id,
            outcome=outcome,
            action="BUY",
            confidence=confidence,
            price=round_to_tick(expected_fill, tick),
            size=size,
            reason=(
                f"prob={prob:.3f} net_edge={net_edge:+.3f} "
                f"tte={tte:.0f}s status={prediction.feature_status}{reentry_tag}"
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
        if self.exit_rule == "hold_to_expiry":
            return None
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
    # Slot state management
    # ──────────────────────────────────────────────────────────────────────────

    def reset_slot_state(self) -> None:
        """Reset per-slot counters. Call on market rollover (new market_id)."""
        self._slot_trade_count = 0
        self._slot_realized_pnl = 0.0
        self._slot_blocked_direction = None

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamic edge threshold
    # ──────────────────────────────────────────────────────────────────────────

    def _required_edge(self, tte: float) -> float:
        """Dynamic min-edge scaled by time remaining in the slot.

        Near expiry (tte→0): spread widens, book thins, gamma is high — require MORE edge.
        With plenty of time (tte→max): model has time to be right — allow lower threshold.

        At tte=0:                     required = 2 × min_edge  (coin-flip zone)
        At tte=max_seconds_to_expiry: required = 1 × min_edge  (plenty of time)

        Formula: min_edge × (1 + (1 - tte / max_seconds_to_expiry))
        """
        tte_clamped = max(0.0, min(tte, self.max_seconds_to_expiry))
        return self.min_edge * (1.0 + (1.0 - tte_clamped / self.max_seconds_to_expiry))

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

    def _estimate_buy_vwap(self, book: OrderBook, size_usdc: float) -> Optional[float]:
        """Estimate average fill price for a marketable buy from ask depth."""
        if size_usdc <= 0 or not book.asks:
            return None

        remaining_usdc = size_usdc
        filled_shares = 0.0
        spent_usdc = 0.0

        for level in book.asks:
            ask = float(level.price)
            available_shares = max(0.0, float(level.size))
            if ask <= 0 or available_shares <= 0:
                continue

            level_capacity_usdc = ask * available_shares
            take_usdc = min(remaining_usdc, level_capacity_usdc)
            take_shares = take_usdc / ask
            spent_usdc += take_usdc
            filled_shares += take_shares
            remaining_usdc -= take_usdc
            if remaining_usdc <= 1e-9:
                break

        if filled_shares <= 0:
            return None
        return spent_usdc / filled_shares

    def _taker_fee_per_share(self, price: float) -> float:
        if not self.fee_enabled or price <= 0.0 or price >= 1.0:
            return 0.0
        return self.taker_fee_rate * price * ((price * (1.0 - price)) ** self.taker_fee_exponent)

    def _net_edge(self, prob: float, expected_fill: Optional[float]) -> float:
        """Compute net edge given a pre-computed expected fill price."""
        if expected_fill is None:
            return float("-inf")
        fee_per_share = self._taker_fee_per_share(expected_fill)
        return prob - expected_fill - fee_per_share
