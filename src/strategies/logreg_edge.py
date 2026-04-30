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

import collections
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from ._edge_stability import EdgeStabilityTracker
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
        self.delta: float = float(config.get("delta", 0.025))
        self.max_spread_pct: float = float(config.get("max_spread_pct", 0.06))
        self.min_seconds_to_expiry: float = float(config.get("min_seconds_to_expiry", 10.0))
        self.max_seconds_to_expiry: float = float(config.get("max_seconds_to_expiry", 300.0))

        # Exit: hold to expiry
        self.exit_rule = "hold_to_expiry"
        self.stop_loss_pct: float = float(config.get("stop_loss_pct", 999.0))
        self.max_hold_seconds: int = int(config.get("max_hold_seconds", 240))
        self.profit_target_pct: float = float(config.get("profit_target_pct", 999.0))

        # Sizing: fractional Kelly clamped to a [min, max] USDC band.
        #   - min_position_size_usdc: floor applied only when Kelly > 0, so
        #     trades are operationally meaningful on small balances.
        #   - max_position_size_usdc: hard ceiling on per-trade notional.
        #     The strategy enforces this explicitly (do NOT rely on
        #     RiskManager alone — its size check is share-denominated and
        #     the units are load-dependent; see tasks/todo.md).
        # Legacy `position_size_usdc` is read as a fallback for max so
        # older configs keep working.
        self.kelly_fraction: float = float(config.get("kelly_fraction", 0.15))
        _legacy_cap = float(config.get("position_size_usdc", 30.0))
        self.max_position_size_usdc: float = float(
            config.get("max_position_size_usdc", _legacy_cap)
        )
        self.min_position_size_usdc: float = float(
            config.get("min_position_size_usdc", 0.0)
        )
        # Kept as an alias (= max) so dashboards / telemetry that still
        # read the single `position_size_usdc` key show something useful.
        self.position_size_usdc: float = self.max_position_size_usdc

        # Rolling YES orderbook history for v5 microstructure features:
        # (wall_ts, bid_depth_top3, ask_depth_top3). 30s is enough for the
        # 10s imbalance window + 5s ratio-change lag with slack.
        self._ob_history: Deque[Tuple[float, float, float]] = collections.deque()
        self._ob_history_max_age: float = 30.0

        # Edge-stability gate — require the edge to clear the threshold on
        # this many consecutive ticks (within the same slot) before firing.
        # 1 = no debouncing (legacy behavior). See _edge_stability.py.
        self.min_stable_ticks: int = int(config.get("min_stable_ticks", 1))
        self._stability = EdgeStabilityTracker(n_ticks=self.min_stable_ticks)

        # Late-slot snipe — opt-in microstructure path. Bypasses the normal
        # min_seconds_to_expiry / min_entry_price / max_entry_price gates;
        # has its own (different-shape) gates instead. Does NOT consult the
        # model — thesis is microstructure mispricing harvest, not prediction.
        snipe_cfg = dict(config.get("late_snipe") or {})
        self.late_snipe_enabled: bool = bool(snipe_cfg.get("enabled", False))
        self.late_snipe_max_tte_s: float = float(snipe_cfg.get("max_tte_s", 30.0))
        self.late_snipe_min_entry_price: float = float(snipe_cfg.get("min_entry_price", 0.90))
        self.late_snipe_max_entry_price: float = float(snipe_cfg.get("max_entry_price", 0.95))
        self.late_snipe_min_moneyness_bps: float = float(snipe_cfg.get("min_moneyness_bps", 50.0))
        self.late_snipe_max_spread_pct: float = float(snipe_cfg.get("max_spread_pct", 0.03))
        self.late_snipe_size_usdc: float = float(snipe_cfg.get("size_usdc", 5.0))
        # Once-per-slot guard, keyed by slot_ts (slot_expiry_ts − 300). Slot
        # rollover changes slot_ts → comparison naturally lets the next slot
        # snipe again without an explicit reset hook.
        self._snipe_fired_for_slot: Optional[int] = None

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
        # Polymarket-derived features (computed per-cycle, logged to JSONL)
        self.last_microprice_yes: Optional[float] = None
        self.last_micro_delta_yes: Optional[float] = None
        self.last_f_ret_30s: Optional[float] = None
        self.last_f_delta: Optional[float] = None

        # Late-slot snipe — opt-in microstructure path. Bypasses the normal
        # min_seconds_to_expiry / min_entry_price / max_entry_price gates;
        # has its own (different-shape) gates instead. Does NOT consult the
        # model — thesis is microstructure mispricing harvest, not prediction.
        snipe_cfg = dict(config.get("late_snipe") or {})
        self.late_snipe_enabled: bool = bool(snipe_cfg.get("enabled", False))
        self.late_snipe_max_tte_s: float = float(snipe_cfg.get("max_tte_s", 30.0))
        self.late_snipe_min_entry_price: float = float(snipe_cfg.get("min_entry_price", 0.90))
        self.late_snipe_max_entry_price: float = float(snipe_cfg.get("max_entry_price", 0.95))
        self.late_snipe_min_moneyness_bps: float = float(snipe_cfg.get("min_moneyness_bps", 50.0))
        self.late_snipe_max_spread_pct: float = float(snipe_cfg.get("max_spread_pct", 0.03))
        self.late_snipe_size_usdc: float = float(snipe_cfg.get("size_usdc", 5.0))
        # Once-per-slot guard, keyed by slot_ts (slot_expiry_ts − 300). Slot
        # rollover changes slot_ts → comparison naturally lets the next slot
        # snipe again without an explicit reset hook.
        self._snipe_fired_for_slot: Optional[int] = None

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

        # Record YES orderbook depth history for v5 microstructure features.
        # NOTE: this runs on every cycle, including while we're holding an
        # open position (we short-circuit with "in_position" below). That's
        # intentional — the 30s rolling window needs to be warm when the
        # next slot opens and we start evaluating entries. Do NOT move this
        # under the "not in position" branch.
        yes_book_for_ob = order_books.get(self._yes_token_id)
        if yes_book_for_ob and yes_book_for_ob.bids and yes_book_for_ob.asks:
            bid_d = sum(float(e.size) for e in yes_book_for_ob.bids[:3])
            ask_d = sum(float(e.size) for e in yes_book_for_ob.asks[:3])
            self._ob_history.append((now_wall, bid_d, ask_d))
            cutoff = now_wall - self._ob_history_max_age
            while self._ob_history and self._ob_history[0][0] < cutoff:
                self._ob_history.popleft()

        self._auto_recover_position(by_token, now_mono)

        # If already in a position, hold to expiry (no exit logic)
        if self.active_token_id is not None:
            self.last_skip_reason = "in_position"
            return []

        signals = self._check_entry(order_books, by_token, market_data, now_wall, now_mono, balance)
        if signals or not self.late_snipe_enabled:
            return signals
        return self._check_late_snipe(order_books, by_token, market_data, now_wall, now_mono)

    def should_enter(self, signal: Signal) -> bool:
        # The min_confidence gate is enforced earlier in _check_entry against
        # the model's raw probability (see "Confidence gate" there). By the
        # time we're here, the Signal's confidence is clamped up to at least
        # self.min_confidence, so a post-hoc check would be a no-op. Relying
        # on the _check_entry gate + validate_signal's structural checks.
        return self.validate_signal(signal)

    # ------------------------------------------------------------------
    # Entry logic: p_hat vs q_t +/- c_t +/- delta
    # ------------------------------------------------------------------

    def _check_entry(
        self,
        order_books: Dict[str, OrderBook],
        by_token: Dict[str, Position],
        market_data: Dict[str, Any],
        now_wall: float,
        now_mono: float,
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

        # ── Polymarket-derived features ──────────────────────────────────
        # Computed every entry-check cycle and stored as observables so
        # _log_decision can pick them up for the JSONL decision log.

        # Microprice: size-weighted fair value. When the bid side is thick
        # relative to ask, microprice pulls toward the ask (and vice versa),
        # signaling short-term directional pressure.
        yes_bid_size = float(yes_book.bids[0].size) if yes_book.bids else 0.0
        yes_ask_size = float(yes_book.asks[0].size) if yes_book.asks else 0.0
        total_size = yes_bid_size + yes_ask_size
        if total_size > 0:
            microprice_yes = (yes_bid * yes_ask_size + yes_ask * yes_bid_size) / total_size
        else:
            microprice_yes = q_t  # degenerate: fall back to mid
        self.last_microprice_yes = microprice_yes
        self.last_micro_delta_yes = microprice_yes - q_t

        # f_ret_30s: 30-second return on YES market mid.
        # Uses the strategy's rolling price history (base.py record_price).
        # Returns None when < 15s of history exists (avoids leaking stale data).
        q_t_30s_ago = self.get_price_ago(self._yes_token_id, 30.0, now=now_mono)
        if q_t_30s_ago is not None and q_t_30s_ago > 0:
            self.last_f_ret_30s = (q_t - q_t_30s_ago) / q_t_30s_ago
        else:
            self.last_f_ret_30s = None

        # f_delta: (btc_spot - strike) / strike. Canonical BTC delta used
        # by the v2 model (logreg_model.py:116). Recomputed here from the
        # live btc_feed so it's always fresh and available for logging even
        # when the active model doesn't use it as a feature.
        strike_price = market_data.get("strike_price")
        btc_prices = []
        if self.btc_feed is not None:
            btc_prices = getattr(self.btc_feed, "get_recent_prices", lambda w=10: [])(10)
        if btc_prices and strike_price and float(strike_price) > 0:
            btc_mid = float(btc_prices[-1][1])
            self.last_f_delta = (btc_mid - float(strike_price)) / float(strike_price)
        else:
            self.last_f_delta = None

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

        # Tighten delta as expiry approaches: market is better calibrated and
        # execution risk is higher near close, so demand more edge.
        effective_delta = self.delta * (1.0 + 0.5 * max(0.0, 1.0 - tte / 120.0))

        # Edge-stability gate: track whether each side has cleared the
        # threshold on enough consecutive ticks. Keyed by slot_expiry_ts so
        # the counters reset automatically at rollover. With min_stable_ticks=1
        # both flags are ready on the first above-threshold tick (legacy behavior).
        slot_expiry_ts = market_data.get("slot_expiry_ts")
        stab = self._stability.update(
            yes_above=edge_yes >= effective_delta,
            no_above=edge_no >= effective_delta,
            slot_ts=float(slot_expiry_ts) if slot_expiry_ts is not None else None,
        )

        # Entry decision: pick the side with higher edge, must exceed delta
        # AND have been above threshold for min_stable_ticks in this slot.
        if edge_yes >= edge_no and edge_yes >= effective_delta and stab.yes_ready:
            side = "YES"
            edge = edge_yes
            entry_ask = yes_ask
            book = yes_book
            prob = p_hat
        elif edge_no > edge_yes and edge_no >= effective_delta and stab.no_ready:
            side = "NO"
            edge = edge_no
            entry_ask = no_ask
            book = no_book
            prob = 1.0 - p_hat
        elif (edge_yes >= effective_delta and not stab.yes_ready) or (
            edge_no >= effective_delta and not stab.no_ready
        ):
            # Edge cleared threshold but hasn't held long enough yet — wait.
            best_count = max(stab.yes_count, stab.no_count)
            self.last_skip_reason = (
                f"edge_not_stable: {best_count}/{self.min_stable_ticks} consecutive ticks"
            )
            return []
        else:
            best = max(edge_yes, edge_no)
            self.last_skip_reason = f"edge_low: best={best:+.3f} < delta={effective_delta:.3f} (tte={tte:.0f}s)"
            return []

        # Confidence gate: reject if model's probability on the chosen side
        # is below the configured floor. This must run BEFORE building the
        # Signal because Signal.confidence gets clamped up to min_confidence
        # for display/logging — that clamp would otherwise let low-conviction
        # predictions slip past validate_signal's check.
        if prob < self.min_confidence:
            self.last_skip_reason = (
                f"confidence_low: side={side} prob={prob:.3f} < min_confidence={self.min_confidence:.3f}"
            )
            return []

        # Model-side probability floors. If the loaded model_service exposes a
        # `thresholds` dict with `min_prob_yes` / `max_prob_yes_for_no`,
        # enforce them as additional gates on top of `delta`. Catches cases
        # where one side of the calibrator is unreliable in a region the
        # global delta gate doesn't cover. See tasks/postmortem_2026-04-25.md.
        model_thresholds = getattr(self.model_service, "thresholds", None)
        if not isinstance(model_thresholds, dict):
            model_thresholds = {}
        min_prob_yes = float(model_thresholds.get("min_prob_yes", 0.0))
        max_prob_yes_for_no = float(model_thresholds.get("max_prob_yes_for_no", 1.0))
        if side == "YES" and p_hat < min_prob_yes:
            self.last_skip_reason = (
                f"model_prob_yes_low: p_hat={p_hat:.3f} < min_prob_yes={min_prob_yes:.3f}"
            )
            return []
        if side == "NO" and p_hat > max_prob_yes_for_no:
            self.last_skip_reason = (
                f"model_prob_yes_high_for_no: p_hat={p_hat:.3f} > max_prob_yes_for_no={max_prob_yes_for_no:.3f}"
            )
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
    # Late-slot snipe — microstructure path, model-agnostic
    # ------------------------------------------------------------------

    def _check_late_snipe(
        self,
        order_books: Dict[str, OrderBook],
        by_token: Dict[str, Position],
        market_data: Dict[str, Any],
        now_wall: float,
        now_mono: float,
    ) -> List[Signal]:
        """Late-slot, near-resolution sniping. See class docstring + the
        ``late_snipe`` config block for the thesis. Fires at most once per
        slot; gated by tte, market price, |moneyness|, and spread."""
        slot_expiry_ts = float(market_data.get("slot_expiry_ts") or 0.0)
        tte = max(0.0, slot_expiry_ts - now_wall)
        if tte > self.late_snipe_max_tte_s:
            self.last_skip_reason = (
                f"snipe_tte: {tte:.0f}s > {self.late_snipe_max_tte_s:.0f}s"
            )
            return []

        slot_ts = int(slot_expiry_ts) - 300
        if self._snipe_fired_for_slot == slot_ts:
            self.last_skip_reason = "snipe_already_fired_this_slot"
            return []

        if not self.is_flat(by_token):
            self.last_skip_reason = "snipe_in_position"
            return []

        # Need strike + BTC mid to evaluate moneyness.
        strike = float(market_data.get("strike_price") or 0.0)
        if strike <= 0:
            self.last_skip_reason = "snipe_missing_strike"
            return []
        # Prefer caller-provided btc_mid (lets tests inject a value); fall
        # back to the live feed when the bot is wired up to one.
        btc_mid = market_data.get("btc_mid")
        if btc_mid is None:
            btc_mid = self._latest_btc_mid()
        if btc_mid is None or float(btc_mid) <= 0:
            self.last_skip_reason = "snipe_missing_btc"
            return []
        btc_mid = float(btc_mid)
        moneyness_bps = (btc_mid - strike) / strike * 10_000.0

        yes_book = order_books.get(self._yes_token_id)
        no_book = order_books.get(self._no_token_id)
        if yes_book is None or no_book is None:
            self.last_skip_reason = "snipe_missing_orderbook"
            return []

        yes_bid = float(yes_book.bids[0].price) if yes_book.bids else 0.0
        yes_ask = float(yes_book.asks[0].price) if yes_book.asks else 0.0
        no_bid = float(no_book.bids[0].price) if no_book.bids else 0.0
        no_ask = float(no_book.asks[0].price) if no_book.asks else 0.0

        # Pick the side. YES snipe needs BTC above strike (positive moneyness)
        # AND the YES ask in the snipe band. NO snipe is the mirror.
        side: Optional[str] = None
        entry_ask = 0.0
        book: Optional[OrderBook] = None
        if (
            moneyness_bps >= self.late_snipe_min_moneyness_bps
            and self.late_snipe_min_entry_price <= yes_ask <= self.late_snipe_max_entry_price
        ):
            side = "YES"
            entry_ask = yes_ask
            book = yes_book
        elif (
            moneyness_bps <= -self.late_snipe_min_moneyness_bps
            and self.late_snipe_min_entry_price <= no_ask <= self.late_snipe_max_entry_price
        ):
            side = "NO"
            entry_ask = no_ask
            book = no_book
        else:
            self.last_skip_reason = (
                f"snipe_no_match: moneyness={moneyness_bps:+.0f}bps "
                f"yes_ask={yes_ask:.3f} no_ask={no_ask:.3f}"
            )
            return []

        # Spread gate: tighter than normal entry, since there's no time to
        # wait for spreads to come in.
        side_bid = yes_bid if side == "YES" else no_bid
        sp_pct = spread_pct(side_bid, entry_ask)
        if sp_pct > self.late_snipe_max_spread_pct:
            self.last_skip_reason = (
                f"snipe_spread_wide: {side} {sp_pct*100:.1f}% > "
                f"{self.late_snipe_max_spread_pct*100:.1f}%"
            )
            return []

        # Size: small fixed USDC notional, converted to shares. Bypasses
        # Kelly entirely — Kelly on near-certain trades is huge, and we
        # don't want to dominate the position book if conditions are unusual.
        shares = round(self.late_snipe_size_usdc / entry_ask, 2)
        if shares <= 0:
            self.last_skip_reason = "snipe_size_zero"
            return []

        # Mark the slot as sniped so we don't double-fire within the same
        # intra-cycle pass. Reset is implicit: when slot_ts changes on
        # rollover, the equality check above lets the next slot fire again.
        # We do NOT optimistically set active_token_id/entry_*: that pattern
        # caused phantom positions when orders were rejected downstream.
        # _auto_recover_position picks up the real position from inventory
        # after the fill lands.
        self._snipe_fired_for_slot = slot_ts

        tick = book.tick_size or 0.001
        return [Signal(
            market_id=self._market_id,
            outcome=side,
            action="BUY",
            confidence=entry_ask,  # market-implied probability of the chosen side
            price=round_to_tick(entry_ask, tick),
            size=shares,
            reason=(
                f"late_snipe {side} ask={entry_ask:.3f} "
                f"moneyness={moneyness_bps:+.0f}bps tte={tte:.0f}s "
                f"spread={sp_pct*100:.1f}% size=${self.late_snipe_size_usdc:.2f}"
            ),
        )]

    def _latest_btc_mid(self) -> Optional[float]:
        """Return the freshest BTC mid from the strategy's BtcPriceFeed."""
        if self.btc_feed is None:
            return None
        try:
            return self.btc_feed.get_latest_mid()
        except AttributeError:
            return None

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
            # 4 hours — long enough for btc_ret_3600s and multi-TF RSI warmup
            # on 1m/3m/5m/15m. Must match training's BTC window
            # (_BTC_WINDOW_SECONDS in src/backtest/s3_snapshot_loader.py).
            btc_prices = getattr(
                self.btc_feed, "get_recent_prices", lambda w=14400: []
            )(14400)
            # Guard against trading on stale BTC data (feed dead >60s)
            if btc_prices and (now_wall - btc_prices[-1][0]) > 60:
                self.logger.warning("btc_feed stale (>60s), skipping prediction")
                self.last_feature_status = "stale_btc"
                return None

        # Convert monotonic yes_history to wall-clock so all snapshot
        # timestamps share the same domain (btc_prices, slot_expiry_ts use wall clock).
        mono_to_wall = now_wall - time.monotonic()
        yes_history = [(ts + mono_to_wall, p)
                       for ts, p in (self._price_history.get(self._yes_token_id) or [])]

        snapshot = {
            "btc_prices": btc_prices,
            "yes_book": order_books.get(self._yes_token_id),
            "no_book": order_books.get(self._no_token_id),
            "yes_history": yes_history,
            "yes_ob_history": list(self._ob_history),
            "question": market_data.get("question", ""),
            "strike_price": market_data.get("strike_price"),
            "slot_expiry_ts": market_data.get("slot_expiry_ts"),
            "now_ts": now_wall,
            # Forward the recent-outcomes list cycle_runner injects into
            # market_data so the signed-v2 recent_up_rate_* features see
            # real outcome history instead of the 0.5 prior.
            "recent_slot_outcomes": market_data.get("recent_slot_outcomes"),
        }

        try:
            prediction = self.model_service.predict(snapshot)
        except Exception as exc:
            self.logger.warning("logreg_edge: predict failed: %s", exc)
            self.last_feature_status = "predict_error"
            return None

        self.last_feature_status = prediction.feature_status
        self.last_model_version = prediction.model_version

        if prediction.prob_yes is None:
            self.logger.debug(
                "logreg_edge: model returned no prediction (status=%s, btc_prices=%d)",
                prediction.feature_status, len(btc_prices),
            )

        return prediction

    # ------------------------------------------------------------------
    # Hot-reload for online retraining
    # ------------------------------------------------------------------

    def reload_model(self, new_dir: str) -> bool:
        """Swap self.model_service for a freshly-loaded model from disk.

        Called by CycleRunner between cycles when Retrainer has a new
        candidate ready AND the strategy is flat. Only swaps if the new
        model loads cleanly (ready=True); on failure the old model
        stays in place and we log the skip.
        """
        # Import here to avoid a cycle on module load (LogRegV4Model
        # transitively imports xgb_model → PredictionResult).
        from ..models.logreg_v4_model import LogRegV4Model
        try:
            new_model = LogRegV4Model.load(new_dir, logger=self.logger)
        except Exception as exc:
            self.logger.warning("reload_model: load failed for %s: %s", new_dir, exc)
            return False
        if not getattr(new_model, "ready", False):
            self.logger.warning("reload_model: new model at %s not ready — keeping old", new_dir)
            return False
        old_version = getattr(self.model_service, "model_version", "unknown")
        self.model_service = new_model
        self.logger.info(
            "logreg model hot-reloaded: %s → %s (from %s)",
            old_version, new_model.model_version, new_dir,
        )
        return True

    # ------------------------------------------------------------------
    # Kelly sizing (same formula as ProbEdgeStrategy)
    # ------------------------------------------------------------------

    def _kelly_size(self, prob: float, price: float, balance: float) -> float:
        """Fractional Kelly notional for a binary contract, clamped to the
        [min_position_size_usdc, max_position_size_usdc] band.

            f_kelly = (p - x) / (1 - x)
            f_used  = kelly_fraction * f_kelly
            usdc    = f_used * balance
            return  = clip(usdc, min, max)   if usdc > 0
                      0                       otherwise

        The floor is applied only when Kelly is positive — a zero-Kelly
        signal (no edge at the ask) returns 0 and no trade fires. By the
        time this runs, the entry gate (delta + min_confidence) has
        already confirmed meaningful edge, so floor-enforcement just
        bumps operationally-too-small raw Kelly outputs up to a placeable
        size. The max ceiling is enforced here (not trusted to the
        RiskManager) because RiskManager's size check is share-
        denominated and the units are load-dependent — see
        tasks/todo.md for the refactor.
        """
        if price <= 0 or price >= 1 or prob <= price:
            return 0.0
        if balance <= 0:
            # Unknown/zero balance: fall back to the floor (or the legacy
            # single-number cap for configs that don't set a floor).
            return self.min_position_size_usdc or self.max_position_size_usdc
        f_full = (prob - price) / (1.0 - price)
        f_used = max(0.0, self.kelly_fraction * f_full)
        usdc = f_used * balance
        if usdc <= 0:
            return 0.0
        usdc = min(usdc, self.max_position_size_usdc)
        if self.min_position_size_usdc > 0:
            usdc = max(usdc, self.min_position_size_usdc)
        return usdc
