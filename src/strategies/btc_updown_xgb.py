"""Probability-based BTC Up/Down strategy driven by an XGBoost model."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import Signal, Strategy
from ..api.types import OrderBook, Position
from ..models import BTCUpDownXGBModel
from ..utils.market_utils import get_mid_price, round_to_tick


class BTCUpDownXGBStrategy(Strategy):
    """Trade YES/NO when the model-implied edge exceeds market prices."""

    def __init__(
        self,
        config: Dict[str, Any],
        btc_feed,
        model_service: Optional[BTCUpDownXGBModel] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="btc_updown_xgb", config=config)

        self.logger = logger or logging.getLogger(__name__)
        self.btc_feed = btc_feed
        self.model_service = model_service or BTCUpDownXGBModel(
            model_path=config.get("model_path", "models/btc_updown_xgb.json"),
            feature_schema_path=config.get("feature_schema_path", "models/btc_updown_xgb_features.json"),
            metadata_path=config.get("metadata_path", "models/btc_updown_xgb_meta.json"),
            logger=self.logger,
        )

        thresholds = self.model_service.thresholds
        self.position_size_usdc: float = float(config.get("position_size_usdc", 20.0))
        self.stop_loss_pct: float = float(config.get("stop_loss_pct", 0.12))
        self.max_hold_seconds: int = int(config.get("max_hold_seconds", 240))
        self.min_edge: float = float(config.get("min_edge", thresholds.get("min_edge", 0.03)))
        self.min_prob_yes: float = float(config.get("min_prob_yes", thresholds.get("min_prob_yes", 0.54)))
        self.max_prob_yes_for_no: float = float(config.get("max_prob_yes_for_no", thresholds.get("max_prob_yes_for_no", 0.46)))
        self.max_spread_pct: float = float(config.get("max_spread_pct", thresholds.get("max_spread_pct", 0.06)))
        self.exit_edge: float = float(config.get("exit_edge", thresholds.get("exit_edge", -0.01)))
        self.min_seconds_to_expiry: float = float(config.get("min_seconds_to_expiry", thresholds.get("min_seconds_to_expiry", 20.0)))
        self.max_seconds_to_expiry: float = float(config.get("max_seconds_to_expiry", thresholds.get("max_seconds_to_expiry", 240.0)))

        self._price_history: Dict[str, List[Tuple[float, float]]] = {}
        self._history_max_age: float = 300.0

        self.last_prob_yes: Optional[float] = None
        self.last_edge_yes: Optional[float] = None
        self.last_edge_no: Optional[float] = None
        self.last_feature_status: str = ""
        self.last_model_version: str = self.model_service.model_version

    def set_tokens(self, market_id: str, yes_token_id: str, no_token_id: str) -> None:
        if (yes_token_id != self._yes_token_id or no_token_id != self._no_token_id) and self._yes_token_id:
            self._reset_position_state()
        self._market_id = market_id
        self._yes_token_id = yes_token_id
        self._no_token_id = no_token_id
        self._outcome_map = {yes_token_id: "YES", no_token_id: "NO"}

    def record_price(self, token_id: str, mid: float, ts: Optional[float] = None) -> None:
        now = ts if ts is not None else time.monotonic()
        buf = self._price_history.setdefault(token_id, [])
        buf.append((now, mid))
        cutoff = now - self._history_max_age
        self._price_history[token_id] = [(t, p) for t, p in buf if t >= cutoff]

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        if not self._yes_token_id or not self._no_token_id:
            return []

        order_books: Dict[str, OrderBook] = market_data.get("order_books", {})
        positions: List[Position] = market_data.get("positions", [])
        by_token: Dict[str, Position] = {p.token_id: p for p in positions}
        now_mono = time.monotonic()
        now_wall = time.time()

        for tid in (self._yes_token_id, self._no_token_id):
            book = order_books.get(tid)
            if book is None:
                continue
            mid = get_mid_price(book)
            if mid is None:
                continue
            self.record_price(tid, mid, now_mono)

        if self.active_token_id is None:
            for tid in (self._yes_token_id, self._no_token_id):
                pos = by_token.get(tid)
                if pos and pos.size > 0:
                    self.active_token_id = tid
                    self.entry_price = pos.average_price
                    self.entry_timestamp = now_mono
                    self.entry_size = pos.size
                    break

        prediction = self._predict(order_books, market_data, now_wall)

        exit_sig = self._check_exit(order_books, by_token, now_mono, prediction)
        if exit_sig:
            self._pending_exit_entry_price = self.entry_price
            self._pending_exit_entry_size = self.entry_size
            self._reset_position_state()
            return [exit_sig]

        if self.active_token_id is not None:
            return []

        return self._check_entry(order_books, by_token, market_data, now_mono, now_wall, prediction)

    def should_enter(self, signal: Signal) -> bool:
        return self.validate_signal(signal) and signal.confidence >= self.min_confidence

    def _predict(
        self,
        order_books: Dict[str, OrderBook],
        market_data: Dict[str, Any],
        now_wall: float,
    ):
        self.last_prob_yes = None
        self.last_edge_yes = None
        self.last_edge_no = None

        if self.btc_feed is None or not getattr(self.btc_feed, "is_healthy", lambda: False)():
            self.last_feature_status = "stale_btc_feed"
            return None

        yes_book = order_books.get(self._yes_token_id)
        no_book = order_books.get(self._no_token_id)
        if yes_book is None or no_book is None:
            self.last_feature_status = "missing_order_books"
            return None

        snapshot = {
            "btc_prices": getattr(self.btc_feed, "get_recent_prices", lambda window_s=300: [])(300),
            "yes_book": yes_book,
            "no_book": no_book,
            "yes_history": self._price_history.get(self._yes_token_id, []),
            "no_history": self._price_history.get(self._no_token_id, []),
            "question": market_data.get("question", ""),
            "strike_price": market_data.get("strike_price"),
            "slot_expiry_ts": market_data.get("slot_expiry_ts"),
            "now_ts": now_wall,
        }
        prediction = self.model_service.predict(snapshot)
        self.last_feature_status = prediction.feature_status
        self.last_model_version = prediction.model_version

        if prediction.prob_yes is None or prediction.prob_no is None:
            return None

        yes_ask = float(yes_book.asks[0].price) if yes_book.asks else None
        no_ask = float(no_book.asks[0].price) if no_book.asks else None
        if yes_ask and no_ask:
            self.last_prob_yes = prediction.prob_yes
            self.last_edge_yes = prediction.prob_yes - yes_ask
            self.last_edge_no = prediction.prob_no - no_ask

        return prediction

    def _check_entry(
        self,
        order_books: Dict[str, OrderBook],
        by_token: Dict[str, Position],
        market_data: Dict[str, Any],
        now_mono: float,
        now_wall: float,
        prediction,
    ) -> List[Signal]:
        if prediction is None or prediction.prob_yes is None or prediction.prob_no is None:
            return []

        slot_expiry_ts = float(market_data.get("slot_expiry_ts") or 0.0)
        seconds_to_expiry = max(0.0, slot_expiry_ts - now_wall)
        if seconds_to_expiry < self.min_seconds_to_expiry or seconds_to_expiry > self.max_seconds_to_expiry:
            return []

        yes_book = order_books.get(self._yes_token_id)
        no_book = order_books.get(self._no_token_id)
        if yes_book is None or no_book is None:
            return []

        yes_bid = float(yes_book.bids[0].price) if yes_book.bids else 0.0
        yes_ask = float(yes_book.asks[0].price) if yes_book.asks else 0.0
        no_bid = float(no_book.bids[0].price) if no_book.bids else 0.0
        no_ask = float(no_book.asks[0].price) if no_book.asks else 0.0

        yes_spread_pct = _spread_pct(yes_bid, yes_ask)
        no_spread_pct = _spread_pct(no_bid, no_ask)
        if yes_spread_pct > self.max_spread_pct and no_spread_pct > self.max_spread_pct:
            self.last_feature_status = "spread_too_wide"
            return []

        edge_yes = prediction.prob_yes - yes_ask if yes_ask > 0 else -1.0
        edge_no = prediction.prob_no - no_ask if no_ask > 0 else -1.0
        self.last_prob_yes = prediction.prob_yes
        self.last_edge_yes = edge_yes
        self.last_edge_no = edge_no

        candidates: List[Tuple[str, OrderBook, float, float, str]] = []
        if yes_ask > 0 and self.min_entry_price <= yes_ask <= self.max_entry_price and yes_spread_pct <= self.max_spread_pct and edge_yes >= self.min_edge and prediction.prob_yes >= self.min_prob_yes:
            candidates.append(("YES", yes_book, yes_ask, edge_yes, f"prob_yes={prediction.prob_yes:.3f} edge_yes={edge_yes:.3f}"))
        if no_ask > 0 and self.min_entry_price <= no_ask <= self.max_entry_price and no_spread_pct <= self.max_spread_pct and edge_no >= self.min_edge and prediction.prob_yes <= self.max_prob_yes_for_no:
            candidates.append(("NO", no_book, no_ask, edge_no, f"prob_no={prediction.prob_no:.3f} edge_no={edge_no:.3f}"))
        if not candidates:
            return []

        outcome, book, best_ask, edge, reason = max(candidates, key=lambda item: item[3])
        token_id = self._yes_token_id if outcome == "YES" else self._no_token_id
        if not self.is_flat(by_token):
            return []

        tick = book.tick_size or 0.001
        size = round(self.position_size_usdc / best_ask, 2)
        confidence = max(0.5, min(1.0, max(prediction.prob_yes, prediction.prob_no)))

        self.active_token_id = token_id
        self.entry_price = best_ask
        self.entry_timestamp = now_mono
        self.entry_size = size

        return [
            Signal(
                market_id=self._market_id,
                outcome=outcome,
                action="BUY",
                confidence=confidence,
                price=round_to_tick(best_ask, tick),
                size=size,
                reason=f"xgb_entry {reason} status={prediction.feature_status}",
            )
        ]

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
        held_size = float(pos.size) if pos and pos.size > 0 else float(self.entry_size or 0.0)
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

        pnl_pct = (best_bid - self.entry_price) / self.entry_price if self.entry_price else 0.0
        time_held = now_mono - self.entry_timestamp if self.entry_timestamp else 0.0
        outcome = self._outcome_map.get(self.active_token_id, "")

        reason = None
        if -pnl_pct >= self.stop_loss_pct:
            reason = f"stop_loss loss={-pnl_pct:.2%} >= limit={self.stop_loss_pct:.2%}"
        elif time_held >= self.max_hold_seconds:
            reason = f"time_limit held={time_held:.0f}s >= max={self.max_hold_seconds}s"
        elif prediction is not None and prediction.prob_yes is not None and prediction.prob_no is not None:
            held_edge = prediction.prob_yes - best_bid if outcome == "YES" else prediction.prob_no - best_bid
            if outcome == "YES":
                self.last_edge_yes = held_edge
            else:
                self.last_edge_no = held_edge

            if held_edge <= self.exit_edge:
                reason = f"edge_reprice held_edge={held_edge:.3f} <= exit_edge={self.exit_edge:.3f}"
            elif outcome == "YES" and prediction.prob_yes <= self.max_prob_yes_for_no:
                reason = f"model_flip prob_yes={prediction.prob_yes:.3f} <= {self.max_prob_yes_for_no:.3f}"
            elif outcome == "NO" and prediction.prob_yes >= self.min_prob_yes:
                reason = f"model_flip prob_yes={prediction.prob_yes:.3f} >= {self.min_prob_yes:.3f}"

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


def _spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return math.inf
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return math.inf
    return max(0.0, ask - bid) / mid
