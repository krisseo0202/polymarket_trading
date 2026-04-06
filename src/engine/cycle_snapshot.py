"""Single source of truth for the current trading cycle state.

Written atomically by bot.py at the end of every cycle.
Read by dashboard.py to display consistent state without extra API calls.

Assembly:
    snap = build_cycle_snapshot(market_id, ..., yes_book, no_book, ...)
    market_data = snap.to_market_data(yes_book, no_book, positions, balance)
    signals = strategy.analyze(market_data)
    snap.update_from_strategy(strategy)
    snapshot_store.save(snap)
"""

import dataclasses
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SLOT_INTERVAL_S = 300


class BotStatus(str, Enum):
    INIT = "INIT"
    EVALUATING = "EVALUATING"
    IN_POSITION = "IN_POSITION"
    COOLDOWN = "COOLDOWN"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class CycleSnapshot:
    """Snapshot of all state the bot held at the end of the current cycle.

    Build with build_cycle_snapshot(), not manually.
    Strategy reads market_data via to_market_data(); dashboard reads this object.
    Both get their numbers from the same source.
    """

    # ── Slot info ──────────────────────────────────────────────────────────────
    market_id: str = ""
    question: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    slot_ts: Optional[int] = None       # floor(unix / 300) * 300
    slot_end_ts: Optional[int] = None   # slot_ts + 300
    tte_seconds: Optional[float] = None

    # ── Reference price ───────────────────────────────────────────────────────
    price_to_beat: Optional[float] = None   # Chainlink slot-open (was: strike)

    # ── BTC current ───────────────────────────────────────────────────────────
    btc_now: Optional[float] = None

    # ── YES order book (top-of-book) ──────────────────────────────────────────
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    yes_mid: Optional[float] = None

    # ── NO order book (top-of-book) ───────────────────────────────────────────
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    no_mid: Optional[float] = None

    # ── Order book depth (top 5 levels) — for dashboard display ──────────────
    # Each entry: {"price": float, "size": float}
    yes_bids: List[Dict[str, float]] = field(default_factory=list)
    yes_asks: List[Dict[str, float]] = field(default_factory=list)
    no_bids: List[Dict[str, float]] = field(default_factory=list)
    no_asks: List[Dict[str, float]] = field(default_factory=list)

    # ── Positions: token_id → {"size": float, "avg_cost": float} ─────────────
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── Active orders (serialized Order dicts) ────────────────────────────────
    active_orders: List[Dict[str, Any]] = field(default_factory=list)

    # ── Bot operational status ────────────────────────────────────────────────
    bot_status: str = BotStatus.INIT.value
    held_side: str = "FLAT"    # "YES" | "NO" | "FLAT"

    # ── PnL ───────────────────────────────────────────────────────────────────
    daily_realized_pnl: float = 0.0
    slot_realized_pnl: float = 0.0
    unrealized_pnl: Optional[float] = None

    # ── Strategy telemetry (populated by update_from_strategy) ────────────────
    strategy_name: str = ""
    strategy_status: str = ""
    strategy_prob_yes: Optional[float] = None
    strategy_prob_no: Optional[float] = None
    strategy_edge_yes: Optional[float] = None
    strategy_edge_no: Optional[float] = None
    strategy_net_edge_yes: Optional[float] = None
    strategy_net_edge_no: Optional[float] = None
    strategy_expected_fill_yes: Optional[float] = None
    strategy_expected_fill_no: Optional[float] = None
    strategy_required_edge: Optional[float] = None
    strategy_tte_seconds: Optional[float] = None
    strategy_distance_to_break_pct: Optional[float] = None
    strategy_distance_to_strike_bps: Optional[float] = None
    strategy_model_version: str = ""
    strategy_feature_status: str = ""
    strategy_score_breakdown: Optional[dict] = None
    strategy_skip_reason: str = ""

    # ── Metadata ──────────────────────────────────────────────────────────────
    cycle_count: int = 0
    paper_trading: bool = True
    updated_at: float = field(default_factory=time.time)

    # ── Legacy aliases (kept so dashboards reading old JSON keys still work) ───
    # These are populated by to_dict() via _LEGACY_ALIASES below.

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Write legacy keys so old dashboard reads (snapshot.get("strike") etc.) keep working.
        d.setdefault("strike", d.get("price_to_beat"))
        d.setdefault("start_ts", d.get("slot_ts"))
        d.setdefault("end_ts", d.get("slot_end_ts"))
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleSnapshot":
        # Migrate old field names transparently.
        migrated = dict(data)
        if "slot_ts" not in migrated and "start_ts" in migrated:
            migrated["slot_ts"] = migrated["start_ts"]
        if "slot_end_ts" not in migrated and "end_ts" in migrated:
            migrated["slot_end_ts"] = migrated["end_ts"]
        if "price_to_beat" not in migrated and "strike" in migrated:
            migrated["price_to_beat"] = migrated["strike"]
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in migrated.items() if k in known})

    def to_market_data(
        self,
        yes_book,
        no_book,
        positions,
        balance: float,
    ) -> Dict[str, Any]:
        """Produce the market_data dict that strategy.analyze() expects.

        The full OrderBook objects are passed because the strategy needs
        depth for VWAP estimation — scalars are not enough.
        Everything else (strike, slot expiry, token IDs, question) comes
        from the snapshot so there is exactly one source.
        """
        return {
            "markets": [],
            "order_books": {
                self.yes_token_id: yes_book,
                self.no_token_id: no_book,
            },
            "positions": positions,
            "balance": balance,
            "price_history": {},
            "question": self.question,
            "strike_price": self.price_to_beat,
            "slot_expiry_ts": self.slot_end_ts,
        }

    def update_from_strategy(self, strategy) -> None:
        """Copy last_* telemetry from a strategy into this snapshot.

        Call this after strategy.analyze() so the saved snapshot reflects
        what the strategy saw and decided this cycle.
        """
        self.strategy_name = getattr(strategy, "name", "")
        active_tid = getattr(strategy, "active_token_id", None)
        self.strategy_status = "POSITION_OPEN" if active_tid else "WATCHING"
        self.strategy_prob_yes = getattr(strategy, "last_prob_yes", None)
        self.strategy_prob_no = getattr(strategy, "last_prob_no", None)
        self.strategy_edge_yes = getattr(strategy, "last_edge_yes", None)
        self.strategy_edge_no = getattr(strategy, "last_edge_no", None)
        self.strategy_net_edge_yes = getattr(strategy, "last_net_edge_yes", None)
        self.strategy_net_edge_no = getattr(strategy, "last_net_edge_no", None)
        self.strategy_expected_fill_yes = getattr(strategy, "last_expected_fill_yes", None)
        self.strategy_expected_fill_no = getattr(strategy, "last_expected_fill_no", None)
        self.strategy_required_edge = getattr(strategy, "last_required_edge", None)
        self.strategy_tte_seconds = getattr(strategy, "last_tte_seconds", None)
        self.strategy_distance_to_break_pct = getattr(strategy, "last_distance_to_break_pct", None)
        self.strategy_distance_to_strike_bps = getattr(strategy, "last_distance_to_strike_bps", None)
        self.strategy_model_version = getattr(strategy, "last_model_version", "")
        self.strategy_feature_status = getattr(strategy, "last_feature_status", "")
        self.strategy_score_breakdown = getattr(strategy, "last_score_breakdown", None)
        self.strategy_skip_reason = getattr(strategy, "last_skip_reason", "")


def build_cycle_snapshot(
    *,
    market_id: str,
    question: str,
    yes_token_id: str,
    no_token_id: str,
    slot_ctx,                   # SlotContext | None
    btc_now: Optional[float],
    yes_book,                   # OrderBook | None
    no_book,                    # OrderBook | None
    inventories: Dict[str, Any],
    execution_tracker,
    risk_manager,
    state,                      # BotState
    paper_trading: bool,
    now: Optional[float] = None,
) -> CycleSnapshot:
    """Build a complete CycleSnapshot from the raw inputs of a single cycle.

    This is the single assembly point — call it once per cycle, then derive
    market_data via snap.to_market_data() and save via SnapshotStore.save().

    Strategy telemetry is NOT included here because the strategy hasn't run
    yet.  Call snap.update_from_strategy(strategy) after analyze().
    """
    now = now if now is not None else time.time()

    # ── Slot timestamps ────────────────────────────────────────────────────────
    if slot_ctx is not None:
        slot_ts = slot_ctx.slot_start_ts
        slot_end_ts = slot_ctx.slot_end_ts
        price_to_beat = slot_ctx.strike_price
    else:
        slot_ts = int(math.floor(now / SLOT_INTERVAL_S) * SLOT_INTERVAL_S)
        slot_end_ts = slot_ts + SLOT_INTERVAL_S
        price_to_beat = None

    tte = max(0.0, slot_end_ts - now)

    # ── Order book top-of-book ─────────────────────────────────────────────────
    yes_bid = float(yes_book.bids[0].price) if yes_book and yes_book.bids else None
    yes_ask = float(yes_book.asks[0].price) if yes_book and yes_book.asks else None
    no_bid  = float(no_book.bids[0].price)  if no_book  and no_book.bids  else None
    no_ask  = float(no_book.asks[0].price)  if no_book  and no_book.asks  else None
    yes_mid = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else (yes_bid or yes_ask)
    no_mid  = (no_bid  + no_ask)  / 2 if no_bid  and no_ask  else (no_bid  or no_ask)

    # ── Order book depth (top 5 levels) ───────────────────────────────────────
    _OB_DEPTH = 5

    def _levels(book, side: str):
        levels = getattr(book, side, []) if book else []
        return [
            {"price": float(lvl.price), "size": float(lvl.size)}
            for lvl in levels[:_OB_DEPTH]
        ]

    yes_bids_depth = _levels(yes_book, "bids")
    yes_asks_depth = _levels(yes_book, "asks")
    no_bids_depth  = _levels(no_book,  "bids")
    no_asks_depth  = _levels(no_book,  "asks")

    # ── Positions ──────────────────────────────────────────────────────────────
    pos_dict: Dict[str, Dict[str, float]] = {
        tid: {"size": inv.position, "avg_cost": inv.avg_cost}
        for tid, inv in inventories.items()
        if inv.position != 0
    }

    # ── Resolved slot outcome (written at rollover) ───────────────────────────
    slot_outcome: Optional[str] = None   # "Up" | "Down" | None (current slot unresolved)

    # ── Active orders ──────────────────────────────────────────────────────────
    orders = [
        {
            "order_id": o.order_id,
            "token_id": o.token_id,
            "outcome": o.outcome,
            "side": o.side,
            "price": o.price,
            "size": o.size,
            "status": o.status,
            "filled_qty": o.filled_qty,
        }
        for o in execution_tracker.active_orders.values()
    ]

    # ── Bot status ─────────────────────────────────────────────────────────────
    has_position = any(inv.position != 0 for inv in inventories.values())
    cb_active = getattr(risk_manager, "circuit_breaker_active", False)
    if cb_active:
        bot_status = BotStatus.COOLDOWN.value
    elif has_position:
        bot_status = BotStatus.IN_POSITION.value
    else:
        bot_status = BotStatus.EVALUATING.value

    # ── Held side (YES / NO / FLAT) ────────────────────────────────────────────
    yes_inv = inventories.get(yes_token_id)
    no_inv = inventories.get(no_token_id)
    if yes_inv and yes_inv.position > 0:
        held_side = "YES"
    elif no_inv and no_inv.position > 0:
        held_side = "NO"
    else:
        held_side = "FLAT"

    # ── Unrealized PnL ────────────────────────────────────────────────────────
    from ..utils.market_utils import get_mid_price  # local import to avoid circular dep
    book_map = {yes_token_id: yes_book, no_token_id: no_book}
    unrealized: Optional[float] = None
    if has_position:
        total = 0.0
        complete = True
        for tid, inv in inventories.items():
            if inv.position == 0:
                continue
            book = book_map.get(tid)
            mid = get_mid_price(book) if book else None
            if mid is None:
                complete = False
                break
            total += (mid - inv.avg_cost) * inv.position
        if complete:
            unrealized = total

    return CycleSnapshot(
        market_id=market_id,
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        slot_ts=slot_ts,
        slot_end_ts=slot_end_ts,
        tte_seconds=tte,
        price_to_beat=price_to_beat,
        btc_now=btc_now,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mid=yes_mid,
        no_bid=no_bid,
        no_ask=no_ask,
        no_mid=no_mid,
        yes_bids=yes_bids_depth,
        yes_asks=yes_asks_depth,
        no_bids=no_bids_depth,
        no_asks=no_asks_depth,
        positions=pos_dict,
        active_orders=orders,
        bot_status=bot_status,
        held_side=held_side,
        daily_realized_pnl=state.daily_realized_pnl,
        slot_realized_pnl=state.slot_realized_pnl,
        unrealized_pnl=unrealized,
        cycle_count=state.cycle_count,
        paper_trading=paper_trading,
    )


class SnapshotStore:
    """Atomic JSON writer/reader for CycleSnapshot.

    Uses write-to-tmp-then-rename so a mid-write crash never leaves a
    corrupt snapshot file.  fsync is intentionally omitted — the atomic
    rename already guarantees a consistent read; fsync would add 1-50ms
    of latency on every cycle for a monitoring file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._tmp = path + ".tmp"

    def save(self, snap: CycleSnapshot) -> None:
        snap.updated_at = time.time()
        with open(self._tmp, "w", encoding="utf-8") as f:
            json.dump(snap.to_dict(), f, indent=2, default=str)
            f.flush()
        os.replace(self._tmp, self.path)

    def load(self) -> Optional[CycleSnapshot]:
        """Load snapshot from disk. Returns None if missing or corrupt."""
        try:
            if not os.path.exists(self.path):
                return None
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CycleSnapshot.from_dict(data)
        except Exception as exc:
            logger.warning("SnapshotStore.load failed: %s", exc)
            return None
