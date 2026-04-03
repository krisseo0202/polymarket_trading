"""Single source of truth for the current trading cycle state.

Written atomically by bot.py at the end of every cycle.
Read by dashboard.py to display consistent state without extra API calls.
"""

import dataclasses
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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

    Fields mirror the user-facing requirements:
      slot info, strike, btc_now, YES/NO bid/ask/mid,
      positions, active_orders, bot_status, pnl_state.
    """

    # ── Slot info ──────────────────────────────────────────────────────────────
    market_id: str = ""
    question: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    tte_seconds: Optional[float] = None

    # ── Strike ────────────────────────────────────────────────────────────────
    strike: Optional[float] = None

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

    # ── Positions: token_id → {"size": float, "avg_cost": float} ─────────────
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── Active orders (serialized Order dicts) ────────────────────────────────
    active_orders: List[Dict[str, Any]] = field(default_factory=list)

    # ── Bot operational status ────────────────────────────────────────────────
    bot_status: str = BotStatus.INIT.value

    # ── PnL ───────────────────────────────────────────────────────────────────
    daily_realized_pnl: float = 0.0
    slot_realized_pnl: float = 0.0
    unrealized_pnl: Optional[float] = None

    # ── Strategy telemetry ────────────────────────────────────────────────────
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

    # ── Metadata ──────────────────────────────────────────────────────────────
    cycle_count: int = 0
    paper_trading: bool = True
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleSnapshot":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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
