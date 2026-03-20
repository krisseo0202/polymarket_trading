"""Crash-safe JSON state persistence for the BTC Up/Down bot"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, Optional


@dataclass
class BotState:
    """Persistent bot state — survives restarts and crashes"""
    active_order_ids: Dict[str, Optional[str]] = field(default_factory=dict)
    # token_id -> order_id (None means no resting order for that token)

    daily_realized_pnl: float = 0.0
    daily_reset_date: str = ""        # "YYYY-MM-DD"; reset when date changes
    cycle_count: int = 0
    inventories: Dict[str, Dict] = field(default_factory=dict)
    # token_id -> {"position": float, "avg_cost": float}

    # Strategy snapshot fields (written every cycle + every 30s intra-cycle)
    strategy_name: str = ""
    strategy_bias: str = ""                     # "LONG" | "SHORT" | "NONE"
    strategy_zscore: Optional[float] = None     # btc_vol_reversion only
    strategy_momentum_pct: Optional[float] = None  # btc_updown only
    strategy_last_signal: str = ""              # e.g. "BUY YES @ 0.6500 | momentum=2.3%"
    strategy_last_signal_ts: float = 0.0
    strategy_status: str = ""                   # "WATCHING" | "POSITION_OPEN" | "EXITED"

    def __post_init__(self):
        if not self.daily_reset_date:
            self.daily_reset_date = str(date.today())


class StateStore:
    """
    Atomic JSON state persistence.
    Writes to <path>.tmp then os.replace() so a mid-write crash
    never leaves a corrupt state file.
    """

    def __init__(self, path: str):
        self.path = path
        self._tmp_path = path + ".tmp"

    def load(self) -> BotState:
        """
        Load state from disk.  Returns a fresh default BotState if the
        file is missing, empty, or corrupt — never raises.
        """
        try:
            if not os.path.exists(self.path):
                return BotState()
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BotState(
                active_order_ids=data.get("active_order_ids", {}),
                daily_realized_pnl=float(data.get("daily_realized_pnl", 0.0)),
                daily_reset_date=data.get("daily_reset_date", str(date.today())),
                cycle_count=int(data.get("cycle_count", 0)),
                inventories=data.get("inventories", {}),
                strategy_name=data.get("strategy_name", ""),
                strategy_bias=data.get("strategy_bias", ""),
                strategy_zscore=data.get("strategy_zscore"),
                strategy_momentum_pct=data.get("strategy_momentum_pct"),
                strategy_last_signal=data.get("strategy_last_signal", ""),
                strategy_last_signal_ts=float(data.get("strategy_last_signal_ts", 0.0)),
                strategy_status=data.get("strategy_status", ""),
            )
        except Exception:
            # Corrupt / unreadable — start fresh rather than crashing
            return BotState()

    def save(self, state: BotState) -> None:
        """
        Atomically write state to disk.
        The tmp file is flushed and synced before the rename so the
        destination is always either the old or the new complete file.
        """
        data = asdict(state)
        with open(self._tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self._tmp_path, self.path)
