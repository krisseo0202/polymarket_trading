"""Crash-safe JSON state persistence for the BTC Up/Down bot"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, List, Optional


@dataclass
class BotState:
    """Persistent bot state — survives restarts and crashes"""
    active_order_ids: Dict[str, Optional[str]] = field(default_factory=dict)
    # token_id -> order_id (None means no resting order for that token)

    daily_realized_pnl: float = 0.0
    daily_reset_date: str = ""        # "YYYY-MM-DD"; reset when date changes
    session_wins: int = 0
    session_losses: int = 0
    cycle_count: int = 0
    inventories: Dict[str, Dict] = field(default_factory=dict)
    # token_id -> {"position": float, "avg_cost": float}

    # Strategy snapshot fields (written every cycle + every 30s intra-cycle)
    strategy_name: str = ""
    strategy_bias: str = ""                     # "LONG" | "SHORT" | "NONE"
    strategy_zscore: Optional[float] = None     # btc_vol_reversion only
    strategy_momentum_pct: Optional[float] = None  # btc_updown only
    strategy_prob_yes: Optional[float] = None
    strategy_prob_no: Optional[float] = None
    strategy_edge_yes: Optional[float] = None
    strategy_edge_no: Optional[float] = None
    strategy_net_edge_yes: Optional[float] = None
    strategy_net_edge_no: Optional[float] = None
    strategy_expected_fill_yes: Optional[float] = None
    strategy_expected_fill_no: Optional[float] = None
    strategy_tte_seconds: Optional[float] = None
    strategy_distance_to_break_pct: Optional[float] = None
    strategy_distance_to_strike_bps: Optional[float] = None
    strategy_model_version: str = ""
    strategy_feature_status: str = ""
    strategy_last_signal: str = ""              # e.g. "BUY YES @ 0.6500 | momentum=2.3%"
    strategy_last_signal_ts: float = 0.0
    strategy_status: str = ""                   # "WATCHING" | "POSITION_OPEN" | "EXITED"
    strategy_score_breakdown: Optional[dict] = None  # sigmoid model feature contributions
    strategy_required_edge: Optional[float] = None   # dynamic TTE-scaled entry threshold

    # Chainlink reference price (slot-open "price to beat")
    chainlink_ref_price: Optional[float] = None
    chainlink_ref_slot_ts: Optional[int] = None
    chainlink_healthy: bool = False

    # Per-slot realized PnL — resets on each 5-min market rollover
    slot_realized_pnl: float = 0.0

    # Trade history — last 20 fills, newest last
    # Each entry: {"ts": float, "action": str, "outcome": str, "price": float, "size": float}
    trade_log: List[Dict] = field(default_factory=list)

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
                strategy_prob_yes=data.get("strategy_prob_yes"),
                strategy_prob_no=data.get("strategy_prob_no"),
                strategy_edge_yes=data.get("strategy_edge_yes"),
                strategy_edge_no=data.get("strategy_edge_no"),
                strategy_net_edge_yes=data.get("strategy_net_edge_yes"),
                strategy_net_edge_no=data.get("strategy_net_edge_no"),
                strategy_expected_fill_yes=data.get("strategy_expected_fill_yes"),
                strategy_expected_fill_no=data.get("strategy_expected_fill_no"),
                strategy_tte_seconds=data.get("strategy_tte_seconds"),
                strategy_distance_to_break_pct=data.get("strategy_distance_to_break_pct"),
                strategy_distance_to_strike_bps=data.get("strategy_distance_to_strike_bps"),
                strategy_model_version=data.get("strategy_model_version", ""),
                strategy_feature_status=data.get("strategy_feature_status", ""),
                strategy_last_signal=data.get("strategy_last_signal", ""),
                strategy_last_signal_ts=float(data.get("strategy_last_signal_ts", 0.0)),
                strategy_status=data.get("strategy_status", ""),
                strategy_score_breakdown=data.get("strategy_score_breakdown"),
                strategy_required_edge=data.get("strategy_required_edge"),
                chainlink_ref_price=data.get("chainlink_ref_price"),
                chainlink_ref_slot_ts=data.get("chainlink_ref_slot_ts"),
                chainlink_healthy=data.get("chainlink_healthy", False),
                slot_realized_pnl=float(data.get("slot_realized_pnl", 0.0)),
                trade_log=data.get("trade_log", []),
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
