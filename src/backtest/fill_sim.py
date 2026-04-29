"""Shared slot-level fill simulator — used by feature_probe and train.

Entry price rule:
    1. Prefer the real recorded ask (``yes_ask`` / ``no_ask``) — these come
       from the saved 5s orderbook snapshots, so they reflect the actual top
       of VWAP-5 at decision time.
    2. Fall back to ``mid + synthetic_half_spread`` only when ask is missing
       or non-positive (rare — about 4% of rows in practice).

Fees:
    Flat ``fee_bps`` subtracted from PnL on the entry side. Polymarket's
    protocol fee is currently 0, but this lets us model gas costs and any
    future fee schedule. Default 0.

Bet sizing:
    Each slot risks 1 unit at the entry price — pnl = (payout - entry) /
    entry. Consistent with the existing ``scripts/backtest_logreg_edge.py``
    convention so results are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


_MIN_PRICE = 0.01
_MAX_PRICE = 0.99
DEFAULT_SYNTHETIC_HALF_SPREAD = 0.01
ENTRY_THRESHOLD = 0.55  # p_up >= this → buy YES; 1 - p_up >= this → buy NO


@dataclass
class FillConfig:
    synthetic_half_spread: float = DEFAULT_SYNTHETIC_HALF_SPREAD
    fee_bps: float = 0.0  # per-trade fee in basis points (1 bps = 0.01% of entry)
    entry_threshold: float = ENTRY_THRESHOLD


@dataclass
class FillMetrics:
    pnl: float
    sharpe: float
    n_trades: int
    win_rate: float
    mean_entry_cost_bps: float  # avg effective cost = spread + fee, in bps of payout


def slot_pnl(
    val_df: pd.DataFrame,
    p_up: np.ndarray,
    config: Optional[FillConfig] = None,
) -> FillMetrics:
    """Slot-level PnL using real asks when available.

    ``val_df`` and ``p_up`` must be aligned row-for-row. Only the last snapshot
    per ``slot_ts`` becomes a candidate trade (matches the probe/train
    convention). Returns zero-trade metrics when nothing fires.
    """
    cfg = config or FillConfig()
    if "slot_ts" not in val_df.columns or len(val_df) == 0:
        return _empty()

    assert len(p_up) == len(val_df), \
        f"p_up length {len(p_up)} must match val_df rows {len(val_df)}"

    frame = val_df.copy()
    frame["_p_up"] = p_up
    last = (
        frame.sort_values("snapshot_ts")
        .groupby("slot_ts", as_index=False)
        .tail(1)
    )

    pnls: List[float] = []
    wins = 0
    entry_costs_bps: List[float] = []

    for _, row in last.iterrows():
        p = float(row["_p_up"])
        label = float(row["label"])

        if p >= cfg.entry_threshold:
            entry, real = _entry_price(row, side="yes", cfg=cfg)
            payout = 1.0 if label == 1.0 else 0.0
            pnl = _pnl_for_entry(entry, payout, cfg.fee_bps)
            pnls.append(pnl)
            wins += int(payout == 1.0)
            entry_costs_bps.append(_entry_cost_bps(row, "yes", entry, cfg))
        elif p <= (1.0 - cfg.entry_threshold):
            entry, real = _entry_price(row, side="no", cfg=cfg)
            payout = 1.0 if label == 0.0 else 0.0
            pnl = _pnl_for_entry(entry, payout, cfg.fee_bps)
            pnls.append(pnl)
            wins += int(payout == 1.0)
            entry_costs_bps.append(_entry_cost_bps(row, "no", entry, cfg))

    arr = np.asarray(pnls, dtype=float)
    if arr.size == 0:
        return _empty()

    sharpe = 0.0
    if arr.std(ddof=1) > 0:
        sharpe = float(arr.mean() / arr.std(ddof=1) * np.sqrt(arr.size))

    return FillMetrics(
        pnl=float(arr.sum()),
        sharpe=sharpe,
        n_trades=int(arr.size),
        win_rate=float(wins / arr.size),
        mean_entry_cost_bps=float(np.mean(entry_costs_bps)) if entry_costs_bps else 0.0,
    )


def _entry_price(row: pd.Series, side: str, cfg: FillConfig) -> tuple:
    """Return (entry_price, used_real_ask) clamped to [0.01, 0.99]."""
    ask_col = "yes_ask" if side == "yes" else "no_ask"
    mid_col = "yes_mid" if side == "yes" else "no_mid"
    ask = float(row.get(ask_col, 0.0) or 0.0)
    if ask > 0:
        return min(_MAX_PRICE, max(_MIN_PRICE, ask)), True
    mid = float(row.get(mid_col, 0.5) or 0.5)
    entry = min(_MAX_PRICE, max(_MIN_PRICE, mid + cfg.synthetic_half_spread))
    return entry, False


def _pnl_for_entry(entry: float, payout: float, fee_bps: float) -> float:
    """PnL per unit risked = (payout - entry)/entry, minus flat fee in bps."""
    gross = (payout - entry) / entry
    fee = fee_bps / 10_000.0
    return gross - fee


def _entry_cost_bps(row: pd.Series, side: str, entry: float, cfg: FillConfig) -> float:
    """Effective round-trip cost in bps of entry price.

    Reported for observability; not used in PnL. Sums the ask-vs-true-fair
    half-spread plus the fee. Treats mid as the "fair" reference.
    """
    mid_col = "yes_mid" if side == "yes" else "no_mid"
    mid = float(row.get(mid_col, 0.5) or 0.5)
    if mid <= 0:
        return cfg.fee_bps
    spread_bps = max(0.0, (entry - mid) / mid) * 10_000.0
    return spread_bps + cfg.fee_bps


def _empty() -> FillMetrics:
    return FillMetrics(pnl=0.0, sharpe=0.0, n_trades=0, win_rate=0.0, mean_entry_cost_bps=0.0)


def as_dict(metrics: FillMetrics) -> Dict[str, float]:
    """JSON-friendly dict for meta.json storage."""
    return {
        "pnl": metrics.pnl,
        "sharpe": metrics.sharpe,
        "n_trades": metrics.n_trades,
        "win_rate": metrics.win_rate,
        "mean_entry_cost_bps": metrics.mean_entry_cost_bps,
    }
