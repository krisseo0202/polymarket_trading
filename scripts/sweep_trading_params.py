"""Grid-sweep the bot's trading-gate parameters on a trained model.

For a given trained LogRegFBModel artifact + dataset, walks every combination
of (delta, min_confidence, max_spread_pct, entry_price_range, TTE window)
against the val split, simulating the bot's intra-cycle entry flow:

    - For each slot, walk snapshots in time order
    - First snapshot that passes every gate fires ONE entry (is_flat rule)
    - Hold to expiry — realized PnL from the fill simulator (real asks + fee)

Outputs a ranked CSV + a terminal summary. No model retraining — this is
post-hoc gate tuning for an already-fit artifact.

Usage:
    .venv/bin/python scripts/sweep_trading_params.py \\
        --model-dir models/full_week_tte_weighted \\
        --dataset experiments/full-week/dataset.parquet \\
        --out experiments/full-week/param_sweep.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.fill_sim import FillConfig, as_dict
from src.backtest.fill_sim import slot_pnl as _slot_pnl_impl
from src.backtest.period_split import split_by_val_ratio
from src.models.logreg_fb_model import LogRegFBModel


# Default grid — ~100 configs. Override via --grid if you want broader search.
_DEFAULT_GRID: Dict[str, List] = {
    "delta":               [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040],
    "min_confidence":      [0.50, 0.52, 0.55, 0.58, 0.60],
    "max_spread_pct":      [0.02, 0.03, 0.04, 0.06, 0.10],
    "min_entry_price":     [0.20, 0.30, 0.40],
    "max_entry_price":     [0.60, 0.70, 0.80],
    # TTE window (inclusive of both ends). Use (lo, hi) tuples.
    "tte_window":          [(120, 240), (60, 240), (30, 270), (10, 290)],
}


@dataclass
class GateConfig:
    """One row of the sweep — mirrors the bot's _check_entry gates."""
    delta: float
    min_confidence: float
    max_spread_pct: float
    min_entry_price: float
    max_entry_price: float
    tte_window: tuple  # (min_tte, max_tte) in seconds


@dataclass
class SweepResult:
    config: GateConfig
    pnl: float
    sharpe: float
    n_trades: int
    win_rate: float
    mean_entry_cost_bps: float
    n_slots_with_trade: int
    slot_coverage: float  # fraction of val slots where we fired a trade


# ---------------------------------------------------------------------------
# Entry-gate simulator
# ---------------------------------------------------------------------------


def _p_yes_for_rows(model: LogRegFBModel, df: pd.DataFrame) -> np.ndarray:
    """Score every row through the trained model. Returns calibrated p_yes."""
    X = df[model.feature_names].to_numpy(dtype=float)
    raw = model._model.predict_proba(model._scaler.transform(X))[:, 1]
    if model._calibrator is not None:
        raw = model._calibrator.transform(raw)
    return np.clip(raw, 0.0, 1.0)


def _apply_gates(
    df: pd.DataFrame, p_yes: np.ndarray, cfg: GateConfig
) -> pd.DataFrame:
    """Return a subframe of rows that pass every gate.

    Mirrors src/strategies/logreg_edge.py:_check_entry exactly. The caller
    must still apply the "first qualifying snapshot per slot" rule.
    """
    tte = (
        df["slot_expiry_ts"].to_numpy(dtype=float)
        - df["snapshot_ts"].to_numpy(dtype=float)
    )
    yes_mid = df.get("yes_mid")
    yes_spread_pct = df.get("yes_spread_pct")
    yes_ask = df.get("yes_ask")
    no_ask = df.get("no_ask")
    if any(v is None for v in (yes_mid, yes_spread_pct, yes_ask, no_ask)):
        return df.iloc[0:0]
    yes_mid = yes_mid.to_numpy(dtype=float)
    spread_pct = yes_spread_pct.to_numpy(dtype=float)
    yes_ask_arr = yes_ask.to_numpy(dtype=float)
    no_ask_arr = no_ask.to_numpy(dtype=float)

    # TTE gate
    tte_lo, tte_hi = cfg.tte_window
    tte_mask = (tte >= tte_lo) & (tte <= tte_hi)

    # Edge gate (taker-cost model: edge_yes = p - mid - half_spread)
    half_spread = spread_pct * yes_mid / 2.0
    edge_yes = p_yes - yes_mid - half_spread
    edge_no = yes_mid - p_yes - half_spread
    edge_best = np.maximum(edge_yes, edge_no)
    edge_mask = edge_best >= cfg.delta

    # Side + entry price
    chose_yes = edge_yes >= edge_no
    entry = np.where(chose_yes, yes_ask_arr, no_ask_arr)
    entry_valid = (entry > 0) & (entry >= cfg.min_entry_price) & (entry <= cfg.max_entry_price)

    # Confidence on chosen side
    prob_chosen = np.where(chose_yes, p_yes, 1.0 - p_yes)
    conf_mask = prob_chosen >= cfg.min_confidence

    # Spread
    spread_mask = spread_pct <= cfg.max_spread_pct

    passing = tte_mask & edge_mask & entry_valid & conf_mask & spread_mask
    return df[passing].copy().assign(
        _p_yes=p_yes[passing],
        _chose_yes=chose_yes[passing],
        _tte=tte[passing],
    )


def _first_pass_per_slot(passing_df: pd.DataFrame) -> pd.DataFrame:
    """Emulate the is_flat guard: one trade per slot, first qualifying snapshot."""
    if passing_df.empty:
        return passing_df
    return (
        passing_df.sort_values(["slot_ts", "snapshot_ts"])
        .groupby("slot_ts", as_index=False)
        .head(1)
    )


def _simulate_pnl(
    entries: pd.DataFrame, fill_cfg: FillConfig
) -> Dict[str, float]:
    """Realized PnL for the chosen entries via the shared fill simulator.

    slot_pnl uses p_yes ≥ 0.55 / ≤ 0.45 as the side decision, which matches
    our edge-gate side choice when min_confidence ≥ 0.50, so we can pass p_yes
    directly. Threshold in FillConfig is entry_threshold.
    """
    if entries.empty:
        return {
            "pnl": 0.0, "sharpe": 0.0, "n_trades": 0,
            "win_rate": 0.0, "mean_entry_cost_bps": 0.0,
        }
    p_yes = entries["_p_yes"].to_numpy(dtype=float)
    # slot_pnl re-applies its own side-choice rule; keep it aligned by using
    # the SAME fill config (entry_threshold defaults to 0.55 — close enough).
    return as_dict(_slot_pnl_impl(entries, p_yes, config=fill_cfg))


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------


def sweep(
    model: LogRegFBModel,
    val_df: pd.DataFrame,
    grid: Dict[str, List],
    fill_cfg: FillConfig,
) -> List[SweepResult]:
    """Run every combination in ``grid``. Returns all results, unsorted."""
    p_yes = _p_yes_for_rows(model, val_df)
    total_slots = val_df["slot_ts"].nunique()

    # Expand grid combinatorially
    keys = list(grid.keys())
    configs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg_dict = dict(zip(keys, combo))
        # Skip configs where min_entry_price >= max_entry_price (degenerate).
        if cfg_dict["min_entry_price"] >= cfg_dict["max_entry_price"]:
            continue
        configs.append(GateConfig(**cfg_dict))

    results: List[SweepResult] = []
    for cfg in configs:
        passing = _apply_gates(val_df, p_yes, cfg)
        entries = _first_pass_per_slot(passing)
        metrics = _simulate_pnl(entries, fill_cfg)
        n_slots_with_trade = int(entries["slot_ts"].nunique()) if not entries.empty else 0
        results.append(SweepResult(
            config=cfg,
            pnl=metrics["pnl"],
            sharpe=metrics["sharpe"],
            n_trades=metrics["n_trades"],
            win_rate=metrics["win_rate"],
            mean_entry_cost_bps=metrics["mean_entry_cost_bps"],
            n_slots_with_trade=n_slots_with_trade,
            slot_coverage=n_slots_with_trade / max(total_slots, 1),
        ))
    return results


def _results_to_dataframe(results: List[SweepResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "delta":              r.config.delta,
            "min_confidence":     r.config.min_confidence,
            "max_spread_pct":     r.config.max_spread_pct,
            "min_entry_price":    r.config.min_entry_price,
            "max_entry_price":    r.config.max_entry_price,
            "tte_lo":             r.config.tte_window[0],
            "tte_hi":             r.config.tte_window[1],
            "pnl":                r.pnl,
            "sharpe":             r.sharpe,
            "n_trades":           r.n_trades,
            "win_rate":           r.win_rate,
            "mean_entry_cost_bps": r.mean_entry_cost_bps,
            "n_slots_with_trade": r.n_slots_with_trade,
            "slot_coverage":      r.slot_coverage,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-sweep trading-gate parameters")
    parser.add_argument("--model-dir", required=True, help="LogRegFBModel artifact dir")
    parser.add_argument("--dataset", required=True, help="Parquet with slot_ts, snapshot_ts, label, features, yes/no asks")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--synthetic-half-spread", type=float, default=0.01)
    parser.add_argument(
        "--min-trades", type=int, default=20,
        help="Drop configs with fewer than N trades from the ranking (default 20).",
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    model = LogRegFBModel.load(args.model_dir)
    if not model.ready:
        raise SystemExit(f"Model at {args.model_dir} did not load cleanly.")
    logging.info("Loaded model: %d features, version=%s",
                 len(model.feature_names), model.model_version)

    df = pd.read_parquet(args.dataset)
    frames = split_by_val_ratio(df, val_ratio=args.val_ratio, ts_col="slot_ts")
    val_df = frames["validation"]
    logging.info("Val slice: %d rows, %d slots", len(val_df), val_df["slot_ts"].nunique())

    fill_cfg = FillConfig(
        synthetic_half_spread=args.synthetic_half_spread,
        fee_bps=args.fee_bps,
    )

    results = sweep(model, val_df, _DEFAULT_GRID, fill_cfg)
    df_results = _results_to_dataframe(results)

    # Filter, rank, persist.
    qualified = df_results[df_results["n_trades"] >= args.min_trades]
    by_pnl = qualified.sort_values("pnl", ascending=False).head(args.top_n)
    by_sharpe = qualified.sort_values("sharpe", ascending=False).head(args.top_n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.sort_values("pnl", ascending=False).to_csv(out_path, index=False)
    logging.info("Full sweep → %s (%d configs)", out_path, len(df_results))

    print(f"\n== Sweep done: {len(df_results)} configs ==")
    print(f"({len(qualified)} met the --min-trades={args.min_trades} threshold)\n")

    cols_to_show = [
        "delta", "min_confidence", "max_spread_pct", "min_entry_price", "max_entry_price",
        "tte_lo", "tte_hi", "pnl", "sharpe", "n_trades", "win_rate", "slot_coverage",
    ]

    print(f"Top {args.top_n} by PnL:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 240, "display.float_format", lambda v: f"{v:.3f}"):
        print(by_pnl[cols_to_show].to_string(index=False))

    print(f"\nTop {args.top_n} by Sharpe:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 240, "display.float_format", lambda v: f"{v:.3f}"):
        print(by_sharpe[cols_to_show].to_string(index=False))

    # Stability check: do the top-by-PnL configs also rank well by Sharpe?
    top_pnl_set = set(by_pnl.index)
    top_sharpe_set = set(by_sharpe.index)
    overlap = top_pnl_set & top_sharpe_set
    print(f"\nOverlap between top-{args.top_n} PnL and Sharpe lists: {len(overlap)} configs")


if __name__ == "__main__":
    main()
