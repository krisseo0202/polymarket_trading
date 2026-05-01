"""Feature-family ablation study.

For each group in ``src.models.feature_group_map.FEATURE_GROUPS``: drop all
features in that group, retrain LogReg + XGB on the same train/val split,
backtest on the test split with realistic fills, and record:

    group | n_features_dropped | dpnl | dbrier | dsharpe | dwin_rate | n_trades

The deltas are vs the BASELINE model (all features active). Negative ΔPNL
means dropping the group costs P&L (the group is load-bearing); positive
ΔPNL means the model performed better without the group (drop candidate).

Fixed test window is REQUIRED. The script raises if ``--test-period`` is
absent — comparing different windows defeats the purpose of ablation.

Usage:
    python scripts/feature_ablation.py \\
        --selection experiments/signed-v1/selection.yaml \\
        --training-period 2026-04-15..2026-04-21 \\
        --valid-period   2026-04-21..2026-04-22 \\
        --test-period    2026-04-22..2026-04-29 \\
        --out experiments/ablation/$(date +%Y%m%d-%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import brier_score_loss

from src.backtest.fill_sim import FillConfig
from src.backtest.period_split import add_period_arguments, resolve_split_from_args
from src.models.feature_group_map import FEATURE_GROUPS

# train.py is a script, not a module under src/. Import via path.
import importlib.util
_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "train.py")
_train_spec = importlib.util.spec_from_file_location("_train_mod", _TRAIN_PATH)
_train_mod = importlib.util.module_from_spec(_train_spec)
sys.modules["_train_mod"] = _train_mod
_train_spec.loader.exec_module(_train_mod)

load_selection = _train_mod.load_selection
train_logreg = _train_mod.train_logreg
train_xgb = _train_mod.train_xgb
slot_pnl = _train_mod.slot_pnl
per_side_calibration_gap = _train_mod.per_side_calibration_gap
compute_tte_sample_weights = _train_mod.compute_tte_sample_weights


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    label: str
    n_features: int
    brier: float
    sharpe: float
    pnl: float
    win_rate: float
    n_trades: int
    yes_calibration_gap: float
    no_calibration_gap: float

    @classmethod
    def from_xgb_eval(
        cls,
        label: str,
        feature_names: List[str],
        val_df: pd.DataFrame,
        p_val: np.ndarray,
        fill_cfg: FillConfig,
    ) -> "RunMetrics":
        slot_metrics = slot_pnl(val_df, p_val, config=fill_cfg)
        gaps = per_side_calibration_gap(val_df, p_val)
        return cls(
            label=label,
            n_features=len(feature_names),
            brier=float(brier_score_loss(val_df["label"].to_numpy(dtype=float), p_val)),
            sharpe=float(slot_metrics["sharpe"]),
            pnl=float(slot_metrics["pnl"]),
            win_rate=float(slot_metrics["win_rate"]),
            n_trades=int(slot_metrics["n_trades"]),
            yes_calibration_gap=float(gaps["yes_calibration_gap"]),
            no_calibration_gap=float(gaps["no_calibration_gap"]),
        )


# ---------------------------------------------------------------------------
# Train + eval one feature subset
# ---------------------------------------------------------------------------


def _fit_and_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    fill_cfg: FillConfig,
    label: str,
    use_tte_weights: bool = False,
) -> RunMetrics:
    """Train XGB on (train, val), predict on test, return metrics on test."""
    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=float)
    X_val = val_df[features].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=float)
    X_test = test_df[features].to_numpy(dtype=float)

    sw = compute_tte_sample_weights(train_df, enabled=use_tte_weights) if use_tte_weights else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_model, xgb_cal, _ = train_xgb(X_train, y_train, X_val, y_val, sample_weight=sw)
    raw = xgb_model.predict_proba(X_test)[:, 1]
    p_test = xgb_cal.transform(raw)
    return RunMetrics.from_xgb_eval(label, features, test_df, p_test, fill_cfg)


# ---------------------------------------------------------------------------
# Ablation loop
# ---------------------------------------------------------------------------


def run_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    active_features: List[str],
    fill_cfg: FillConfig,
    groups: Optional[List[str]] = None,
    use_tte_weights: bool = False,
) -> Tuple[RunMetrics, List[Dict[str, float]]]:
    """Run baseline + per-group ablation. Return (baseline, rows)."""
    baseline = _fit_and_eval(
        train_df, val_df, test_df, active_features, fill_cfg, label="baseline",
        use_tte_weights=use_tte_weights,
    )
    logging.info(
        "BASELINE | n=%d | brier=%.4f sharpe=%.2f pnl=%.2f wr=%.3f trades=%d",
        baseline.n_features, baseline.brier, baseline.sharpe,
        baseline.pnl, baseline.win_rate, baseline.n_trades,
    )

    targets = list(groups) if groups else list(FEATURE_GROUPS.keys())
    rows: List[Dict[str, float]] = []
    active_set = set(active_features)

    for group in targets:
        members = FEATURE_GROUPS.get(group, [])
        # Only drop members that are actually active in the selection.
        drop_set = active_set & set(members)
        if not drop_set:
            logging.warning("group=%s has 0 features in active selection — skipping", group)
            continue
        ablated = [f for f in active_features if f not in drop_set]
        if not ablated:
            logging.warning("group=%s would drop ALL features — skipping", group)
            continue

        m = _fit_and_eval(
            train_df, val_df, test_df, ablated, fill_cfg, label=f"drop_{group}",
            use_tte_weights=use_tte_weights,
        )
        delta = {
            "group": group,
            "n_dropped": len(drop_set),
            "n_remaining": len(ablated),
            "brier": m.brier,
            "sharpe": m.sharpe,
            "pnl": m.pnl,
            "win_rate": m.win_rate,
            "n_trades": m.n_trades,
            "yes_calibration_gap": m.yes_calibration_gap,
            "no_calibration_gap": m.no_calibration_gap,
            "dpnl": m.pnl - baseline.pnl,
            "dbrier": m.brier - baseline.brier,
            "dsharpe": m.sharpe - baseline.sharpe,
            "dwin_rate": m.win_rate - baseline.win_rate,
        }
        rows.append(delta)
        logging.info(
            "DROP %-22s | dpnl=%+8.2f dbrier=%+0.4f dsharpe=%+0.2f dwr=%+0.3f trades=%d",
            group, delta["dpnl"], delta["dbrier"], delta["dsharpe"],
            delta["dwin_rate"], delta["n_trades"],
        )

    return baseline, rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-family feature ablation study")
    parser.add_argument("--selection", required=True, help="Path to selection.yaml")
    parser.add_argument("--out", required=True, help="Output dir for ablation_results.csv")
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Only ablate these groups (default: all 12).",
    )
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--synthetic-half-spread", type=float, default=0.01)
    parser.add_argument("--use-tte-weights", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    add_period_arguments(parser)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not getattr(args, "test_period", None):
        raise SystemExit(
            "--test-period is REQUIRED for feature_ablation. "
            "All ablation runs must be evaluated on the same held-out window."
        )

    sel = load_selection(args.selection)
    logging.info("Loaded selection: %s (%d active features)",
                 sel.run_id, len(sel.active_features))

    df = pd.read_parquet(sel.probe_dataset)
    if df.empty:
        raise SystemExit("Probe dataset is empty.")
    logging.info("Loaded dataset %s — %d rows", sel.probe_dataset, len(df))

    split = resolve_split_from_args(args, df, val_ratio=args.val_ratio)
    train_df = split.frames["training"]
    val_df = split.frames.get("validation")
    test_df = split.frames.get("test")
    if val_df is None or val_df.empty:
        raise SystemExit("Val split empty.")
    if test_df is None or test_df.empty:
        raise SystemExit("Test split empty — pass --test-period explicitly.")
    logging.info(
        "Splits: train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    fill_cfg = FillConfig(
        fee_bps=args.fee_bps,
        synthetic_half_spread=args.synthetic_half_spread,
    )

    baseline, rows = run_ablation(
        train_df, val_df, test_df,
        active_features=sel.active_features,
        fill_cfg=fill_cfg,
        groups=args.groups,
        use_tte_weights=args.use_tte_weights,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_results.csv"
    fieldnames = [
        "group", "n_dropped", "n_remaining", "n_trades",
        "brier", "sharpe", "pnl", "win_rate",
        "yes_calibration_gap", "no_calibration_gap",
        "dpnl", "dbrier", "dsharpe", "dwin_rate",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    (out_dir / "baseline.json").write_text(
        json.dumps(asdict(baseline), indent=2), encoding="utf-8"
    )
    logging.info("Wrote %s and baseline.json", csv_path)

    # Pretty print top winners (positive dpnl) and losers (most-negative dpnl)
    rows_sorted = sorted(rows, key=lambda r: r["dpnl"], reverse=True)
    print("\nAblation results (sorted by ΔPNL desc):")
    print(f"{'group':<22} {'dpnl':>8} {'dbrier':>8} {'dsharpe':>8} {'dwr':>7} {'n':>5}")
    for r in rows_sorted:
        print(
            f"{r['group']:<22} {r['dpnl']:+8.2f} {r['dbrier']:+0.4f} "
            f"{r['dsharpe']:+8.2f} {r['dwin_rate']:+0.3f} {r['n_trades']:>5}"
        )


if __name__ == "__main__":
    main()
