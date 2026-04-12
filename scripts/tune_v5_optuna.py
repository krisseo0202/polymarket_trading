"""
Multi-objective Optuna tuning for logreg_v5.

Objectives:
  minimize valid_brier, maximize final_equity

Search space (defaults — overridable via --search-space-file JSON):
  C              : log-uniform [1e-3, 10]
  row_interval   : categorical {5, 10, 15, 20, 30}
  edge_threshold : uniform [0.0, 0.08]
  kelly_mult     : uniform [0.05, 0.40]
  max_frac       : uniform [0.05, 0.25]

Feature subset: fixed for a single study. Overridable via
--feature-subset-file (JSON list of column names). Default = V5_FEATURES.

Usage:
  python scripts/tune_v5_optuna.py --trials 100 --round 0
  python scripts/tune_v5_optuna.py --trials 50 --round 1 \\
      --config experiments/v5/round_0/suggestion.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import NSGAIISampler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_and_compare_v5 as v5mod  # noqa: E402
from _v5_feature_docs import AVAILABLE_FEATURES  # noqa: E402

DEFAULT_SPACE = {
    "C":              {"type": "log_uniform", "low": 1e-3, "high": 10.0},
    "row_interval":   {"type": "categorical", "choices": [5, 10, 15, 20, 30]},
    "edge_threshold": {"type": "uniform", "low": 0.0, "high": 0.08},
    "kelly_mult":     {"type": "uniform", "low": 0.05, "high": 0.40},
    "max_frac":       {"type": "uniform", "low": 0.05, "high": 0.25},
}


# ── Dataset cache (per row_interval) ──────────────────────────────────────

_DATASETS: dict = {}


def get_dataset(ob_df: pd.DataFrame, btc_df: pd.DataFrame,
                labels: dict, row_interval: int) -> pd.DataFrame:
    if row_interval not in _DATASETS:
        _DATASETS[row_interval] = v5mod.build_dataset(
            ob_df, btc_df, labels, row_interval=row_interval
        )
    return _DATASETS[row_interval]


# ── Suggestion sampler ────────────────────────────────────────────────────

def suggest_param(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "log_uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if t == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    raise ValueError(f"unknown param type: {t}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--config", default=None,
                    help="JSON with 'feature_subset', 'hparam_ranges', 'seed_trial' "
                         "(typically a prior round's suggestion.json).")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, f"experiments/v5/round_{args.round}"
    )
    os.makedirs(out_dir, exist_ok=True)

    cfg = json.loads(open(args.config).read()) if args.config else {}
    space = cfg.get("hparam_ranges", DEFAULT_SPACE)
    features = cfg.get("feature_subset", v5mod.V5_FEATURES)
    seed_trial = cfg.get("seed_trial")

    # Validate feature subset
    bad = [f for f in features if f not in AVAILABLE_FEATURES]
    if bad:
        print(f"ERROR: unknown features in subset: {bad}", file=sys.stderr)
        sys.exit(1)
    if not features:
        print("ERROR: empty feature subset", file=sys.stderr); sys.exit(1)

    # Validate search space keys
    required_keys = {"C", "row_interval", "edge_threshold", "kelly_mult", "max_frac"}
    missing = required_keys - set(space.keys())
    if missing:
        print(f"ERROR: search space missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Round {args.round}  |  trials={args.trials}  |  features={len(features)}")
    print(f"out: {out_dir}")
    print(f"features: {features}")

    # Load raw inputs once
    ob_df = pd.read_csv(os.path.join(REPO_ROOT, "data/live_orderbook_snapshots.csv"))
    btc_df = pd.read_csv(os.path.join(REPO_ROOT, "data/btc_live_1s.csv"))
    labels = v5mod.v3.derive_labels(ob_df)
    print(f"Labels: {len(labels)} slots")

    def objective(trial: optuna.Trial):
        C = suggest_param(trial, "C", space["C"])
        row_interval = int(suggest_param(trial, "row_interval", space["row_interval"]))
        edge_threshold = float(suggest_param(trial, "edge_threshold", space["edge_threshold"]))
        kelly_mult = float(suggest_param(trial, "kelly_mult", space["kelly_mult"]))
        max_frac = float(suggest_param(trial, "max_frac", space["max_frac"]))

        df = get_dataset(ob_df, btc_df, labels, row_interval)
        # guard: every feature must exist in df
        miss = [f for f in features if f not in df.columns]
        if miss:
            raise optuna.TrialPruned(f"features missing from df: {miss}")

        metrics = v5mod.run_experiment(
            df, features,
            {"C": C, "edge_threshold": edge_threshold,
             "kelly_mult": kelly_mult, "max_frac": max_frac},
        )
        trial.set_user_attr("row_acc", metrics["row_acc"])
        trial.set_user_attr("slot_acc", metrics["slot_acc"])
        trial.set_user_attr("n_trades", metrics["n_trades"])
        trial.set_user_attr("win_rate", metrics["win_rate"])
        trial.set_user_attr("valid_logloss", metrics["valid_logloss"])
        return metrics["valid_brier"], metrics["final_equity"]

    storage = f"sqlite:///{out_dir}/study.db"
    study = optuna.create_study(
        study_name=f"logreg_v5_round_{args.round}",
        storage=storage,
        directions=["minimize", "maximize"],
        sampler=NSGAIISampler(seed=42),
        load_if_exists=True,
    )

    if seed_trial:
        try:
            study.enqueue_trial(seed_trial, skip_if_exists=True)
            print(f"Enqueued seed trial: {seed_trial}")
        except Exception as e:
            print(f"WARN: could not enqueue seed trial: {e}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    pareto = study.best_trials
    print(f"\nPareto front: {len(pareto)} trial(s)")
    top_rows = []
    for t in pareto:
        row = {
            "number": t.number,
            "brier": t.values[0],
            "final_equity": t.values[1],
            "params": t.params,
            **{k: t.user_attrs.get(k) for k in
               ("row_acc", "slot_acc", "n_trades", "win_rate", "valid_logloss")},
        }
        top_rows.append(row)
        print(f"  #{t.number:3d}  brier={t.values[0]:.4f}  equity=${t.values[1]:.2f}  "
              f"params={t.params}")

    top_rows.sort(key=lambda r: (r["brier"], -r["final_equity"]))
    with open(os.path.join(out_dir, "top_trials.json"), "w") as f:
        json.dump({
            "round": args.round,
            "n_trials": len(study.trials),
            "feature_subset": features,
            "search_space": space,
            "pareto": top_rows,
            "all_trials": [
                {
                    "number": t.number,
                    "brier": t.values[0] if t.values else None,
                    "final_equity": t.values[1] if t.values else None,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }, f, indent=2)

    # Pareto scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [t.values[0] for t in study.trials if t.values]
    ys = [t.values[1] for t in study.trials if t.values]
    px = [t.values[0] for t in pareto]
    py = [t.values[1] for t in pareto]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c="#888", s=12, alpha=0.5, label="all trials")
    ax.scatter(px, py, c="#d62728", s=40, label="Pareto front", zorder=3)
    ax.axhline(100, color="k", lw=0.5, ls="--")
    ax.set_xlabel("valid Brier (minimize)")
    ax.set_ylabel("final equity $ (maximize)")
    ax.set_title(f"logreg_v5 — round {args.round}  ({len(study.trials)} trials)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto.png"), dpi=120)

    print(f"\nSaved → {out_dir}/{{study.db,top_trials.json,pareto.png}}")


if __name__ == "__main__":
    main()
