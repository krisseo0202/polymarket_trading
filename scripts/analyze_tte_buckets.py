"""
Analyze logistic regression model accuracy by TTE (time-to-expiry) bucket.

Usage:
    python scripts/analyze_tte_buckets.py
    python scripts/analyze_tte_buckets.py --ob data/live_orderbook_snapshots.csv --btc data/btc_live_1s.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_logreg_v2 import build_dataset, _derive_outcomes
from src.models.logreg_model import LR_FEATURES, LogRegModel


BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 180), (180, 240), (240, 300)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze model accuracy by TTE bucket")
    p.add_argument("--ob", default="data/live_orderbook_snapshots.csv")
    p.add_argument("--btc", default="data/btc_live_1s.csv")
    p.add_argument("--model-dir", default="models/logreg")
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--row-interval", type=int, default=15)
    return p.parse_args()


def _bucket_label(lo: int, hi: int) -> str:
    return f"{lo:>3d}-{hi:<3d}s"


def _print_table(title: str, df: pd.DataFrame, probs: np.ndarray) -> None:
    from sklearn.metrics import brier_score_loss

    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"{'TTE Bucket':>12s} | {'Count':>5s} | {'Acc':>6s} | {'Brier':>6s} | "
          f"{'P(Up)':>6s} | {'Up%':>6s} | {'CalGap':>6s}")
    print(f"{'-' * 12}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 6}-+-"
          f"{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}")

    tte = df["time_to_expiry_sec"].values
    y = df["target_up"].values

    for lo, hi in BUCKETS:
        mask = (tte >= lo) & (tte < hi)
        n = mask.sum()
        if n == 0:
            print(f"{_bucket_label(lo, hi):>12s} | {'---':>5s} |")
            continue

        y_b = y[mask]
        p_b = probs[mask]
        acc = float(np.mean((p_b >= 0.5) == y_b))
        brier = brier_score_loss(y_b, p_b)
        mean_p = float(p_b.mean())
        up_rate = float(y_b.mean())
        cal_gap = abs(mean_p - up_rate)

        print(f"{_bucket_label(lo, hi):>12s} | {n:>5d} | {acc:>6.3f} | {brier:>6.4f} | "
              f"{mean_p:>6.3f} | {up_rate:>6.3f} | {cal_gap:>6.3f}")

    # Overall
    acc_all = float(np.mean((probs >= 0.5) == y))
    brier_all = brier_score_loss(y, probs)
    print(f"{'-' * 12}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 6}-+-"
          f"{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}")
    print(f"{'Overall':>12s} | {len(y):>5d} | {acc_all:>6.3f} | {brier_all:>6.4f} | "
          f"{probs.mean():>6.3f} | {y.mean():>6.3f} | {abs(probs.mean() - y.mean()):>6.3f}")


def main() -> None:
    args = _parse_args()

    # Load data
    ob_df = pd.read_csv(args.ob)
    btc_df = pd.read_csv(args.btc)
    outcomes = _derive_outcomes(btc_df, ob_df["slot_ts"].unique())
    print(f"Slots with outcomes: {len(outcomes)}")

    # Build dataset
    dataset = build_dataset(ob_df, btc_df, outcomes, args.row_interval)
    print(f"Dataset: {len(dataset)} rows, {dataset['contract_id'].nunique()} contracts")

    # Load model
    model = LogRegModel.load(args.model_dir)
    if not model.ready:
        print("Model not found!", file=sys.stderr)
        sys.exit(1)

    # Predict on all rows
    X = dataset[LR_FEATURES].to_numpy(dtype=float)
    X_scaled = model._scaler.transform(X)
    probs = model._model.predict_proba(X_scaled)[:, 1]

    # Walk-forward split
    contracts = dataset["contract_id"].unique()
    split_idx = max(1, int(len(contracts) * (1.0 - args.valid_fraction)))
    train_contracts = set(contracts[:split_idx])
    valid_contracts = set(contracts[split_idx:])

    train_mask = dataset["contract_id"].isin(train_contracts).values
    valid_mask = dataset["contract_id"].isin(valid_contracts).values

    # Print tables
    _print_table("ALL DATA", dataset, probs)
    _print_table(f"TRAIN ({len(train_contracts)} contracts)", dataset[train_mask], probs[train_mask])
    _print_table(f"VALIDATION ({len(valid_contracts)} contracts)", dataset[valid_mask], probs[valid_mask])

    # Recommendation
    from sklearn.metrics import brier_score_loss

    print(f"\n{'=' * 72}")
    print("  RECOMMENDATION")
    print(f"{'=' * 72}")

    tte = dataset["time_to_expiry_sec"].values[valid_mask]
    y_v = dataset["target_up"].values[valid_mask]
    p_v = probs[valid_mask]

    weak_buckets = []
    strong_buckets = []
    for lo, hi in BUCKETS:
        mask = (tte >= lo) & (tte < hi)
        if mask.sum() < 5:
            continue
        acc = float(np.mean((p_v[mask] >= 0.5) == y_v[mask]))
        brier = brier_score_loss(y_v[mask], p_v[mask])
        if acc < 0.52 or brier > 0.25:
            weak_buckets.append((lo, hi, acc, brier))
        else:
            strong_buckets.append((lo, hi, acc, brier))

    if weak_buckets:
        print("  Weak TTE zones (accuracy < 52% or Brier > 0.25 on validation):")
        for lo, hi, acc, brier in weak_buckets:
            print(f"    {_bucket_label(lo, hi)}: acc={acc:.3f}, Brier={brier:.4f}")
    else:
        print("  No weak TTE zones found — model performs well across all buckets.")

    if strong_buckets:
        lo_min = min(b[0] for b in strong_buckets)
        hi_max = max(b[1] for b in strong_buckets)
        print(f"\n  Suggested TTE range: min_seconds_to_expiry={lo_min}, max_seconds_to_expiry={hi_max}")
    else:
        print("\n  Insufficient data for recommendation.")


if __name__ == "__main__":
    main()
