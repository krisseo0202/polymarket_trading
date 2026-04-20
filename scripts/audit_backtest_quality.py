"""
Quality audit of the logreg toy backtest.

Checks:
  1. Train/test contract split is time-disjoint (no row overlap).
  2. Feature coefficients — is dist_to_strike dominating?
  3. Metrics by time-to-expiry bucket. If early-tte rows have Brier ~= 0.25 and
     AUC ~= 0.5, then the "good" calibration numbers come from near-expiry rows
     where dist_to_strike ≈ outcome (leakage).
  4. Reliability curve: predicted p vs observed frequency per decile.
  5. Trading PnL restricted to early-tte entries (what a live bot sees).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.metrics import brier_score_loss, log_loss

from scripts.backtest_logreg_edge import (
    FEATURES, build_merged_dataset, load_and_derive, train_model,
)
from scripts.sweep_backtest_logreg import find_trades_first_eligible


def _safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--taker-fee", type=float, default=0.072,
                    help="Polymarket taker fee rate (default 7.2% for crypto)")
    args = ap.parse_args()

    btc_df, markets_df, ob_df = load_and_derive(
        os.path.join(_ROOT, "data", "backtest_working", "btc_1s.csv"),
        os.path.join(_ROOT, "data", "backtest_working", "live_orderbook_snapshots.csv"),
    )
    merged = build_merged_dataset(btc_df, markets_df, ob_df, 15)
    model, scaler, train_df, test_df, p_hat = train_model(merged, 0.20)

    print("=" * 78)
    print("  Backtest Quality Audit")
    print("=" * 78)

    # ── 1. Split disjointness ───────────────────────────────────────────
    train_cids = set(train_df["contract_id"].unique())
    test_cids = set(test_df["contract_id"].unique())
    overlap = train_cids & test_cids
    train_last = max(train_cids) if train_cids else None
    test_first = min(test_cids) if test_cids else None
    print(f"\n  1) Split disjointness")
    print(f"     Train contracts : {len(train_cids)} (last  slot_ts={train_last})")
    print(f"     Test  contracts : {len(test_cids)} (first slot_ts={test_first})")
    print(f"     Overlap         : {len(overlap)}  {'LEAK' if overlap else 'OK'}")
    print(f"     Time-ordered    : {'OK' if test_first and train_last and test_first > train_last else 'LEAK'}")

    # ── 2. Feature coefficients ─────────────────────────────────────────
    print(f"\n  2) Feature coefficients (scaled, sorted by |coef|)")
    coefs = list(zip(FEATURES, model.coef_[0]))
    for feat, c in sorted(coefs, key=lambda x: abs(x[1]), reverse=True):
        marker = "  <-- SUSPECT" if feat == "dist_to_strike" else ""
        print(f"     {feat:<22s} {c:+.4f}{marker}")
    print(f"     {'intercept':<22s} {model.intercept_[0]:+.4f}")

    # ── 3. Metrics by time-to-expiry bucket ─────────────────────────────
    print(f"\n  3) Row-level metrics by time-to-expiry bucket")
    print(f"     (if early-tte rows are weak, the 'good' aggregate is from leak)")
    buckets = [(0, 30, "≤30s — near expiry (leaky zone)"),
               (30, 60, "30-60s"),
               (60, 120, "60-120s"),
               (120, 180, "120-180s"),
               (180, 240, "180-240s"),
               (240, 301, "≥240s — what a live bot sees")]
    print(f"\n     {'bucket':>32}  {'n':>5}  {'Brier':>7}  {'LogL':>7}  {'AUC':>6}  {'Acc':>6}")
    y = test_df["target_up"].to_numpy(dtype=float)
    tte = test_df["time_to_expiry_sec"].to_numpy(dtype=float)
    for lo, hi, label in buckets:
        mask = (tte >= lo) & (tte < hi)
        n = int(mask.sum())
        if n < 5:
            print(f"     {label:>32}  {n:>5}  (too few)")
            continue
        b = brier_score_loss(y[mask], p_hat[mask])
        ll = log_loss(y[mask], p_hat[mask], labels=[0.0, 1.0])
        a = _safe_auc(y[mask], p_hat[mask])
        acc = float(((p_hat[mask] >= 0.5).astype(int) == y[mask].astype(int)).mean())
        print(f"     {label:>32}  {n:>5}  {b:>7.4f}  {ll:>7.4f}  {a:>6.3f}  {acc:>5.1%}")

    # ── 4. Reliability curve ────────────────────────────────────────────
    print(f"\n  4) Reliability curve (predicted p vs observed frequency)")
    print(f"     Well-calibrated: |pred - obs| small across every decile")
    print(f"\n     {'decile':>12}  {'n':>5}  {'pred':>6}  {'obs':>6}  {'gap':>7}")
    edges = np.linspace(0.0, 1.0, 11)
    total_ece_rows = 0
    total_ece_gap = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (p_hat >= lo) & (p_hat <= hi) if i == 9 else (p_hat >= lo) & (p_hat < hi)
        n = int(mask.sum())
        if n < 3:
            continue
        pred = float(p_hat[mask].mean())
        obs = float(y[mask].mean())
        total_ece_rows += n
        total_ece_gap += n * abs(pred - obs)
        flag = "  OK" if abs(pred - obs) < 0.05 else "  **"
        print(f"     {lo:>4.2f}-{hi:>4.2f}  {n:>5}  {pred:>6.3f}  {obs:>6.3f}  "
              f"{pred-obs:>+7.3f}{flag}")

    # ── 5. Trading restricted to early-tte entries ──────────────────────
    print(f"\n  5) Trading PnL restricted to first-eligible entries with tte ≥ 240s")
    print(f"     (honest lower bound — the bot must decide in the first minute)")
    # Filter test_df to only early-tte rows and re-run first-eligible picker
    test_early = test_df[test_df["time_to_expiry_sec"] >= 240].copy()
    print(f"     Early-tte rows: {len(test_early)} of {len(test_df)}  "
          f"({test_early['contract_id'].nunique()} contracts)")
    fee_str = "0%" if args.taker_fee == 0 else f"{args.taker_fee*100:.1f}%×p(1-p)"
    print(f"     (taker fee = {fee_str})")
    print(f"\n     {'delta':>6}  {'n':>4}  {'win%':>6}  {'fees':>7}  {'totPnL':>9}  {'avgPnL':>8}  {'ROI%':>7}")
    for d in [0.01, 0.02, 0.03, 0.04, 0.05]:
        trades, _ = find_trades_first_eligible(test_early, d, 20.0, taker_fee_rate=args.taker_fee)
        if not trades:
            print(f"     {d:>6.3f}  {'—':>4}")
            continue
        pnl = np.array([t["pnl"] for t in trades])
        fees = sum(float(t.get("fee", 0.0)) for t in trades)
        wins = sum(1 for t in trades if t["win"])
        roi = pnl.sum() / (len(trades) * 20.0) * 100
        print(f"     {d:>6.3f}  {len(trades):>4}  {wins/len(trades)*100:>5.1f}%  "
              f"${fees:>5.2f}  ${pnl.sum():>+7.2f}  ${pnl.mean():>+6.2f}  {roi:>+6.2f}%")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
