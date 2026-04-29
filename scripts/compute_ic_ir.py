"""Compute Information Coefficient (IC) and Information Ratio (IR) per feature.

IC per feature = Spearman correlation between the feature value at decision time
and the realized binary outcome (label = 1 if Up, 0 if Down) across snapshots.

Two IR flavors are reported:
  - Time-series IR per feature: mean(IC_t) / std(IC_t), where IC_t is computed
    on rolling time bins. Tells you how stable a signal is across time.
  - Combined cross-sectional IR: mean|IC| * sqrt(N_eff), where N_eff is the
    effective number of independent signals from PCA on the feature covariance
    (N_eff = (sum lambda)^2 / sum lambda^2). The naive IC*sqrt(N) overstates
    IR for our feature set because many features are highly collinear
    (e.g. btc_ret_5s/15s/30s/60s, yes_mid vs no_mid).

Usage:
    ./.venv/bin/python scripts/compute_ic_ir.py
    ./.venv/bin/python scripts/compute_ic_ir.py --per-slot
    ./.venv/bin/python scripts/compute_ic_ir.py --dataset data/snapshots_local.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backtest import (
    build_snapshot_dataset,
    load_btc_prices,
    load_market_history,
    load_probability_ticks,
)
from src.models.schema import FEATURE_COLUMNS

_BAD_STATUS_TOKENS = ("missing", "insufficient")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-feature IC and IR")
    p.add_argument("--dataset", default="data/snapshots_local.parquet",
                   help="Pre-built snapshot dataset (csv or parquet). Built from sources if absent.")
    p.add_argument("--market-csv", default="data/analysis/btc_updown_12h.csv")
    p.add_argument("--prob-ticks", default="data/probability_ticks.jsonl")
    p.add_argument("--btc-file", default="data/2026-04-12/btc_live_1s_20260412T041444Z.csv")
    p.add_argument("--synth-outcomes", action="store_true",
                   help="Synthesize Up/Down labels from BTC start vs end price when "
                        "the market CSV does not overlap the probability ticks. "
                        "Auto-enabled if no overlap is detected.")
    p.add_argument("--output", default="data/analysis/ic_ir_report.csv")
    p.add_argument("--min-variance", type=float, default=1e-12,
                   help="Drop features whose stdev across snapshots is below this")
    p.add_argument("--per-slot", action="store_true",
                   help="Use only the last snapshot per slot (removes intra-slot autocorrelation)")
    p.add_argument("--n-bins", type=int, default=10,
                   help="Number of time bins (quantiles of slot_ts) for time-series IR")
    p.add_argument("--min-bin-size", type=int, default=30,
                   help="Skip bins with fewer than this many rows when computing IC_t")
    return p.parse_args()


def _asof(ts: np.ndarray, px: np.ndarray, target: float):
    idx = int(np.searchsorted(ts, target, side="right")) - 1
    if idx < 0:
        return None
    return float(px[idx])


def _synthesize_markets(prob_ticks: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Label each unique slot using BTC start vs end-of-window price."""
    btc_ts = btc_df["ts"].astype(float).to_numpy()
    btc_px = btc_df["price"].astype(float).to_numpy()
    rows = []
    for slot_ts in sorted(prob_ticks["slot_ts"].unique()):
        slot_ts = int(slot_ts)
        start = _asof(btc_ts, btc_px, float(slot_ts))
        end = _asof(btc_ts, btc_px, float(slot_ts + 300))
        if start is None or end is None or start <= 0:
            continue
        rows.append({
            "slot_ts": slot_ts,
            "question": "synth",
            "outcome": "Up" if end >= start else "Down",
            "strike_price": start,
            "label": 1 if end >= start else 0,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("slot_ts").reset_index(drop=True)
    return out


def _load_or_build(args: argparse.Namespace) -> pd.DataFrame:
    if os.path.exists(args.dataset):
        if args.dataset.lower().endswith(".parquet"):
            return pd.read_parquet(args.dataset)
        return pd.read_csv(args.dataset)
    print(f"Dataset not found at {args.dataset}; building from sources...", flush=True)
    ticks = load_probability_ticks(args.prob_ticks)
    btc = load_btc_prices(args.btc_file) if args.btc_file else None
    if btc is None or btc.empty:
        raise SystemExit("BTC price file is required to build snapshots.")

    markets = load_market_history(args.market_csv) if os.path.exists(args.market_csv) else pd.DataFrame()
    overlap = 0
    if not markets.empty:
        overlap = len(set(markets["slot_ts"]).intersection(set(ticks["slot_ts"])))
    if args.synth_outcomes or overlap == 0:
        if not args.synth_outcomes:
            print(f"WARNING: market CSV has 0 slot overlap with prob ticks. "
                  f"Synthesizing Up/Down labels from BTC start vs end (exchange feed, "
                  f"not Chainlink).", flush=True)
        markets = _synthesize_markets(ticks, btc)
        print(f"Synthesized {len(markets)} resolved slots from BTC.", flush=True)

    df = build_snapshot_dataset(markets, ticks, btc_df=btc)
    if df.empty:
        raise SystemExit("Built dataset is empty. Check inputs.")
    return df


def _filter_usable(df: pd.DataFrame) -> pd.DataFrame:
    if "feature_status" not in df.columns:
        return df
    status = df["feature_status"].astype(str).str.lower()
    bad = status.apply(lambda s: any(tok in s for tok in _BAD_STATUS_TOKENS))
    return df.loc[~bad].copy()


def _active_features(df: pd.DataFrame, min_variance: float) -> Tuple[List[str], List[str]]:
    active, dropped = [], []
    for f in FEATURE_COLUMNS:
        if f not in df.columns:
            dropped.append(f"{f} (absent)")
            continue
        if df[f].std(ddof=0) <= min_variance:
            dropped.append(f"{f} (zero-var)")
            continue
        active.append(f)
    return active, dropped


def _per_feature_ic(df: pd.DataFrame, features: List[str], label: str) -> pd.DataFrame:
    y = df[label].astype(float)
    ics = df[features].corrwith(y, method="spearman")
    return pd.DataFrame({
        "feature": ics.index,
        "ic": ics.values,
        "n": len(df),
    })


def _time_series_ir(
    df: pd.DataFrame, features: List[str], label: str,
    n_bins: int, min_bin_size: int,
) -> pd.DataFrame:
    df = df.sort_values("slot_ts").copy()
    df["__bin"] = pd.qcut(df["slot_ts"], q=min(n_bins, df["slot_ts"].nunique()),
                          labels=False, duplicates="drop")
    bin_ics: List[pd.Series] = []
    for _, g in df.groupby("__bin"):
        if len(g) < min_bin_size:
            continue
        y = g[label].astype(float)
        if y.std(ddof=0) == 0:
            continue
        bin_ics.append(g[features].corrwith(y, method="spearman"))
    if not bin_ics:
        return pd.DataFrame({"feature": features, "ic_mean": np.nan,
                             "ic_std": np.nan, "ir_ts": np.nan, "n_bins": 0})
    mat = pd.concat(bin_ics, axis=1)
    counts = mat.notna().sum(axis=1)
    mean = mat.mean(axis=1)
    std = mat.std(axis=1, ddof=1)
    ir = mean / std.where(std > 0, np.nan)
    return pd.DataFrame({
        "feature": mean.index,
        "ic_mean": mean.values,
        "ic_std": std.values,
        "ir_ts": ir.values,
        "n_bins": counts.values,
    })


def _effective_n(df: pd.DataFrame, features: List[str]) -> float:
    X = df[features].to_numpy(dtype=float)
    std = X.std(axis=0, ddof=0)
    keep = std > 0
    if keep.sum() < 2:
        return float(keep.sum())
    Xs = (X[:, keep] - X[:, keep].mean(axis=0)) / std[keep]
    cov = np.cov(Xs, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    return float((s ** 2) / np.sum(eigvals ** 2))


def main() -> None:
    args = _parse_args()
    df = _load_or_build(args)
    print(f"Loaded snapshots: {len(df):,}", flush=True)

    df = _filter_usable(df)
    print(f"After dropping rows with missing/insufficient features: {len(df):,}", flush=True)

    if args.per_slot:
        df = df.sort_values("snapshot_ts").groupby("slot_ts", as_index=False).last()
        print(f"Per-slot aggregation (last snapshot per slot): {len(df):,} rows", flush=True)

    if df.empty or "label" not in df.columns:
        raise SystemExit("No usable rows or no label column.")

    features, dropped = _active_features(df, args.min_variance)
    print(f"Active features: {len(features)} / {len(FEATURE_COLUMNS)}")
    if dropped:
        print(f"  dropped ({len(dropped)}): {dropped}")

    ic = _per_feature_ic(df, features, "label")
    ir = _time_series_ir(df, features, "label", args.n_bins, args.min_bin_size)
    report = ic.merge(ir, on="feature", how="left")
    report["abs_ic"] = report["ic"].abs()
    report = report.sort_values("abs_ic", ascending=False).drop(columns="abs_ic")

    n_eff = _effective_n(df, features)
    mean_abs_ic = float(report["ic"].abs().mean())
    combined_ir = mean_abs_ic * np.sqrt(n_eff) if n_eff > 0 else 0.0
    naive_ir = mean_abs_ic * np.sqrt(len(features))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    report.to_csv(args.output, index=False)

    label_balance = float(df["label"].mean())

    print()
    print(f"Snapshots used:     {len(df):,}")
    print(f"Label balance (Up): {label_balance:.3f}")
    print(f"Features evaluated: {len(features)}")
    print(f"Mean |IC|:          {mean_abs_ic:.4f}")
    print(f"Effective N (PCA):  {n_eff:.2f}  (raw N={len(features)})")
    print(f"Combined IR:        {combined_ir:.3f}   <- mean|IC| * sqrt(N_eff)")
    print(f"Naive IC*sqrt(N):   {naive_ir:.3f}   <- inflated by collinearity")
    print(f"\nReport written to {args.output}")
    print("\nTop features by |IC|:")
    cols = ["feature", "ic", "ic_mean", "ic_std", "ir_ts", "n_bins"]
    cols = [c for c in cols if c in report.columns]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "  nan "
    print(report[cols].head(15).to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
