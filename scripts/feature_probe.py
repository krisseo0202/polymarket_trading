"""Pre-model feature-selection probe.

Runs a "sophisticated data scientist" pass over a training dataset so Claude
can decide which features belong in the main model. Produces:

  experiments/<run>/probe/
    importance_table.csv     # one row per feature + the random_value sentinel
    correlation_matrix.csv   # pairwise Pearson (tight)
    probe_models/
      logreg.pkl, scaler.pkl, xgb.json
    report.md                # human + Claude-readable summary

The sentinel row is produced by injecting a random_value column into the
dataset. Any feature whose XGB permutation-importance sits below the
sentinel is auto-flagged as "worse than random".

Usage:
    # Load from S3 over an inclusive date range:
    python scripts/feature_probe.py \\
        --start-date 2026-04-12 --end-date 2026-04-18 \\
        --out experiments/2026-04-18-familyC-probe

    # Or consume a pre-built parquet:
    python scripts/feature_probe.py --dataset experiments/foo/dataset.parquet \\
        --out experiments/foo/probe
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.backtest.fill_sim import FillConfig, as_dict, slot_pnl
from src.models.tte_weights import (
    BUCKET_BOUNDS,
    bucket_names,
    bucket_range,
    tte_series_to_buckets,
)
from src.backtest.period_split import (
    add_period_arguments,
    resolve_split_from_args,
    split_by_val_ratio,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required for feature_probe; pip install xgboost") from exc


_NON_FEATURE_COLUMNS = {
    "slot_ts",
    "snapshot_ts",
    "slot_expiry_ts",
    "label",
    "outcome",
    "feature_status",
    "question",
}

_DEFAULT_CORR_THRESHOLD = 0.85
_DEFAULT_VAL_RATIO = 0.20  # last 20% of slots by time
_PERM_N_REPEATS = 10
_TOP_N_IN_REPORT = 25


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_dataset(
    parquet_path: Optional[str],
    dates: Optional[List[str]],
    bucket: str,
    random_seed: int,
    n_sentinels: int,
) -> pd.DataFrame:
    """Either read a prebuilt parquet or pull from S3 with sentinels injected."""
    if parquet_path:
        df = pd.read_parquet(parquet_path)
        # Parquet may have been built without sentinels. Ensure at least one.
        if not any(c.startswith("random_value") for c in df.columns):
            _append_sentinels(df, seed=random_seed, n=n_sentinels)
        return df

    if not dates:
        raise ValueError("Provide --dataset or --start-date/--end-date")

    from src.backtest.s3_snapshot_loader import load_dates as s3_load_dates

    logging.info("Loading %d date(s) from s3://%s/", len(dates), bucket)
    return s3_load_dates(
        dates=dates,
        bucket=bucket,
        add_random_sentinel=True,
        random_seed=random_seed,
        n_random_sentinels=n_sentinels,
    )


def expand_date_range(start_date: str, end_date: str) -> List[str]:
    """Return the inclusive list of ISO dates between start_date and end_date."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Dates must be YYYY-MM-DD: {exc}") from exc
    if end < start:
        raise SystemExit(f"--end-date ({end_date}) is before --start-date ({start_date})")
    out: List[str] = []
    current = start
    while current <= end:
        out.append(current.isoformat())
        current += timedelta(days=1)
    return out


def _append_sentinels(df: pd.DataFrame, seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    if n == 1:
        df["random_value"] = rng.standard_normal(len(df))
        return
    for idx in range(n):
        df[f"random_value_{idx}"] = rng.standard_normal(len(df))


def split_by_slot(df: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy helper kept for tests/scripts/test_feature_probe.py. New code
    should call ``resolve_split_from_args`` so it also respects
    ``--training-period`` / ``--valid-period`` / ``--test-period``."""
    if "slot_ts" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have slot_ts and label columns")
    frames = split_by_val_ratio(df, val_ratio=val_ratio, ts_col="slot_ts")
    train, val = frames["training"], frames["validation"]
    logging.info(
        "Slot split: train=%d rows / %d slots, val=%d rows / %d slots",
        len(train), train["slot_ts"].nunique(), len(val), val["slot_ts"].nunique(),
    )
    return train, val


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Everything numeric that's not metadata or the label."""
    out: List[str] = []
    for col in df.columns:
        if col in _NON_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        out.append(col)
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@dataclass
class ProbeModels:
    scaler: StandardScaler
    logreg: LogisticRegression
    xgb: XGBClassifier
    feature_names: List[str]
    logreg_val_brier: float
    xgb_val_brier: float


def fit_probes(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str]
) -> ProbeModels:
    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=float)
    X_val = val_df[features].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    logreg = LogisticRegression(
        penalty="l1", C=0.1, solver="saga", max_iter=2_000, n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logreg.fit(X_train_s, y_train)
    logreg_p = logreg.predict_proba(X_val_s)[:, 1]
    logreg_brier = float(brier_score_loss(y_val, logreg_p))

    xgb = XGBClassifier(
        max_depth=4,
        n_estimators=400,
        learning_rate=0.05,
        eval_metric="logloss",
        early_stopping_rounds=25,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_p = xgb.predict_proba(X_val)[:, 1]
    xgb_brier = float(brier_score_loss(y_val, xgb_p))

    logging.info("Probe Brier: logreg=%.4f xgb=%.4f", logreg_brier, xgb_brier)

    return ProbeModels(
        scaler=scaler,
        logreg=logreg,
        xgb=xgb,
        feature_names=features,
        logreg_val_brier=logreg_brier,
        xgb_val_brier=xgb_brier,
    )


def permutation_importance_xgb(
    models: ProbeModels, val_df: pd.DataFrame, n_repeats: int
) -> np.ndarray:
    """Permutation importance on the XGB probe, on val — one number per feature."""
    X_val = val_df[models.feature_names].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=float)
    # scoring=neg_brier_score so higher = more important.
    result = permutation_importance(
        models.xgb,
        X_val,
        y_val,
        n_repeats=n_repeats,
        n_jobs=1,
        scoring="neg_brier_score",
        random_state=0,
    )
    return result.importances_mean


def per_feature_univariate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    fill_config: FillConfig,
) -> pd.DataFrame:
    """Per-feature univariate LogReg → val Brier + solo-PnL via real asks."""
    rows: List[Dict[str, float]] = []
    y_train = train_df["label"].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=float)

    for feat in features:
        x_tr = train_df[[feat]].to_numpy(dtype=float)
        x_va = val_df[[feat]].to_numpy(dtype=float)
        if np.nanstd(x_tr) == 0:
            rows.append({"feature": feat, "univariate_brier": 0.25,
                         "solo_pnl": 0.0, "solo_sharpe": 0.0, "solo_n_trades": 0,
                         "solo_entry_cost_bps": 0.0})
            continue
        scaler = StandardScaler().fit(x_tr)
        lr = LogisticRegression(max_iter=1000, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(scaler.transform(x_tr), y_train)
        p_val = lr.predict_proba(scaler.transform(x_va))[:, 1]
        brier = float(brier_score_loss(y_val, p_val))

        metrics = slot_pnl(val_df, p_val, config=fill_config)
        rows.append({
            "feature": feat,
            "univariate_brier": brier,
            "solo_pnl": metrics.pnl,
            "solo_sharpe": metrics.sharpe,
            "solo_n_trades": metrics.n_trades,
            "solo_entry_cost_bps": metrics.mean_entry_cost_bps,
        })
    return pd.DataFrame(rows).set_index("feature")


def time_stability(
    models: ProbeModels, val_df: pd.DataFrame, n_repeats: int
) -> np.ndarray:
    """Permutation importance on first-half vs second-half of val slots.

    Returns (importance_first - importance_second) per feature — large positive
    value = feature helped early and faded. Large negative = feature gained.
    """
    slots = np.sort(val_df["slot_ts"].unique())
    mid = len(slots) // 2
    first_half = val_df[val_df["slot_ts"].isin(slots[:mid])]
    second_half = val_df[val_df["slot_ts"].isin(slots[mid:])]
    if first_half.empty or second_half.empty:
        return np.zeros(len(models.feature_names), dtype=float)

    def _imp(frame: pd.DataFrame) -> np.ndarray:
        X = frame[models.feature_names].to_numpy(dtype=float)
        y = frame["label"].to_numpy(dtype=float)
        return permutation_importance(
            models.xgb,
            X,
            y,
            n_repeats=max(3, n_repeats // 2),
            n_jobs=1,
            scoring="neg_brier_score",
            random_state=1,
        ).importances_mean

    return _imp(first_half) - _imp(second_half)


def correlation_matrix(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return df[features].corr(method="pearson").round(4)


# ---------------------------------------------------------------------------
# Redundancy clusters
# ---------------------------------------------------------------------------


def find_redundancy_clusters(
    corr: pd.DataFrame, importance: pd.Series, threshold: float
) -> List[List[Tuple[str, float]]]:
    """Greedy clustering: features with |corr| > threshold form a cluster.

    Only features present in ``importance`` are considered (lets callers pass
    a pre-filtered series, e.g., above-random-floor only).
    """
    candidates = set(importance.index) & set(corr.columns)
    seen: set = set()
    clusters: List[List[Tuple[str, float]]] = []
    for feat in importance.sort_values(ascending=False).index:
        if feat in seen or feat not in candidates:
            continue
        cluster = [(feat, float(importance[feat]))]
        seen.add(feat)
        for other in candidates:
            if other in seen or other == feat:
                continue
            if abs(corr.at[feat, other]) > threshold:
                cluster.append((other, float(importance[other])))
                seen.add(other)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _compute_tte(df: pd.DataFrame) -> np.ndarray:
    """Seconds-to-expiry per row, derived from slot_expiry_ts - snapshot_ts.

    Preferred over the ``seconds_to_expiry`` feature column because that
    column may not be present if feature_columns selection excludes it.
    """
    expiry = df["slot_expiry_ts"].to_numpy(dtype=float)
    snap = df["snapshot_ts"].to_numpy(dtype=float)
    return np.maximum(0.0, expiry - snap)


def tte_bucket_analysis(
    val_df: pd.DataFrame,
    logreg_p: np.ndarray,
    xgb_p: np.ndarray,
    fill_config: FillConfig,
) -> pd.DataFrame:
    """Per-TTE-bucket diagnostics: predictability vs tradable edge.

    The distinction the user called out: **predictable is not the same as
    profitable**. Near-resolution rows (very_late) are trivially predictable
    but carry no edge. The core 60–180s window is where we need the model
    to be both calibrated AND tradable.

    Returns a DataFrame with one row per bucket containing:
      - ``n`` row count
      - ``brier_lr`` / ``brier_xgb`` calibrated Brier score on that bucket
      - ``log_loss_lr`` / ``log_loss_xgb`` log loss per bucket
      - ``ece_xgb`` expected calibration error (10-bin) on XGB probs
      - ``mean_spread_pct`` avg yes_spread_pct (proxy for exec cost)
      - ``solo_pnl_lr`` / ``solo_pnl_xgb`` realized edge after spread+fee
      - ``pass_rate`` pct of rows where XGB fires a trade under the same rule
    """
    tte = _compute_tte(val_df)
    buckets = tte_series_to_buckets(tte)
    y = val_df["label"].to_numpy(dtype=float)

    # Guard probabilities to avoid log(0) in log loss.
    eps = 1e-12
    p_lr = np.clip(logreg_p, eps, 1 - eps)
    p_xg = np.clip(xgb_p, eps, 1 - eps)

    mean_spread = val_df.get("yes_spread_pct")
    mean_spread_arr = (
        mean_spread.to_numpy(dtype=float) if mean_spread is not None else np.zeros(len(val_df))
    )

    rows: List[Dict[str, object]] = []
    for name in bucket_names():
        mask = buckets == name
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": name, "tte_range": bucket_range(name), "n": 0})
            continue

        yb = y[mask]
        plr = p_lr[mask]
        pxg = p_xg[mask]

        brier_lr = float(np.mean((plr - yb) ** 2))
        brier_xgb = float(np.mean((pxg - yb) ** 2))
        log_loss_lr = float(-np.mean(yb * np.log(plr) + (1 - yb) * np.log(1 - plr)))
        log_loss_xgb = float(-np.mean(yb * np.log(pxg) + (1 - yb) * np.log(1 - pxg)))

        # Expected calibration error (10-bin) on XGB — catches "confident but wrong"
        ece = 0.0
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(pxg, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        for b in range(10):
            bm = bin_idx == b
            if bm.any():
                ece += (bm.sum() / n) * abs(yb[bm].mean() - pxg[bm].mean())

        # Realized edge via the existing fill simulator, restricted to bucket rows.
        # slot_pnl returns a FillMetrics dataclass; wrap via as_dict for access.
        bucket_df = val_df[mask].reset_index(drop=True)
        solo_lr = as_dict(slot_pnl(bucket_df, plr, config=fill_config))
        solo_xg = as_dict(slot_pnl(bucket_df, pxg, config=fill_config))

        # Trade pass rate: fraction of bucket rows where XGB would have fired.
        threshold = fill_config.entry_threshold
        pass_rate = float(((pxg >= threshold) | (pxg <= 1 - threshold)).mean())

        rows.append({
            "bucket": name,
            "tte_range": bucket_range(name),
            "n": n,
            "brier_lr": brier_lr,
            "brier_xgb": brier_xgb,
            "log_loss_lr": log_loss_lr,
            "log_loss_xgb": log_loss_xgb,
            "ece_xgb": float(ece),
            "mean_spread_pct": float(np.mean(mean_spread_arr[mask])),
            "solo_pnl_lr": solo_lr["pnl"],
            "solo_sharpe_lr": solo_lr["sharpe"],
            "solo_n_trades_lr": solo_lr["n_trades"],
            "solo_pnl_xgb": solo_xg["pnl"],
            "solo_sharpe_xgb": solo_xg["sharpe"],
            "solo_n_trades_xgb": solo_xg["n_trades"],
            "pass_rate_xgb": pass_rate,
        })
    return pd.DataFrame(rows)


def sentinel_columns(features: List[str]) -> List[str]:
    return [f for f in features if f.startswith("random_value")]


def _to_markdown(df: pd.DataFrame) -> str:
    """Lightweight markdown table to avoid the tabulate dependency."""
    if df.empty:
        return "_empty_"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):d}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):+.4f}")
            elif isinstance(v, (bool, np.bool_)):
                cells.append("true" if bool(v) else "false")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_importance_table(
    models: ProbeModels,
    perm_importance: np.ndarray,
    stability_delta: np.ndarray,
    univariate_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    # Correlation with label (training set) — lets Claude see raw signal.
    y = train_df["label"].to_numpy(dtype=float)
    corr_with_label: Dict[str, float] = {}
    for feat in models.feature_names:
        col = train_df[feat].to_numpy(dtype=float)
        if np.nanstd(col) == 0:
            corr_with_label[feat] = 0.0
        else:
            corr_with_label[feat] = float(np.corrcoef(col, y)[0, 1])

    coefs = models.logreg.coef_[0]
    xgb_gain = models.xgb.feature_importances_

    df = pd.DataFrame({
        "feature": models.feature_names,
        "logreg_coef": coefs,
        "logreg_abs_coef": np.abs(coefs),
        "xgb_gain": xgb_gain,
        "xgb_perm_importance": perm_importance,
        "stability_delta": stability_delta,
        "corr_with_label": [corr_with_label[f] for f in models.feature_names],
    }).set_index("feature")

    df = df.join(univariate_df, how="left")

    sentinels = sentinel_columns(models.feature_names)
    if sentinels:
        floor = float(df.loc[sentinels, "xgb_perm_importance"].max())
        df["above_random_sentinel"] = df["xgb_perm_importance"] > floor
    else:
        df["above_random_sentinel"] = True

    return df.sort_values("xgb_perm_importance", ascending=False)


def write_report(
    out_dir: Path,
    table: pd.DataFrame,
    corr: pd.DataFrame,
    models: ProbeModels,
    dataset_stats: Dict[str, object],
    bucket_df: Optional[pd.DataFrame] = None,
) -> None:
    sentinels = [f for f in table.index if f.startswith("random_value")]
    floor = float(table.loc[sentinels, "xgb_perm_importance"].max()) if sentinels else 0.0

    above = table[table["above_random_sentinel"]].drop(sentinels, errors="ignore")
    below = table[(~table["above_random_sentinel"])].drop(sentinels, errors="ignore")
    clusters = find_redundancy_clusters(
        corr, above["xgb_perm_importance"], threshold=_DEFAULT_CORR_THRESHOLD
    )

    solo_surprises = above[
        ((above["logreg_coef"] > 0) & (above["solo_pnl"] < 0))
        | ((above["logreg_coef"] < 0) & (above["solo_pnl"] > 0))
    ]
    stability_flags = above[above["stability_delta"].abs() > above["xgb_perm_importance"].abs() * 0.5]

    with (out_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("# Feature Probe Report\n\n")
        f.write(f"- Dataset rows: {dataset_stats['n_rows']}\n")
        f.write(f"- Train slots: {dataset_stats['n_train_slots']}, Val slots: {dataset_stats['n_val_slots']}\n")
        f.write(f"- Features evaluated: {len(table)}\n")
        f.write(f"- Probe Brier — LogReg: `{models.logreg_val_brier:.4f}` · XGB: `{models.xgb_val_brier:.4f}`\n")
        f.write(f"- **Random sentinel floor (xgb_perm_importance):** `{floor:+.6f}` — ")
        f.write(f"{len(below)} feature(s) sit at or below this floor.\n\n")

        if bucket_df is not None and not bucket_df.empty:
            f.write("## TTE bucket analysis — predictability vs tradable edge\n\n")
            f.write(
                "Remember: **predictable is not the same as profitable**. "
                "Near-resolution rows (`very_late`) are trivial to predict but "
                "tight spreads and execution risk wipe out tradable edge. "
                "The `core` bucket (60–180s) is the real target.\n\n"
            )
            display_cols = [
                "bucket", "tte_range", "n",
                "brier_lr", "brier_xgb", "log_loss_xgb", "ece_xgb",
                "mean_spread_pct",
                "solo_pnl_lr", "solo_sharpe_lr",
                "solo_pnl_xgb", "solo_sharpe_xgb",
                "pass_rate_xgb",
            ]
            show = bucket_df[[c for c in display_cols if c in bucket_df.columns]]
            f.write(show.pipe(_to_markdown))
            f.write("\n\n")
            f.write(
                "Training uses sample weights from `src/models/tte_weights.py` "
                "to bias the loss toward the `core` bucket. Tune those weights "
                "if the realized-edge pattern above disagrees with the prior.\n\n"
            )

        f.write(f"## Top {_TOP_N_IN_REPORT} features by permutation importance\n\n")
        f.write(table.head(_TOP_N_IN_REPORT).reset_index().pipe(_to_markdown))
        f.write("\n\n")

        f.write("## Below random sentinel — auto-reject candidates\n\n")
        if below.empty:
            f.write("_None._\n\n")
        else:
            f.write(below.reset_index()[[
                "feature", "xgb_perm_importance", "xgb_gain",
                "logreg_abs_coef", "univariate_brier", "solo_pnl",
            ]].pipe(_to_markdown))
            f.write("\n\n")

        f.write(f"## Redundancy clusters (|corr| > {_DEFAULT_CORR_THRESHOLD})\n\n")
        if not clusters:
            f.write("_None._\n\n")
        else:
            for i, cluster in enumerate(clusters, 1):
                ranked = sorted(cluster, key=lambda t: t[1], reverse=True)
                head_feat = ranked[0][0]
                f.write(f"**Cluster {i}** (keep **{head_feat}**):\n")
                for feat, imp in ranked:
                    f.write(f"  - `{feat}`  perm_importance={imp:+.4f}\n")
                f.write("\n")

        f.write("## Solo-PnL surprises (coef sign disagrees with PnL sign)\n\n")
        if solo_surprises.empty:
            f.write("_None._\n\n")
        else:
            f.write(solo_surprises.reset_index()[[
                "feature", "logreg_coef", "solo_pnl", "solo_sharpe", "solo_n_trades",
            ]].pipe(_to_markdown))
            f.write("\n\n")

        f.write("## Stability flags (importance drop > 50% between val halves)\n\n")
        if stability_flags.empty:
            f.write("_None._\n\n")
        else:
            f.write(stability_flags.reset_index()[[
                "feature", "xgb_perm_importance", "stability_delta",
            ]].pipe(_to_markdown))
            f.write("\n\n")

        f.write("---\n")
        f.write("**Claude, pick features using this order:**\n")
        f.write("1. Drop every row in 'below random sentinel'.\n")
        f.write("2. Within each redundancy cluster, keep the bolded head and drop the rest (unless domain-relevant).\n")
        f.write("3. Review solo-PnL surprises manually.\n")
        f.write("4. Review stability flags; prefer keeping if the second-half delta is mild.\n")
        f.write("5. Emit `selection.yaml` next to this report.\n")


def save_probe_models(out_dir: Path, models: ProbeModels) -> None:
    probe_dir = out_dir / "probe_models"
    probe_dir.mkdir(parents=True, exist_ok=True)
    with (probe_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(models.scaler, f)
    with (probe_dir / "logreg.pkl").open("wb") as f:
        pickle.dump(models.logreg, f)
    models.xgb.save_model(str(probe_dir / "xgb.json"))
    (probe_dir / "feature_names.json").write_text(
        json.dumps(models.feature_names, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Target-feature fast path (single-feature sentinel gate)
# ---------------------------------------------------------------------------


def _run_target_feature_mode(args: argparse.Namespace) -> None:
    """Single-feature sentinel gate.

    Loads dataset (must already contain the target column + a random sentinel),
    fits XGB on train, computes permutation-importance on val, exits 0 if
    target's importance > the sentinel's, else exits 1.
    """
    target = args.target_feature
    if not args.dataset and not (args.start_date and args.end_date):
        raise SystemExit("--target-feature requires either --dataset or --start-date/--end-date")

    dates: Optional[List[str]] = None
    if args.start_date and args.end_date:
        dates = expand_date_range(args.start_date, args.end_date)
    df = load_dataset(
        parquet_path=args.dataset,
        dates=dates,
        bucket=args.bucket,
        random_seed=args.random_seed,
        n_sentinels=max(1, args.n_sentinels),
    )
    if df.empty:
        raise SystemExit("Dataset empty.")
    if target not in df.columns:
        raise SystemExit(
            f"Target feature {target!r} not present in dataset columns. "
            f"Inject the column before invoking the probe."
        )

    split = resolve_split_from_args(args, df, val_ratio=args.val_ratio)
    train_df = split.frames["training"]
    val_df = split.frames.get("validation")
    if val_df is None or val_df.empty:
        raise SystemExit("Val split empty.")

    feats = feature_columns(df)
    if target not in feats:
        raise SystemExit(
            f"Target {target!r} present in dataset but excluded from feature set "
            f"(check _NON_FEATURE_COLUMNS)."
        )
    sentinels = sentinel_columns(feats)
    if not sentinels:
        raise SystemExit("No sentinel column in dataset; cannot gate.")

    logging.info(
        "Target-feature mode: probing %r against sentinels %s on %d val rows",
        target, sentinels, len(val_df),
    )
    models = fit_probes(train_df, val_df, feats)
    perm = permutation_importance_xgb(models, val_df, n_repeats=args.perm_repeats)
    importances = dict(zip(feats, perm))
    target_imp = float(importances[target])
    sentinel_max = max(float(importances[s]) for s in sentinels)

    verdict = target_imp > sentinel_max
    print(
        f"target={target!r} importance={target_imp:.6f} "
        f"sentinel_max={sentinel_max:.6f} "
        f"verdict={'PASS' if verdict else 'FAIL'}"
    )
    raise SystemExit(0 if verdict else 1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-model feature probe")
    parser.add_argument("--dataset", default=None, help="Path to pre-built parquet dataset")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive lower bound (YYYY-MM-DD). Pair with --end-date.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive upper bound (YYYY-MM-DD). Pair with --start-date.",
    )
    parser.add_argument("--bucket", default="k-polymarket-data")
    parser.add_argument("--out", required=True, help="Output directory for probe artifacts")
    parser.add_argument("--val-ratio", type=float, default=_DEFAULT_VAL_RATIO)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-sentinels", type=int, default=1)
    parser.add_argument("--perm-repeats", type=int, default=_PERM_N_REPEATS)
    parser.add_argument(
        "--fee-bps", type=float, default=0.0,
        help="Flat per-trade fee in bps (for gas/protocol fees). Default 0.",
    )
    parser.add_argument(
        "--synthetic-half-spread", type=float, default=0.01,
        help="Fallback half-spread added to mid when real ask is missing.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--target-feature",
        default=None,
        help=(
            "Fast path: probe only this single named column against the random "
            "sentinel. Column must already exist in the dataset. Exits 0 if the "
            "target's permutation-importance > sentinel's, exits 1 otherwise. "
            "Skips report generation."
        ),
    )
    add_period_arguments(parser)
    args = parser.parse_args()

    if args.target_feature:
        _run_target_feature_mode(args)
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates: Optional[List[str]] = None
    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            raise SystemExit("--start-date and --end-date must be provided together")
        dates = expand_date_range(args.start_date, args.end_date)
        logging.info(
            "Expanded date range %s..%s → %d day(s)",
            args.start_date, args.end_date, len(dates),
        )

    df = load_dataset(
        parquet_path=args.dataset,
        dates=dates,
        bucket=args.bucket,
        random_seed=args.random_seed,
        n_sentinels=args.n_sentinels,
    )
    if df.empty:
        raise SystemExit("Dataset is empty — nothing to probe.")

    # Persist the dataset (incl. sentinels) so future probes / training reuse it.
    if not args.dataset:
        dataset_path = out_dir.parent / "dataset.parquet"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dataset_path, index=False)
        logging.info("Wrote dataset → %s", dataset_path)

    split = resolve_split_from_args(args, df, val_ratio=args.val_ratio)
    train_df = split.frames["training"]
    val_df = split.frames.get("validation")
    if val_df is None or val_df.empty:
        raise SystemExit(
            "Val split is empty — need more slots, a lower --val-ratio, or a valid --valid-period."
        )
    # --test-period isn't actionable here (feature probe doesn't evaluate on
    # a held-out set) but is accepted for API symmetry — log it for audit.
    if "test" in split.frames:
        logging.info(
            "Ignoring --test-period (%d rows); feature_probe has no final test phase.",
            len(split.frames["test"]),
        )

    features = feature_columns(df)
    logging.info("Probing %d features (incl. sentinels: %s)",
                 len(features), sentinel_columns(features))

    models = fit_probes(train_df, val_df, features)
    perm = permutation_importance_xgb(models, val_df, n_repeats=args.perm_repeats)
    stability = time_stability(models, val_df, n_repeats=args.perm_repeats)
    fill_cfg = FillConfig(
        synthetic_half_spread=args.synthetic_half_spread,
        fee_bps=args.fee_bps,
    )
    univar = per_feature_univariate(train_df, val_df, features, fill_config=fill_cfg)
    corr = correlation_matrix(train_df, features)

    table = build_importance_table(models, perm, stability, univar, train_df)
    table.to_csv(out_dir / "importance_table.csv")
    corr.to_csv(out_dir / "correlation_matrix.csv")
    save_probe_models(out_dir, models)

    # TTE bucket analysis: per-bucket predictability vs tradable edge so the
    # user can validate the emphasize-core assumption with real numbers.
    X_val = val_df[features].to_numpy(dtype=float)
    lr_p = models.logreg.predict_proba(models.scaler.transform(X_val))[:, 1]
    xgb_p = models.xgb.predict_proba(X_val)[:, 1]
    bucket_df = tte_bucket_analysis(val_df, lr_p, xgb_p, fill_config=fill_cfg)
    bucket_df.to_csv(out_dir / "tte_bucket_analysis.csv", index=False)

    write_report(
        out_dir,
        table,
        corr,
        models,
        dataset_stats={
            "n_rows": len(df),
            "n_train_slots": train_df["slot_ts"].nunique(),
            "n_val_slots": val_df["slot_ts"].nunique(),
        },
        bucket_df=bucket_df,
    )
    print(f"Probe complete → {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
