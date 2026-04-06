"""
Train an XGBoost classifier for BTC Up/Down Polymarket probability trading.

Usage:
    ./.venv/bin/python scripts/train_btc_updown_xgb.py --input data/snapshots.parquet
    ./.venv/bin/python scripts/train_btc_updown_xgb.py --input data/snapshots.csv --output-dir models
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.models import DEFAULT_THRESHOLDS, FEATURE_COLUMNS, coerce_training_frame
from src.models.schema import MODEL_NAME

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - script-level dependency
    raise SystemExit(
        "xgboost is required for training. Install it in your environment first."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BTC Up/Down XGBoost model")
    parser.add_argument("--input", required=True, help="Snapshot dataset path (.csv or .parquet)")
    parser.add_argument("--output-dir", default="models", help="Directory for model artifacts")
    parser.add_argument("--label-col", default="label", help="Binary label column name")
    parser.add_argument("--time-col", default="snapshot_ts", help="Timestamp column for walk-forward split")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="Holdout fraction from the tail")
    parser.add_argument("--num-round", type=int, default=200, help="Boosting rounds")
    parser.add_argument("--eta", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--max-depth", type=int, default=4, help="Tree depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bet-size", type=float, default=20.0, help="Simulated holdout bet size in USDC when fill columns exist")
    parser.add_argument("--cost-per-share", type=float, default=0.0, help="Additional fee/cost per share for edge-aware labels (sim_yes_ask already includes spread)")
    return parser.parse_args()


def _load_frame(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _coerce_label(df: pd.DataFrame, label_col: str) -> pd.Series:
    if label_col in df.columns:
        label = df[label_col]
    elif "outcome" in df.columns:
        label = df["outcome"].map(lambda value: 1 if str(value).strip().lower() in {"up", "yes", "1", "true"} else 0)
    else:
        raise ValueError("Dataset must contain either the configured label column or an 'outcome' column")
    return pd.to_numeric(label, errors="coerce").fillna(0).astype(int)


def _walk_forward_split(
    df: pd.DataFrame,
    y: pd.Series,
    time_col: str,
    valid_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if time_col in df.columns:
        ordered = df.assign(_sort_ts=pd.to_datetime(df[time_col], errors="coerce")).sort_values("_sort_ts")
    else:
        ordered = df.copy()

    split_idx = max(1, int(len(ordered) * (1.0 - valid_fraction)))
    train = ordered.iloc[:split_idx].copy()
    valid = ordered.iloc[split_idx:].copy()
    y_train = y.loc[train.index]
    y_valid = y.loc[valid.index]
    return train, valid, y_train, y_valid


def _build_calibration(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, list]:
    if len(preds) == 0:
        return {"bin_edges": [0.0, 1.0], "bin_values": [0.5]}

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(preds, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0

    unique_edges = [float(edges[0])]
    for edge in edges[1:]:
        if float(edge) > unique_edges[-1]:
            unique_edges.append(float(edge))
    if len(unique_edges) < 2:
        unique_edges = [0.0, 1.0]

    values = []
    global_mean = float(np.mean(labels)) if len(labels) else 0.5
    for idx in range(len(unique_edges) - 1):
        left = unique_edges[idx]
        right = unique_edges[idx + 1]
        if idx == len(unique_edges) - 2:
            mask = (preds >= left) & (preds <= right)
        else:
            mask = (preds >= left) & (preds < right)
        if np.any(mask):
            values.append(float(np.mean(labels[mask])))
        else:
            values.append(global_mean)
    return {"bin_edges": unique_edges, "bin_values": values}


def _apply_calibration(preds: np.ndarray, calibration: Dict[str, list]) -> np.ndarray:
    edges = calibration["bin_edges"]
    values = calibration["bin_values"]
    calibrated = np.empty_like(preds, dtype=float)
    for idx, pred in enumerate(preds):
        pos = int(np.searchsorted(edges, pred, side="right") - 1)
        pos = max(0, min(pos, len(values) - 1))
        calibrated[idx] = float(values[pos])
    return calibrated


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-9
    probs = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def _simulate_holdout_pnl(valid_meta: pd.DataFrame, probs_yes: np.ndarray, y_valid: np.ndarray, bet_size: float) -> Dict[str, float]:
    if "sim_yes_ask" not in valid_meta.columns or "sim_no_ask" not in valid_meta.columns:
        return {}

    pnl_list = []
    pred_yes = probs_yes >= 0.5
    actual_yes = y_valid == 1

    for idx in range(len(probs_yes)):
        if pred_yes[idx]:
            entry = float(valid_meta.iloc[idx]["sim_yes_ask"])
            payout = 1.0 if actual_yes[idx] else 0.0
        else:
            entry = float(valid_meta.iloc[idx]["sim_no_ask"])
            payout = 1.0 if not actual_yes[idx] else 0.0
        entry = max(entry, 0.01)
        shares = bet_size / entry
        pnl_list.append(shares * (payout - entry))

    pnl_arr = np.asarray(pnl_list, dtype=float)
    result = {
        "holdout_accuracy": float(np.mean(pred_yes == actual_yes)) if len(pred_yes) else 0.0,
        "holdout_total_pnl": float(np.sum(pnl_arr)) if len(pnl_arr) else 0.0,
        "holdout_pnl_per_trade": float(np.mean(pnl_arr)) if len(pnl_arr) else 0.0,
        "holdout_trades": int(len(pnl_arr)),
    }

    # Edge-aware metrics from pre-computed columns
    if "edge_yes" in valid_meta.columns and "edge_no" in valid_meta.columns:
        chosen_edge = np.where(pred_yes, valid_meta["edge_yes"].to_numpy(dtype=float), valid_meta["edge_no"].to_numpy(dtype=float))
        result["holdout_mean_realized_edge"] = float(np.mean(chosen_edge)) if len(chosen_edge) else 0.0
        result["holdout_profitable_frac"] = float(np.mean(chosen_edge > 0)) if len(chosen_edge) else 0.0

    return result


def main() -> None:
    args = _parse_args()
    df = _load_frame(args.input)
    if df.empty:
        raise SystemExit("Input dataset is empty")

    y = _coerce_label(df, args.label_col)
    train_df, valid_df, y_train, y_valid = _walk_forward_split(df, y, args.time_col, args.valid_fraction)
    train_X = coerce_training_frame(train_df)
    valid_X = coerce_training_frame(valid_df)
    dtrain = xgb.DMatrix(train_X.to_numpy(dtype=float), label=y_train.to_numpy(dtype=float), feature_names=FEATURE_COLUMNS)
    dvalid = xgb.DMatrix(valid_X.to_numpy(dtype=float), label=y_valid.to_numpy(dtype=float), feature_names=FEATURE_COLUMNS)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "seed": args.seed,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )

    raw_valid = booster.predict(dvalid)
    calibration = _build_calibration(raw_valid, y_valid.to_numpy(dtype=float))
    calibrated_valid = _apply_calibration(raw_valid, calibration)
    backtest_metrics = _simulate_holdout_pnl(valid_df.reset_index(drop=True), calibrated_valid, y_valid.to_numpy(dtype=float), args.bet_size)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "btc_updown_xgb.json")
    feature_path = os.path.join(args.output_dir, "btc_updown_xgb_features.json")
    meta_path = os.path.join(args.output_dir, "btc_updown_xgb_meta.json")

    booster.save_model(model_path)
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": f"{MODEL_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "feature_columns": FEATURE_COLUMNS,
        "thresholds": DEFAULT_THRESHOLDS,
        "calibration": calibration,
        "metrics": {
            "train_rows": int(len(train_X)),
            "valid_rows": int(len(valid_X)),
            "raw_log_loss": _log_loss(y_valid.to_numpy(dtype=float), raw_valid),
            "calibrated_log_loss": _log_loss(y_valid.to_numpy(dtype=float), calibrated_valid),
            "raw_brier": _brier(y_valid.to_numpy(dtype=float), raw_valid),
            "calibrated_brier": _brier(y_valid.to_numpy(dtype=float), calibrated_valid),
            "positive_rate": float(np.mean(y.to_numpy(dtype=float))),
        },
        "training": {
            "input_path": args.input,
            "time_col": args.time_col,
            "label_col": args.label_col,
            "valid_fraction": args.valid_fraction,
            "num_round": args.num_round,
            "eta": args.eta,
            "max_depth": args.max_depth,
            "seed": args.seed,
            "bet_size": args.bet_size,
            "cost_per_share": args.cost_per_share,
        },
    }
    if backtest_metrics:
        metadata["backtest"] = backtest_metrics
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved feature schema: {feature_path}")
    print(f"Saved metadata: {meta_path}")
    print(
        "Validation metrics | "
        f"logloss={metadata['metrics']['calibrated_log_loss']:.5f} "
        f"brier={metadata['metrics']['calibrated_brier']:.5f} "
        f"positive_rate={metadata['metrics']['positive_rate']:.3f}"
    )
    if backtest_metrics:
        parts = [
            "Holdout backtest |",
            f"accuracy={backtest_metrics['holdout_accuracy']:.3f}",
            f"total_pnl={backtest_metrics['holdout_total_pnl']:+.2f}",
            f"pnl_per_trade={backtest_metrics['holdout_pnl_per_trade']:+.4f}",
            f"trades={backtest_metrics['holdout_trades']}",
        ]
        if "holdout_mean_realized_edge" in backtest_metrics:
            parts.append(f"mean_edge={backtest_metrics['holdout_mean_realized_edge']:+.4f}")
            parts.append(f"profitable_frac={backtest_metrics['holdout_profitable_frac']:.3f}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
