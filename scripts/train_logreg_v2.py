"""
Train logistic regression v2 using live orderbook + BTC 1s data.

Usage:
    python scripts/train_logreg_v2.py
    python scripts/train_logreg_v2.py --ob data/live_orderbook_snapshots.csv --btc data/btc_live_1s.csv
    python scripts/train_logreg_v2.py --valid-fraction 0.25 --version logreg_v2
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.logreg_model import LR_FEATURES, LogRegModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LogReg v2 from live data")
    p.add_argument("--ob", default="data/live_orderbook_snapshots.csv",
                   help="Orderbook snapshots CSV")
    p.add_argument("--btc", default="data/btc_live_1s.csv",
                   help="BTC 1-second OHLCV CSV")
    p.add_argument("--output-dir", default="models/logreg",
                   help="Directory for model artifacts")
    p.add_argument("--valid-fraction", type=float, default=0.2,
                   help="Holdout fraction (walk-forward by slot)")
    p.add_argument("--row-interval", type=int, default=15,
                   help="Seconds between decision rows per slot")
    p.add_argument("--version", default="logreg_v2",
                   help="Model version string")
    return p.parse_args()


def _derive_outcomes(btc_df: pd.DataFrame, slots: np.ndarray) -> dict:
    """For each 5-min slot, determine Up/Down from BTC close prices."""
    ts = btc_df["timestamp"].values
    close = btc_df["close"].values
    outcomes = {}
    for slot_ts in slots:
        start_mask = (ts >= slot_ts) & (ts <= slot_ts + 5)
        end_mask = (ts >= slot_ts + 295) & (ts <= slot_ts + 305)
        if not start_mask.any() or not end_mask.any():
            continue
        p_start = close[start_mask][0]
        p_end = close[end_mask][-1]
        outcomes[slot_ts] = 1 if p_end >= p_start else 0
    return outcomes


def _btc_return(ts, close, now, lookback):
    cur_idx = np.searchsorted(ts, now, side="right") - 1
    prev_idx = np.searchsorted(ts, now - lookback, side="right") - 1
    if cur_idx < 0 or prev_idx < 0:
        return 0.0
    prev = close[prev_idx]
    if prev <= 0:
        return 0.0
    return (close[cur_idx] - prev) / prev


def _btc_vol(ts, close, now, lookback):
    start = now - lookback
    mask = (ts >= start) & (ts <= now)
    prices = close[mask]
    if len(prices) < 2:
        return 0.0
    rets = np.diff(prices) / prices[:-1]
    return float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0


def _btc_rsi(ts, close, now, period):
    idx = int(np.searchsorted(ts, now, side="right"))
    need = period + 1
    if idx < need:
        return 50.0
    segment = close[idx - need:idx].astype(float)
    deltas = np.diff(segment)
    gains = float(np.sum(deltas[deltas > 0]))
    losses = float(-np.sum(deltas[deltas < 0]))
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _btc_ma_gap(ts, close, now, n):
    idx = int(np.searchsorted(ts, now, side="right"))
    if idx == 0:
        return 0.0
    start = max(0, idx - n)
    ma = float(np.mean(close[start:idx]))
    spot_idx = idx - 1
    spot = float(close[spot_idx])
    if ma <= 0:
        return 0.0
    return (spot - ma) / ma


def _asof_price(ts, close, target):
    idx = int(np.searchsorted(ts, target, side="right")) - 1
    if idx < 0:
        return None
    return float(close[idx])


def build_dataset(ob_df: pd.DataFrame, btc_df: pd.DataFrame,
                  outcomes: dict, row_interval: int) -> pd.DataFrame:
    """Build feature matrix from orderbook snapshots + BTC data."""
    btc_ts = btc_df["timestamp"].values.astype(float)
    btc_close = btc_df["close"].values.astype(float)

    # Pivot orderbook: for each (slot_ts, elapsed_s), get up and down rows
    up_ob = ob_df[ob_df["side"] == "up"].copy()
    down_ob = ob_df[ob_df["side"] == "down"].copy()

    rows = []
    for slot_ts, target_up in sorted(outcomes.items()):
        expiry_ts = slot_ts + 300

        # Get strike price (BTC price at slot start)
        strike = _asof_price(btc_ts, btc_close, float(slot_ts))
        if strike is None:
            continue

        # Get orderbook snapshots for this slot
        slot_up = up_ob[up_ob["slot_ts"] == slot_ts].sort_values("elapsed_s")
        slot_down = down_ob[down_ob["slot_ts"] == slot_ts].sort_values("elapsed_s")

        if slot_up.empty or slot_down.empty:
            continue

        # Build up_mid history for computing up_mid_ret_30s
        up_mid_history = list(zip(
            slot_up["elapsed_s"].values + slot_ts,
            slot_up["mid"].values,
        ))

        # Sample decision rows at row_interval
        elapsed_values = sorted(slot_up["elapsed_s"].unique())
        sampled = [e for e in elapsed_values if e % row_interval < 3 and e >= row_interval]
        if not sampled:
            sampled = elapsed_values[::max(1, len(elapsed_values) // 20)]

        for elapsed in sampled:
            t = slot_ts + elapsed
            tte = expiry_ts - t

            # BTC features
            spot = _asof_price(btc_ts, btc_close, float(t))
            if spot is None:
                continue

            ret_15s = _btc_return(btc_ts, btc_close, float(t), 15)
            ret_30s = _btc_return(btc_ts, btc_close, float(t), 30)
            ret_60s = _btc_return(btc_ts, btc_close, float(t), 60)
            vol_60s = _btc_vol(btc_ts, btc_close, float(t), 60)
            rsi_14 = _btc_rsi(btc_ts, btc_close, float(t), 14)
            dist_to_strike = (spot - strike) / strike if strike > 0 else 0.0
            ma_12_gap = _btc_ma_gap(btc_ts, btc_close, float(t), 12)

            # Orderbook features — find closest snapshot
            up_row = slot_up.iloc[(slot_up["elapsed_s"] - elapsed).abs().argmin()]
            down_row = slot_down.iloc[(slot_down["elapsed_s"] - elapsed).abs().argmin()]

            up_mid = float(up_row["mid"])
            up_spread = float(up_row["spread"])
            down_spread = float(down_row["spread"])

            y_bid_d = float(up_row["bid_depth_3"])
            y_ask_d = float(up_row["ask_depth_3"])
            y_total = y_bid_d + y_ask_d
            imb_up = y_bid_d / y_total if y_total > 0 else 0.5

            n_bid_d = float(down_row["bid_depth_3"])
            n_ask_d = float(down_row["ask_depth_3"])
            n_total = n_bid_d + n_ask_d
            imb_down = n_bid_d / n_total if n_total > 0 else 0.5

            if y_bid_d > 0 and n_bid_d > 0:
                depth_ratio = math.log(y_bid_d / n_bid_d)
            else:
                depth_ratio = 0.0

            # up_mid_ret_30s from history
            up_mid_ret = 0.0
            target_ts_30 = t - 30.0
            prev_price = None
            for h_ts, h_px in up_mid_history:
                if h_ts <= target_ts_30:
                    prev_price = h_px
                elif h_ts > target_ts_30:
                    break
            if prev_price and prev_price > 0:
                up_mid_ret = (up_mid - prev_price) / prev_price

            rows.append({
                "contract_id": slot_ts,
                "timestamp": t,
                "time_to_expiry_sec": tte,
                "target_up": target_up,
                "ret_15s": ret_15s,
                "ret_30s": ret_30s,
                "ret_60s": ret_60s,
                "rolling_vol_60s": vol_60s,
                "rsi_14": rsi_14,
                "dist_to_strike": dist_to_strike,
                "ma_12_gap": ma_12_gap,
                "up_mid": up_mid,
                "up_spread": up_spread,
                "down_spread": down_spread,
                "book_imbalance_up": imb_up,
                "book_imbalance_down": imb_down,
                "up_mid_ret_30s": up_mid_ret,
                "depth_ratio": depth_ratio,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["contract_id", "timestamp"]).reset_index(drop=True)


def main() -> None:
    args = _parse_args()

    print(f"Loading orderbook data from {args.ob}...")
    ob_df = pd.read_csv(args.ob)
    print(f"  {len(ob_df)} rows, {ob_df['slot_ts'].nunique()} slots")

    print(f"Loading BTC data from {args.btc}...")
    btc_df = pd.read_csv(args.btc)
    print(f"  {len(btc_df)} rows, {(btc_df['timestamp'].max() - btc_df['timestamp'].min())/3600:.1f} hours")

    # Derive outcomes
    slots = ob_df["slot_ts"].unique()
    outcomes = _derive_outcomes(btc_df, slots)
    n_up = sum(v for v in outcomes.values())
    n_down = len(outcomes) - n_up
    print(f"\nDerived outcomes for {len(outcomes)} slots (Up: {n_up}, Down: {n_down})")

    if len(outcomes) < 20:
        print("Too few slots with BTC coverage. Need at least 20.", file=sys.stderr)
        sys.exit(1)

    # Build dataset
    print(f"Building dataset (interval={args.row_interval}s)...")
    dataset = build_dataset(ob_df, btc_df, outcomes, args.row_interval)
    if dataset.empty:
        print("Dataset is empty.", file=sys.stderr)
        sys.exit(1)
    print(f"Dataset: {len(dataset)} rows, {dataset['contract_id'].nunique()} contracts")

    # Verify features
    missing = [f for f in LR_FEATURES if f not in dataset.columns]
    if missing:
        print(f"Missing features: {missing}", file=sys.stderr)
        sys.exit(1)

    X = dataset[LR_FEATURES].to_numpy(dtype=float)
    y = dataset["target_up"].to_numpy(dtype=int)

    # Walk-forward split by contract
    contracts = dataset["contract_id"].unique()
    split_idx = max(1, int(len(contracts) * (1.0 - args.valid_fraction)))
    train_contracts = set(contracts[:split_idx])
    valid_contracts = set(contracts[split_idx:])

    train_mask = dataset["contract_id"].isin(train_contracts)
    valid_mask = dataset["contract_id"].isin(valid_contracts)

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]

    print(f"Train: {len(X_train)} rows ({len(train_contracts)} contracts)")
    print(f"Valid: {len(X_valid)} rows ({len(valid_contracts)} contracts)")

    # Train
    print("\nTraining logistic regression...")
    model = LogRegModel.train(X_train, y_train, model_version=args.version)

    # Evaluate
    from sklearn.metrics import brier_score_loss, log_loss

    train_probs = model._model.predict_proba(model._scaler.transform(X_train))[:, 1]
    valid_probs = model._model.predict_proba(model._scaler.transform(X_valid))[:, 1]

    train_acc = float(np.mean((train_probs >= 0.5) == y_train))
    valid_acc = float(np.mean((valid_probs >= 0.5) == y_valid))
    valid_brier = brier_score_loss(y_valid, valid_probs)
    valid_logloss = log_loss(y_valid, valid_probs)

    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Valid accuracy: {valid_acc:.3f}")
    print(f"Valid Brier:    {valid_brier:.4f}")
    print(f"Valid LogLoss:  {valid_logloss:.4f}")
    print(f"Valid base rate (Up): {y_valid.mean():.3f}")
    print(f"Train base rate (Up): {y_train.mean():.3f}")

    # Feature coefficients
    coefs = model._model.coef_[0]
    intercept = model._model.intercept_[0]
    print(f"\nIntercept: {intercept:+.4f}")
    for feat, coef in zip(LR_FEATURES, coefs):
        print(f"  {feat:25s} {coef:+.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(args.output_dir)
    print(f"\nModel saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
