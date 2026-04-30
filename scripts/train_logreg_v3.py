"""
Train logistic regression v3 using new BTC feature set.

Labels derived from orderbook settlement:
  y=1 if UP mid reaches >= 0.90 near slot end (YES resolves)
  y=0 if UP mid <= 0.10 near slot end (DOWN resolves)

Features (BTC-only from btc_live_1s.csv):
  delta, time_to_expiry, momentum (5s/15s/30s/60s/180s),
  vol (15s/30s/60s), vol_ratio_15_60,
  volume_surge_ratio, vwap_deviation, cvd_60s,
  rsi_14, td_setup_net

Usage:
    python scripts/train_logreg_v3.py
    python scripts/train_logreg_v3.py --row-interval 10
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import live feature helpers so train & serve use one source of truth.
from src.models.logreg_v4_model import (  # noqa: E402
    _microprice_delta,
    _imbalance_mean,
    _imbalance_slope,
    _ratio_change,
)

# ── Feature schema ────────────────────────────────────────────────────────

FEATURES = [
    # NOTE: "delta" (spot-strike)/strike removed — it dominates the model
    # (coef +0.9147) and duplicates info already in market mid price q_t,
    # making p_hat ≈ q_t and edge ≈ 0.
    "time_to_expiry",         # seconds remaining in slot
    # Momentum at multiple horizons
    "ret_5s",
    "ret_15s",
    "ret_30s",
    "ret_60s",
    "ret_180s",
    # Realized volatility
    "vol_15s",
    "vol_30s",
    "vol_60s",
    "vol_ratio_15_60",        # regime-change detector
    # Volume (tick-count proxy)
    "volume_surge_ratio",     # 15s / 60s tick intensity
    "vwap_deviation",         # (close - VWAP_60s) / close
    "cvd_60s",                # cumulative volume delta (normalized)
    # Classic indicators
    "rsi_14",
    "td_setup_net",           # bullish_count - bearish_count from TD Sequential
    # Orderbook context
    "spread",                 # up_ask - up_bid from orderbook
    "ob_imbalance",           # (bid_depth - ask_depth) / (bid + ask) for UP side
    "ob_cross_imbalance",     # UP bid_depth / (UP bid_depth + DOWN bid_depth)
    # v5: rolling microstructure features (top-3 depth aggregates)
    "microprice_delta",             # depth-weighted microprice - mid
    "imbalance_mean_10s",           # mean(ob_imbalance) over last 10s
    "imbalance_slope_10s",          # OLS slope of ob_imbalance over 10s
    "bid_ask_size_ratio_change_5s", # Δ(bid_d/ask_d) vs 5s-ago snapshot
]


# ── Label derivation ─────────────────────────────────────────────────────

def derive_labels(ob_df: pd.DataFrame) -> dict[int, int]:
    """y=1 if UP resolved, y=0 if DOWN resolved. Skip ambiguous."""
    labels = {}
    for slot_ts, grp in ob_df[ob_df["side"] == "up"].groupby("slot_ts"):
        late = grp[grp["elapsed_s"] >= 250]
        if late.empty:
            late = grp
        nonzero = late[late["mid"] > 0.01]
        if nonzero.empty:
            labels[int(slot_ts)] = 0  # mid collapsed → DOWN
            continue
        last_mid = nonzero.iloc[-1]["mid"]
        if last_mid >= 0.90:
            labels[int(slot_ts)] = 1
        elif last_mid <= 0.10:
            labels[int(slot_ts)] = 0
        # else: ambiguous, skip
    return labels


# ── BTC feature helpers (numpy-vectorized) ────────────────────────────────

def _asof(ts: np.ndarray, target: float) -> int:
    return int(np.searchsorted(ts, target, side="right")) - 1


def _ret(ts, close, now, lookback):
    ci = _asof(ts, now)
    pi = _asof(ts, now - lookback)
    if ci < 0 or pi < 0 or close[pi] <= 0:
        return 0.0
    return (close[ci] - close[pi]) / close[pi]


def _vol(ts, close, now, lookback):
    mask = (ts >= now - lookback) & (ts <= now)
    px = close[mask]
    if len(px) < 3:
        return 0.0
    r = np.diff(np.log(px))
    return float(np.std(r, ddof=1))


def _rsi(ts, close, now, period=14):
    idx = int(np.searchsorted(ts, now, side="right"))
    need = period + 1
    if idx < need:
        return 50.0
    seg = close[idx - need:idx].astype(float)
    d = np.diff(seg)
    g = float(np.sum(d[d > 0])) / period
    l = float(-np.sum(d[d < 0])) / period
    if l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + g / l)


def _td_setup_net(ts, close, now, lookback=4):
    """Simplified TD Sequential: net setup count over recent bars.

    Counts consecutive closes where close > close[lookback] (bearish setup, +1)
    minus consecutive close < close[lookback] (bullish setup, -1).
    Returns a value in roughly [-9, +9].
    """
    idx = int(np.searchsorted(ts, now, side="right"))
    if idx < lookback + 1:
        return 0.0
    # Use last 9+lookback bars
    n = min(idx, 13 + lookback)
    seg = close[idx - n:idx].astype(float)
    bull = 0
    bear = 0
    for i in range(lookback, len(seg)):
        if seg[i] < seg[i - lookback]:
            bull += 1
            bear = 0
        elif seg[i] > seg[i - lookback]:
            bear += 1
            bull = 0
        else:
            bull = 0
            bear = 0
    return float(bear - bull)  # positive = bearish pressure


def _volume_features(ts, vol, now):
    """Compute volume_surge_ratio, vwap_deviation, cvd from tick-count volume."""
    mask_15 = (ts >= now - 15) & (ts <= now)
    mask_60 = (ts >= now - 60) & (ts <= now)

    v15 = vol[mask_15]
    v60 = vol[mask_60]

    # Surge ratio
    avg_15 = v15.mean() if len(v15) > 0 else 0.0
    avg_60 = v60.mean() if len(v60) > 0 else 0.0
    surge = (avg_15 / avg_60) if avg_60 > 0 else 0.0

    return surge


def _vwap_deviation(ts, close, vol, now, lookback=60):
    mask = (ts >= now - lookback) & (ts <= now)
    px = close[mask]
    v = vol[mask]
    if len(px) < 2 or v.sum() == 0:
        return 0.0
    vwap = np.sum(px * v) / np.sum(v)
    spot = px[-1]
    return (spot - vwap) / spot if spot > 0 else 0.0


def _cvd(ts, close, vol, now, lookback=60):
    mask = (ts >= now - lookback) & (ts <= now)
    px = close[mask]
    v = vol[mask]
    if len(px) < 2:
        return 0.0
    diffs = np.diff(px)
    buy = np.sum(v[1:][diffs >= 0])
    sell = np.sum(v[1:][diffs < 0])
    total = buy + sell
    return (buy - sell) / total if total > 0 else 0.0


# ── Dataset builder ───────────────────────────────────────────────────────

def build_dataset(
    ob_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    labels: dict[int, int],
    row_interval: int = 15,
) -> pd.DataFrame:
    ts = btc_df["timestamp"].values.astype(float)
    close = btc_df["close"].values.astype(float)
    vol = btc_df["volume"].values.astype(float)

    up_ob = ob_df[ob_df["side"] == "up"]
    down_ob = ob_df[ob_df["side"] == "down"]

    rows = []
    for slot_ts, y in sorted(labels.items()):
        expiry = slot_ts + 300

        # Strike = BTC price at slot start
        si = _asof(ts, float(slot_ts))
        if si < 0:
            continue
        strike = close[si]
        if strike <= 0:
            continue

        # Get orderbook snapshots for spread + imbalance features
        slot_up = up_ob[up_ob["slot_ts"] == slot_ts].sort_values("elapsed_s")
        slot_down = down_ob[down_ob["slot_ts"] == slot_ts].sort_values("elapsed_s")
        if slot_up.empty:
            continue

        # Sample decision rows
        elapsed_vals = sorted(slot_up["elapsed_s"].unique())
        sampled = [e for e in elapsed_vals
                   if e >= row_interval and e % row_interval < 3 and e < 280]
        if not sampled:
            sampled = [e for e in elapsed_vals if row_interval <= e < 280]
            sampled = sampled[::max(1, len(sampled) // 15)]

        # Pre-build the per-slot OB history in wall-clock time for v5 helpers.
        # Columns: (wall_ts, bid_depth_3, ask_depth_3)
        slot_ob_hist: list[tuple[float, float, float]] = [
            (float(slot_ts) + float(r["elapsed_s"]),
             float(r.get("bid_depth_3", 0) or 0.0),
             float(r.get("ask_depth_3", 0) or 0.0))
            for _, r in slot_up.iterrows()
        ]

        for elapsed in sampled:
            t = float(slot_ts + elapsed)
            tte = expiry - t

            spot_idx = _asof(ts, t)
            if spot_idx < 0:
                continue
            spot = close[spot_idx]
            if spot <= 0:
                continue

            # Closest OB snapshot for spread + imbalance
            up_row = slot_up.iloc[(slot_up["elapsed_s"] - elapsed).abs().argmin()]
            spread = float(up_row["spread"])
            up_best_bid = float(up_row.get("best_bid", 0) or 0.0)
            up_best_ask = float(up_row.get("best_ask", 0) or 0.0)

            # Orderbook imbalance features
            up_bid_d = float(up_row.get("bid_depth_3", 0))
            up_ask_d = float(up_row.get("ask_depth_3", 0))
            ob_total = up_bid_d + up_ask_d
            ob_imbalance = (up_bid_d - up_ask_d) / ob_total if ob_total > 0 else 0.0

            # v5 rolling microstructure features (causal: history up to and
            # including t, no look-ahead into future snapshots within the slot)
            ob_hist_causal = [h for h in slot_ob_hist if h[0] <= t]
            microprice_delta = _microprice_delta(up_best_bid, up_best_ask, up_bid_d, up_ask_d)
            imbalance_mean_10s = _imbalance_mean(ob_hist_causal, t, 10.0)
            imbalance_slope_10s = _imbalance_slope(ob_hist_causal, t, 10.0)
            bid_ask_size_ratio_change_5s = _ratio_change(ob_hist_causal, t, 5.0)

            # Cross-side imbalance: UP bid depth vs DOWN bid depth
            ob_cross_imbalance = 0.0
            if not slot_down.empty:
                dn_row = slot_down.iloc[(slot_down["elapsed_s"] - elapsed).abs().argmin()]
                dn_bid_d = float(dn_row.get("bid_depth_3", 0))
                cross_total = up_bid_d + dn_bid_d
                ob_cross_imbalance = up_bid_d / cross_total if cross_total > 0 else 0.5

            v15 = _vol(ts, close, t, 15)
            v30 = _vol(ts, close, t, 30)
            v60 = _vol(ts, close, t, 60)

            rows.append({
                "slot_ts": slot_ts,
                "t": t,
                "y": y,
                "time_to_expiry": tte,
                "ret_5s": _ret(ts, close, t, 5),
                "ret_15s": _ret(ts, close, t, 15),
                "ret_30s": _ret(ts, close, t, 30),
                "ret_60s": _ret(ts, close, t, 60),
                "ret_180s": _ret(ts, close, t, 180),
                "vol_15s": v15,
                "vol_30s": v30,
                "vol_60s": v60,
                "vol_ratio_15_60": (v15 / v60) if v60 > 0 else 0.0,
                "volume_surge_ratio": _volume_features(ts, vol, t),
                "vwap_deviation": _vwap_deviation(ts, close, vol, t),
                "cvd_60s": _cvd(ts, close, vol, t),
                "rsi_14": _rsi(ts, close, t),
                "td_setup_net": _td_setup_net(ts, close, t),
                "spread": spread,
                "ob_imbalance": ob_imbalance,
                "ob_cross_imbalance": ob_cross_imbalance,
                "microprice_delta": microprice_delta,
                "imbalance_mean_10s": imbalance_mean_10s,
                "imbalance_slope_10s": imbalance_slope_10s,
                "bid_ask_size_ratio_change_5s": bid_ask_size_ratio_change_5s,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["slot_ts", "t"]).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train LogReg v3 (BTC features)")
    p.add_argument("--ob", default="data/live_orderbook_snapshots.csv")
    p.add_argument("--btc", default="data/btc_live_1s.csv")
    p.add_argument("--output-dir", default="models/logreg_v5")
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--row-interval", type=int, default=15)
    args = p.parse_args()

    # Load data
    print("Loading data...")
    ob_df = pd.read_csv(args.ob)
    btc_df = pd.read_csv(args.btc)
    print(f"  OB: {len(ob_df)} rows, {ob_df['slot_ts'].nunique()} slots")
    print(f"  BTC: {len(btc_df)} rows, "
          f"{(btc_df['timestamp'].max() - btc_df['timestamp'].min()) / 3600:.1f}h")

    # Labels from orderbook settlement
    labels = derive_labels(ob_df)
    n_up = sum(labels.values())
    n_dn = len(labels) - n_up
    print(f"\nLabels: {len(labels)} slots (UP={n_up}, DOWN={n_dn}, "
          f"rate={n_up / len(labels):.1%})")

    # Build dataset
    print(f"\nBuilding dataset (interval={args.row_interval}s)...")
    df = build_dataset(ob_df, btc_df, labels, args.row_interval)
    if df.empty:
        print("ERROR: empty dataset", file=sys.stderr)
        sys.exit(1)

    n_slots = df["slot_ts"].nunique()
    print(f"  {len(df)} rows from {n_slots} slots "
          f"({len(df) / n_slots:.0f} rows/slot avg)")

    # Walk-forward split
    slots = df["slot_ts"].unique()
    split = max(1, int(len(slots) * (1.0 - args.valid_fraction)))
    train_slots = set(slots[:split])
    valid_slots = set(slots[split:])

    train = df[df["slot_ts"].isin(train_slots)]
    valid = df[df["slot_ts"].isin(valid_slots)]

    X_train = train[FEATURES].values.astype(float)
    y_train = train["y"].values.astype(int)
    X_valid = valid[FEATURES].values.astype(float)
    y_valid = valid["y"].values.astype(int)

    print(f"\nTrain: {len(X_train)} rows ({len(train_slots)} slots, "
          f"base_rate={y_train.mean():.3f})")
    print(f"Valid: {len(X_valid)} rows ({len(valid_slots)} slots, "
          f"base_rate={y_valid.mean():.3f})")

    # Train
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import brier_score_loss, log_loss

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_valid)

    model = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.1)
    model.fit(X_tr_s, y_train)

    # Post-hoc isotonic calibration on validation set
    from sklearn.isotonic import IsotonicRegression
    p_valid_raw = model.predict_proba(X_va_s)[:, 1]
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(p_valid_raw, y_valid)

    # Evaluate (compare raw vs calibrated)
    p_train = model.predict_proba(X_tr_s)[:, 1]
    p_valid = calibrator.predict(p_valid_raw)

    train_acc = float(np.mean((p_train >= 0.5) == y_train))
    valid_acc = float(np.mean((p_valid >= 0.5) == y_valid))
    valid_brier = brier_score_loss(y_valid, p_valid)
    valid_ll = log_loss(y_valid, p_valid)

    valid_brier_raw = brier_score_loss(y_valid, p_valid_raw)

    print(f"\n{'='*50}")
    print(f"Train acc:        {train_acc:.3f}")
    print(f"Valid acc:        {valid_acc:.3f}")
    print(f"Valid Brier (raw): {valid_brier_raw:.4f}")
    print(f"Valid Brier (cal): {valid_brier:.4f}")
    print(f"Valid LL:         {valid_ll:.4f}")
    print(f"{'='*50}")

    # Per-slot accuracy (majority vote per slot)
    valid_with_pred = valid.copy()
    valid_with_pred["p_up"] = p_valid
    slot_acc_rows = []
    for slot_ts, grp in valid_with_pred.groupby("slot_ts"):
        pred_up = (grp["p_up"].mean() >= 0.5)
        actual_up = grp["y"].iloc[0] == 1
        slot_acc_rows.append(pred_up == actual_up)
    slot_acc = np.mean(slot_acc_rows) if slot_acc_rows else 0.0
    print(f"Slot-level acc: {slot_acc:.3f} ({len(slot_acc_rows)} slots)")

    # Calibration buckets
    print("\nCalibration (valid):")
    for lo, hi in [(0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.0)]:
        mask = (p_valid >= lo) & (p_valid < hi)
        if mask.sum() > 0:
            actual = y_valid[mask].mean()
            pred = p_valid[mask].mean()
            print(f"  [{lo:.2f},{hi:.2f}): n={mask.sum():4d}  "
                  f"pred={pred:.3f}  actual={actual:.3f}")

    # Coefficients
    print("\nCoefficients:")
    print(f"  {'intercept':25s} {model.intercept_[0]:+.4f}")
    for feat, coef in sorted(zip(FEATURES, model.coef_[0]),
                              key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:25s} {coef:+.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "logreg_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.output_dir, "logreg_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.output_dir, "logreg_calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
    meta = {
        "model_version": "logreg_v5",
        "features": FEATURES,
        "n_train_rows": len(X_train),
        "n_train_slots": len(train_slots),
        "n_valid_rows": len(X_valid),
        "n_valid_slots": len(valid_slots),
        "valid_accuracy": valid_acc,
        "valid_brier": valid_brier,
        "valid_brier_raw": valid_brier_raw,
        "slot_accuracy": float(slot_acc),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "calibrated": True,
    }
    with open(os.path.join(args.output_dir, "logreg_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save dataset for analysis
    df.to_csv(os.path.join(args.output_dir, "training_data.csv"), index=False)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
