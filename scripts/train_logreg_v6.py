"""
Train LogReg v6 — 11 features after iterative pruning.

Initial 13-feature set (from v4/v5 coef analysis) was trained and two features
collapsed to noise in the joint fit due to collinearity with other retained
features: ret_60s (+0.316 in v5 → -0.020 in v6₁₃) and depth_ratio (+0.283 in
v5 → +0.013 in v6₁₃). Both dropped here.

Dropped as noise / redundant / collinear:
  rsi_14, ob_cross_imbalance, volume_surge_ratio, ob_imbalance/imbalance,
  ret_5s, ret_15s, ret_30s, vol_30s, vol_60s, vol_ratio_15_60,
  time_to_expiry, spread (execution gate only), depth_change,
  ret_60s (collinear with ret_180s + mid_return_5s),
  depth_ratio (collinear with depth_skew).

Kept:
  momentum/dynamics : ret_180s, mid_return_5s, acceleration, cvd_60s,
                      vwap_deviation
  volatility/range  : vol_15s, range_30s, rolling_std_30s
  orderbook         : depth_skew, wall_flag
  regime            : td_setup_net

Reuses build_dataset from train_and_compare_v5.py (unified schema).
Reports walk-forward metrics and a $100 Kelly backtest vs v4 and v5 on
the same held-out slots.

Usage:
    python scripts/train_logreg_v6.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_logreg_v3 as v3  # noqa: E402
import train_and_compare_v5 as v5  # noqa: E402


V6_FEATURES = [
    # Momentum / price dynamics
    "ret_180s",
    "mid_return_5s",
    "acceleration",
    "cvd_60s",
    "vwap_deviation",
    # Volatility / range
    "vol_15s",
    "range_30s",
    "rolling_std_30s",
    # Orderbook microstructure
    "depth_skew",
    "wall_flag",
    # Regime
    "td_setup_net",
]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss

    ob_path = os.path.join(REPO_ROOT, "data/live_orderbook_snapshots.csv")
    btc_path = os.path.join(REPO_ROOT, "data/btc_live_1s.csv")
    v4_dir = os.path.join(REPO_ROOT, "models/logreg_v4")
    v5_dir = os.path.join(REPO_ROOT, "models/logreg_v5")
    v6_dir = os.path.join(REPO_ROOT, "models/logreg_v6")
    os.makedirs(v6_dir, exist_ok=True)

    print("Loading data...")
    ob_df = pd.read_csv(ob_path)
    btc_df = pd.read_csv(btc_path)
    print(f"  OB: {len(ob_df)} rows, {ob_df['slot_ts'].nunique()} slots")
    print(f"  BTC: {len(btc_df)} rows")

    labels = v3.derive_labels(ob_df)
    n_up = sum(labels.values())
    print(f"Labels: {len(labels)} slots  UP={n_up}  rate={n_up / len(labels):.3f}")

    print("\nBuilding dataset (unified v5 schema)...")
    df = v5.build_dataset(ob_df, btc_df, labels, row_interval=15)
    if df.empty:
        print("ERROR: empty dataset", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(df)} rows from {df['slot_ts'].nunique()} slots")

    # Walk-forward 80/20 slot split — same as v5 script for fair comparison
    slots = sorted(df["slot_ts"].unique())
    split = max(1, int(len(slots) * 0.8))
    train_slots = set(slots[:split])
    valid_slots = set(slots[split:])
    train = df[df["slot_ts"].isin(train_slots)].reset_index(drop=True)
    valid = df[df["slot_ts"].isin(valid_slots)].reset_index(drop=True)
    print(f"Train: {len(train)} rows / {len(train_slots)} slots")
    print(f"Valid: {len(valid)} rows / {len(valid_slots)} slots")

    # ── Train v6 ─────────────────────────────────────────────────────────
    X_tr = train[V6_FEATURES].values.astype(float)
    y_tr = train["y"].values.astype(int)
    X_va = valid[V6_FEATURES].values.astype(float)
    y_va = valid["y"].values.astype(int)

    scaler6 = StandardScaler().fit(X_tr)
    Xs_tr = scaler6.transform(X_tr)
    Xs_va = scaler6.transform(X_va)

    model6 = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.1)
    model6.fit(Xs_tr, y_tr)

    p_va_raw6 = model6.predict_proba(Xs_va)[:, 1]
    cal6 = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    cal6.fit(p_va_raw6, y_va)
    p_va6 = cal6.predict(p_va_raw6)

    # ── Score v4 on the SAME valid rows (direct sigmoid, pickle-safe) ────
    with open(os.path.join(v4_dir, "logreg_scaler.pkl"), "rb") as f:
        scaler4 = pickle.load(f)
    with open(os.path.join(v4_dir, "logreg_calibrator.pkl"), "rb") as f:
        cal4 = pickle.load(f)
    meta4 = json.loads(open(os.path.join(v4_dir, "logreg_meta.json")).read())

    X_va4 = valid[v5.V4_FEATURES].values.astype(float)
    p_va_raw4 = v5.score_linear(
        X_va4, scaler4, np.asarray(meta4["coef"]), float(meta4["intercept"])
    )
    if hasattr(cal4, "predict"):
        p_va4 = cal4.predict(p_va_raw4)
    elif hasattr(cal4, "transform"):
        p_va4 = cal4.transform(p_va_raw4)
    else:
        p_va4 = np.asarray(cal4(p_va_raw4))

    # ── Score v5 on the SAME valid rows ──────────────────────────────────
    with open(os.path.join(v5_dir, "logreg_scaler.pkl"), "rb") as f:
        scaler5 = pickle.load(f)
    with open(os.path.join(v5_dir, "logreg_calibrator.pkl"), "rb") as f:
        cal5 = pickle.load(f)
    meta5 = json.loads(open(os.path.join(v5_dir, "logreg_meta.json")).read())

    X_va5 = valid[v5.V5_FEATURES].values.astype(float)
    p_va_raw5 = v5.score_linear(
        X_va5, scaler5, np.asarray(meta5["coef"]), float(meta5["intercept"])
    )
    if hasattr(cal5, "predict"):
        p_va5 = cal5.predict(p_va_raw5)
    elif hasattr(cal5, "transform"):
        p_va5 = cal5.transform(p_va_raw5)
    else:
        p_va5 = np.asarray(cal5(p_va_raw5))

    # ── Metrics ──────────────────────────────────────────────────────────
    def slot_acc(df_v, p):
        tmp = df_v.assign(p=p)
        s = tmp.groupby("slot_ts").agg(y=("y", "first"), pm=("p", "mean"))
        return float(((s["pm"] > 0.5).astype(int) == s["y"]).mean())

    def pack(p):
        return {
            "brier": float(brier_score_loss(y_va, p)),
            "logloss": float(log_loss(y_va, np.clip(p, 1e-6, 1 - 1e-6))),
            "row_acc": float(((p > 0.5).astype(int) == y_va).mean()),
            "slot_acc": slot_acc(valid, p),
        }

    metrics = {"v4": pack(p_va4), "v5": pack(p_va5), "v6": pack(p_va6)}

    print("\n" + "=" * 60)
    print(f"{'metric':<12} {'v4':>14} {'v5':>14} {'v6':>14}")
    for k in ("brier", "logloss", "row_acc", "slot_acc"):
        print(f"{k:<12} {metrics['v4'][k]:>14.4f} "
              f"{metrics['v5'][k]:>14.4f} {metrics['v6'][k]:>14.4f}")
    print("=" * 60)

    # ── Backtest ─────────────────────────────────────────────────────────
    print("\nBacktest on held-out slots ($100 start, edge>0.02, 0.15·Kelly, cap 10%)...")
    bt4 = v5.backtest(valid, p_va4)
    bt5 = v5.backtest(valid, p_va5)
    bt6 = v5.backtest(valid, p_va6)

    def summary(name, bt):
        trades = bt[bt["side"].notna()]
        n = len(trades)
        wr = float(trades["win"].mean()) if n else 0.0
        pnl = float(trades["pnl"].sum()) if n else 0.0
        final = float(bt["equity"].iloc[-1]) if not bt.empty else 100.0
        print(f"  {name}: trades={n}  win_rate={wr:.3f}  "
              f"gross_pnl={pnl:+.2f}  final_equity=${final:.2f}")
        return final, n, wr

    f4, n4, wr4 = summary("v4", bt4)
    f5, n5, wr5 = summary("v5", bt5)
    f6, n6, wr6 = summary("v6", bt6)

    # ── Coefficients ─────────────────────────────────────────────────────
    print("\nv6 coefficients (sorted by |coef|):")
    coefs = list(zip(V6_FEATURES, model6.coef_[0]))
    for feat, c in sorted(coefs, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:<22} {c:+.4f}")
    print(f"  {'intercept':<22} {float(model6.intercept_[0]):+.4f}")

    # ── Persist v6 ───────────────────────────────────────────────────────
    with open(os.path.join(v6_dir, "logreg_model.pkl"), "wb") as f:
        pickle.dump(model6, f)
    with open(os.path.join(v6_dir, "logreg_scaler.pkl"), "wb") as f:
        pickle.dump(scaler6, f)
    with open(os.path.join(v6_dir, "logreg_calibrator.pkl"), "wb") as f:
        pickle.dump(cal6, f)

    meta6 = {
        "model_version": "logreg_v6",
        "features": V6_FEATURES,
        "n_train_rows": int(len(X_tr)),
        "n_valid_rows": int(len(X_va)),
        "n_train_slots": int(len(train_slots)),
        "n_valid_slots": int(len(valid_slots)),
        "valid_brier": metrics["v6"]["brier"],
        "valid_logloss": metrics["v6"]["logloss"],
        "valid_row_accuracy": metrics["v6"]["row_acc"],
        "slot_accuracy": metrics["v6"]["slot_acc"],
        "coef": model6.coef_[0].tolist(),
        "intercept": float(model6.intercept_[0]),
        "calibrated": True,
        "backtest": {
            "starting_equity": 100.0,
            "edge_threshold": 0.02,
            "kelly_mult": 0.15,
            "max_frac": 0.10,
            "v4": {"final_equity": f4, "n_trades": n4, "win_rate": wr4},
            "v5": {"final_equity": f5, "n_trades": n5, "win_rate": wr5},
            "v6": {"final_equity": f6, "n_trades": n6, "win_rate": wr6},
        },
    }
    with open(os.path.join(v6_dir, "logreg_meta.json"), "w") as f:
        json.dump(meta6, f, indent=2)

    print(f"\nSaved v6 → {v6_dir}/")


if __name__ == "__main__":
    main()
