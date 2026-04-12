"""
Train the tuned logreg_v5 (round-2 Pareto trial #40) and backtest it
against logreg_v4 on the same held-out slots. Saves:
  - models/logreg_v5_tuned/ (pkls + meta + training data)
  - data/v4_vs_v5tuned_backtest.png (equity curves + metrics bars)

Chosen config (balanced Pareto point from experiments/v5/round_2):
  features (14): round 2 LLM-suggested subset
  C=0.1186  row_interval=25  edge_threshold=0.0490
  kelly_mult=0.425  max_frac=0.247

Usage:
  python scripts/finalize_tuned_v5.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_and_compare_v5 as v5mod  # noqa: E402

TUNED_FEATURES = [
    "ret_30s", "ret_60s",
    "depth_ratio", "imbalance", "spread",
    "wall_flag", "depth_change",
    "mid_return_5s", "acceleration",
    "rolling_std_30s", "time_to_expiry",
    "vol_ratio_15_60", "cvd_60s", "vwap_deviation",
]

TUNED_HPARAMS = {
    "C": 0.1186,
    "row_interval": 25,
    "edge_threshold": 0.0490,
    "kelly_mult": 0.425,
    "max_frac": 0.247,
}


def main():
    from sklearn.metrics import brier_score_loss, log_loss
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Loading data...")
    ob_df = pd.read_csv(os.path.join(REPO, "data/live_orderbook_snapshots.csv"))
    btc_df = pd.read_csv(os.path.join(REPO, "data/btc_live_1s.csv"))
    labels = v5mod.v3.derive_labels(ob_df)
    print(f"  OB: {len(ob_df)} rows, {ob_df['slot_ts'].nunique()} slots")
    print(f"  Labels: {len(labels)} slots")

    print(f"\nBuilding dataset (row_interval={TUNED_HPARAMS['row_interval']})...")
    df = v5mod.build_dataset(ob_df, btc_df, labels,
                             row_interval=TUNED_HPARAMS["row_interval"])
    print(f"  {len(df)} rows from {df['slot_ts'].nunique()} slots")

    slots = sorted(df["slot_ts"].unique())
    split = max(1, int(len(slots) * 0.8))
    train_slots = set(slots[:split])
    valid_slots = set(slots[split:])
    train = df[df["slot_ts"].isin(train_slots)].reset_index(drop=True)
    valid = df[df["slot_ts"].isin(valid_slots)].reset_index(drop=True)
    print(f"Train: {len(train)} rows / {len(train_slots)} slots")
    print(f"Valid: {len(valid)} rows / {len(valid_slots)} slots")

    print(f"\nTraining tuned v5 with {len(TUNED_FEATURES)} features...")
    result = v5mod.run_experiment(
        df, TUNED_FEATURES,
        {k: v for k, v in TUNED_HPARAMS.items() if k != "row_interval"},
    )
    model_t, scaler_t, cal_t = result["model"], result["scaler"], result["calibrator"]

    X_va = valid[TUNED_FEATURES].values.astype(float)
    y_va = valid["y"].values.astype(int)
    p_vaT = cal_t.predict(model_t.predict_proba(scaler_t.transform(X_va))[:, 1])

    p_va4 = v5mod.score_v4(valid, os.path.join(REPO, "models/logreg_v4"))

    def slot_acc(df_v, p):
        tmp = df_v.assign(p=p)
        s = tmp.groupby("slot_ts").agg(y=("y", "first"), pm=("p", "mean"))
        return float(((s["pm"] > 0.5).astype(int) == s["y"]).mean())

    def m(y, p):
        return {
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "row_acc": float(((p > 0.5).astype(int) == y).mean()),
        }

    metrics = {
        "v4":        {**m(y_va, p_va4), "slot_acc": slot_acc(valid, p_va4)},
        "v5_tuned":  {**m(y_va, p_vaT), "slot_acc": slot_acc(valid, p_vaT)},
    }
    print("\n" + "=" * 62)
    print(f"{'metric':<12}{'v4':>14}{'v5_tuned':>16}{'delta':>14}")
    for k in ("brier", "logloss", "row_acc", "slot_acc"):
        a, b = metrics["v4"][k], metrics["v5_tuned"][k]
        print(f"{k:<12}{a:>14.4f}{b:>16.4f}{(b - a):>+14.4f}")
    print("=" * 62)

    print("\nBacktesting (tuned sizing from round 2 #40)...")
    bt_v4 = v5mod.backtest(
        valid, p_va4,
        edge_threshold=TUNED_HPARAMS["edge_threshold"],
        kelly_mult=TUNED_HPARAMS["kelly_mult"],
        max_frac=TUNED_HPARAMS["max_frac"],
    )
    bt_v5 = v5mod.backtest(
        valid, p_vaT,
        edge_threshold=TUNED_HPARAMS["edge_threshold"],
        kelly_mult=TUNED_HPARAMS["kelly_mult"],
        max_frac=TUNED_HPARAMS["max_frac"],
    )

    def summary(name, bt):
        trades = bt[bt["side"].notna()]
        n = len(trades)
        wr = float(trades["win"].mean()) if n else 0.0
        pnl = float(trades["pnl"].sum()) if n else 0.0
        final = float(bt["equity"].iloc[-1]) if not bt.empty else 100.0
        print(f"  {name:<10} trades={n:3d}  win_rate={wr:.3f}  "
              f"gross_pnl={pnl:+8.2f}  final=${final:.2f}")
        return final, n, wr

    f4, n4, wr4 = summary("v4", bt_v4)
    f5, n5, wr5 = summary("v5_tuned", bt_v5)

    # Honest baseline: v4 under its original sizing, not the tuned sizing.
    bt_v4_baseline = v5mod.backtest(valid, p_va4)
    fb, nb, wrb = summary("v4 (baseline sizing)", bt_v4_baseline)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(np.arange(len(bt_v4)), bt_v4["equity"].values,
            label=f"v4 (tuned sizing) ${f4:.2f}", color="#1f77b4")
    ax.plot(np.arange(len(bt_v4_baseline)),
            bt_v4_baseline["equity"].values,
            label=f"v4 (baseline sizing) ${fb:.2f}",
            color="#1f77b4", ls=":", alpha=0.7)
    ax.plot(np.arange(len(bt_v5)), bt_v5["equity"].values,
            label=f"v5_tuned ${f5:.2f}", color="#d62728", lw=2)
    ax.axhline(100, color="k", lw=0.5, ls="--")
    ax.set_xlabel("held-out slot #")
    ax.set_ylabel("equity ($)")
    ax.set_title("Equity curves — $100 start, round-2 #40 sizing")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    labels_ = ["brier ↓", "logloss ↓", "row_acc ↑", "slot_acc ↑"]
    keys = ["brier", "logloss", "row_acc", "slot_acc"]
    x = np.arange(len(labels_))
    w = 0.35
    ax.bar(x - w / 2, [metrics["v4"][k] for k in keys], w,
           label="v4", color="#1f77b4")
    ax.bar(x + w / 2, [metrics["v5_tuned"][k] for k in keys], w,
           label="v5_tuned", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_)
    ax.set_title("Held-out metrics")
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(REPO, "data/v4_vs_v5tuned_backtest.png")
    plt.savefig(out_png, dpi=120)
    print(f"\nSaved plot → {out_png}")

    out_dir = os.path.join(REPO, "models/logreg_v5_tuned")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "logreg_model.pkl"), "wb") as f:
        pickle.dump(model_t, f)
    with open(os.path.join(out_dir, "logreg_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_t, f)
    with open(os.path.join(out_dir, "logreg_calibrator.pkl"), "wb") as f:
        pickle.dump(cal_t, f)

    meta = {
        "model_version": "logreg_v5_tuned",
        "source_study": "experiments/v5/round_2",
        "source_trial": 40,
        "features": TUNED_FEATURES,
        "hparams": TUNED_HPARAMS,
        "n_train_rows": int(len(train)),
        "n_valid_rows": int(len(valid)),
        "n_train_slots": int(len(train_slots)),
        "n_valid_slots": int(len(valid_slots)),
        "coef": model_t.coef_[0].tolist(),
        "intercept": float(model_t.intercept_[0]),
        "calibrated": True,
        "held_out_metrics": metrics,
        "backtest": {
            "starting_equity": 100.0,
            **TUNED_HPARAMS,
            "v4_tuned_sizing":    {"final_equity": f4, "trades": n4, "win_rate": wr4},
            "v4_baseline_sizing": {"final_equity": fb, "trades": nb, "win_rate": wrb},
            "v5_tuned":           {"final_equity": f5, "trades": n5, "win_rate": wr5},
        },
    }
    with open(os.path.join(out_dir, "logreg_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    df.to_csv(os.path.join(out_dir, "training_data.csv"), index=False)

    print(f"Saved tuned v5 → {out_dir}/")


if __name__ == "__main__":
    main()
