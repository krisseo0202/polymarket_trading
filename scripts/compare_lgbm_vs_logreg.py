#!/usr/bin/env python3
"""
Compare LightGBM vs Logistic Regression on the same features and orderbook backtest.

Trains both models on the v4 feature set (no delta, +OB imbalance), runs an
edge-based backtest using actual Polymarket orderbook fill prices, and reports
side-by-side metrics: Brier score, slot accuracy, total PnL, win rate, drawdown.

Usage:
    python scripts/compare_lgbm_vs_logreg.py
    python scripts/compare_lgbm_vs_logreg.py --delta 0.05 --bet-size 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the v4 dataset builder + label derivation from the training script
from scripts.train_logreg_v3 import (
    FEATURES,
    derive_labels,
    build_dataset,
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import lightgbm as lgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightGBM vs LogReg comparison")
    p.add_argument("--ob", default="data/live_orderbook_snapshots.csv")
    p.add_argument("--btc", default="data/btc_live_1s.csv")
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--row-interval", type=int, default=15)
    p.add_argument("--delta", type=float, default=0.05,
                   help="Min edge to trade (matches strategy default)")
    p.add_argument("--bet-size", type=float, default=30.0,
                   help="USDC per trade (matches position_size_usdc)")
    p.add_argument("--min-entry-price", type=float, default=0.35)
    p.add_argument("--max-entry-price", type=float, default=0.65)
    return p.parse_args()


def merge_with_orderbook(df: pd.DataFrame, ob_df: pd.DataFrame) -> pd.DataFrame:
    """Attach actual yes/no bid/ask at each decision time from orderbook snapshots."""
    ob_up = ob_df[ob_df["side"] == "up"].copy()
    ob_dn = ob_df[ob_df["side"] == "down"].copy()
    ob_up["ob_t"] = (ob_up["slot_ts"] + ob_up["elapsed_s"]).astype(float)
    ob_dn["ob_t"] = (ob_dn["slot_ts"] + ob_dn["elapsed_s"]).astype(float)
    df = df.copy()
    df["t"] = df["t"].astype(float)

    ob_up = ob_up[["slot_ts", "ob_t", "best_bid", "best_ask"]].rename(
        columns={"best_bid": "yes_bid", "best_ask": "yes_ask"}
    )
    ob_dn = ob_dn[["slot_ts", "ob_t", "best_bid", "best_ask"]].rename(
        columns={"best_bid": "no_bid", "best_ask": "no_ask"}
    )

    df = df.sort_values(["slot_ts", "t"]).reset_index(drop=True)
    parts = []
    for slot_ts, grp in df.groupby("slot_ts"):
        u = ob_up[ob_up["slot_ts"] == slot_ts].sort_values("ob_t")
        d = ob_dn[ob_dn["slot_ts"] == slot_ts].sort_values("ob_t")
        if u.empty or d.empty:
            continue
        merged = pd.merge_asof(
            grp.sort_values("t"),
            u.drop(columns=["slot_ts"]),
            left_on="t", right_on="ob_t", direction="backward",
        )
        merged = pd.merge_asof(
            merged.sort_values("t"),
            d.drop(columns=["slot_ts"]),
            left_on="t", right_on="ob_t", direction="backward",
            suffixes=("_up", "_dn"),
        )
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["yes_bid", "yes_ask", "no_bid", "no_ask"])
    return out.reset_index(drop=True)


def edge_backtest(
    test_df: pd.DataFrame,
    p_hat: np.ndarray,
    delta: float,
    bet_size: float,
    min_entry_price: float,
    max_entry_price: float,
    label: str,
) -> dict:
    """For each test contract, find the best-edge entry and simulate the trade.

    Mirrors the logreg_edge strategy logic, but uses actual orderbook fill prices.
    Settlement: winning side pays $0.99, losing side pays $0.01.
    """
    test_df = test_df.copy()
    test_df["p_hat"] = p_hat
    test_df["q_t"] = (test_df["yes_bid"] + test_df["yes_ask"]) / 2.0
    test_df["c_t"] = (test_df["yes_ask"] - test_df["yes_bid"]) / 2.0
    test_df["edge_yes"] = test_df["p_hat"] - test_df["q_t"] - test_df["c_t"]
    test_df["edge_no"] = test_df["q_t"] - test_df["p_hat"] - test_df["c_t"]

    trades = []
    skipped = 0
    for slot_ts, grp in test_df.groupby("slot_ts"):
        # Mirror live strategy: pick the FIRST signal that exceeds delta,
        # not the best in window (which would be hindsight optimization).
        chosen = None
        for _, row in grp.iterrows():
            ey, en = row["edge_yes"], row["edge_no"]
            if ey >= en and ey >= delta:
                ask = row["yes_ask"]
                if min_entry_price <= ask <= max_entry_price:
                    chosen = ("YES", row, ey, ask)
                    break
            elif en > ey and en >= delta:
                ask = row["no_ask"]
                if min_entry_price <= ask <= max_entry_price:
                    chosen = ("NO", row, en, ask)
                    break

        if chosen is None:
            skipped += 1
            continue

        side, row, edge, entry_price = chosen
        outcome_up = int(row["y"]) == 1

        # Settlement: winner = 0.99, loser = 0.01
        if side == "YES":
            settlement = 0.99 if outcome_up else 0.01
        else:
            settlement = 0.99 if not outcome_up else 0.01

        shares = bet_size / entry_price
        pnl = shares * (settlement - entry_price)
        trades.append({
            "slot_ts": int(slot_ts),
            "side": side,
            "edge": edge,
            "entry_price": entry_price,
            "settlement": settlement,
            "pnl": pnl,
            "win": pnl > 0,
        })

    n_trades = len(trades)
    if n_trades == 0:
        return {
            "label": label, "trades": 0, "pnl": 0.0, "win_rate": 0.0,
            "avg_pnl": 0.0, "max_dd": 0.0, "skipped": skipped,
        }

    pnl_arr = np.array([t["pnl"] for t in trades])
    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak).min()
    wins = sum(1 for t in trades if t["win"])
    return {
        "label": label,
        "trades": n_trades,
        "skipped": skipped,
        "pnl": float(pnl_arr.sum()),
        "avg_pnl": float(pnl_arr.mean()),
        "win_rate": wins / n_trades,
        "max_dd": float(dd),
        "yes_trades": sum(1 for t in trades if t["side"] == "YES"),
        "no_trades": sum(1 for t in trades if t["side"] == "NO"),
    }


def main():
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data…")
    ob_df = pd.read_csv(args.ob)
    btc_df = pd.read_csv(args.btc)
    print(f"  OB: {len(ob_df)} rows, {ob_df['slot_ts'].nunique()} slots")
    print(f"  BTC: {len(btc_df)} rows")

    labels = derive_labels(ob_df)
    print(f"  Labels: {len(labels)} slots ({sum(labels.values())} Up)")

    # ── Build features (v4 set, no delta, with OB imbalance) ──────────────
    print(f"\nBuilding feature dataset (interval={args.row_interval}s)…")
    df = build_dataset(ob_df, btc_df, labels, args.row_interval)
    print(f"  {len(df)} rows from {df['slot_ts'].nunique()} slots")

    # ── Attach orderbook fill prices ──────────────────────────────────────
    print("\nMerging with actual orderbook fill prices…")
    df = merge_with_orderbook(df, ob_df)
    print(f"  {len(df)} rows after orderbook merge")

    # ── Walk-forward split (chronological by slot) ────────────────────────
    slots = sorted(df["slot_ts"].unique())
    split = int(len(slots) * (1.0 - args.valid_fraction))
    train_slots = set(slots[:split])
    valid_slots = set(slots[split:])
    train = df[df["slot_ts"].isin(train_slots)]
    valid = df[df["slot_ts"].isin(valid_slots)]
    print(f"\nTrain: {len(train)} rows / {len(train_slots)} slots")
    print(f"Valid: {len(valid)} rows / {len(valid_slots)} slots")

    X_tr = train[FEATURES].values.astype(float)
    y_tr = train["y"].values.astype(int)
    X_va = valid[FEATURES].values.astype(float)
    y_va = valid["y"].values.astype(int)

    # ── Train Logistic Regression (v4 setup) ──────────────────────────────
    print("\n── Training Logistic Regression ──")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.1)
    lr.fit(X_tr_s, y_tr)
    p_lr_raw = lr.predict_proba(X_va_s)[:, 1]
    cal = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    cal.fit(p_lr_raw, y_va)
    p_lr = cal.predict(p_lr_raw)

    lr_brier = brier_score_loss(y_va, p_lr)
    lr_acc = float(np.mean((p_lr >= 0.5) == y_va))

    # ── Train LightGBM ────────────────────────────────────────────────────
    print("\n── Training LightGBM ──")
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,            # small tree to avoid overfit
        "max_depth": 5,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    booster = lgb.train(
        params, lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    p_lgb_raw = booster.predict(X_va, num_iteration=booster.best_iteration)
    cal_lgb = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    cal_lgb.fit(p_lgb_raw, y_va)
    p_lgb = cal_lgb.predict(p_lgb_raw)

    lgb_brier = brier_score_loss(y_va, p_lgb)
    lgb_acc = float(np.mean((p_lgb >= 0.5) == y_va))

    # ── Slot-level accuracy (majority vote per slot) ──────────────────────
    def slot_acc(probs):
        v = valid.copy()
        v["p"] = probs
        correct = 0
        total = 0
        for _, grp in v.groupby("slot_ts"):
            pred_up = grp["p"].mean() >= 0.5
            actual_up = grp["y"].iloc[0] == 1
            correct += int(pred_up == actual_up)
            total += 1
        return correct / total if total else 0.0

    lr_slot = slot_acc(p_lr)
    lgb_slot = slot_acc(p_lgb)

    # ── Run backtest with actual orderbook fills ──────────────────────────
    print("\n── Running edge-based backtest with actual orderbook fills ──")
    print(f"   delta={args.delta}, bet_size=${args.bet_size}, "
          f"price range=[{args.min_entry_price},{args.max_entry_price}]")
    bt_lr = edge_backtest(
        valid, p_lr, args.delta, args.bet_size,
        args.min_entry_price, args.max_entry_price, "LogReg",
    )
    bt_lgb = edge_backtest(
        valid, p_lgb, args.delta, args.bet_size,
        args.min_entry_price, args.max_entry_price, "LightGBM",
    )

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Metric':<28s}  {'LogReg (v4)':>16s}  {'LightGBM':>16s}")
    print("=" * 70)
    print(f"  {'Valid Brier (calibrated)':<28s}  {lr_brier:>16.4f}  {lgb_brier:>16.4f}")
    print(f"  {'Valid row accuracy':<28s}  {lr_acc:>16.1%}  {lgb_acc:>16.1%}")
    print(f"  {'Valid slot accuracy':<28s}  {lr_slot:>16.1%}  {lgb_slot:>16.1%}")
    print("-" * 70)
    print(f"  {'Backtest trades':<28s}  {bt_lr['trades']:>16d}  {bt_lgb['trades']:>16d}")
    print(f"  {'Backtest skipped slots':<28s}  {bt_lr['skipped']:>16d}  {bt_lgb['skipped']:>16d}")
    print(f"  {'Total PnL ($)':<28s}  {bt_lr['pnl']:>+16.2f}  {bt_lgb['pnl']:>+16.2f}")
    print(f"  {'Avg PnL/trade ($)':<28s}  {bt_lr['avg_pnl']:>+16.2f}  {bt_lgb['avg_pnl']:>+16.2f}")
    print(f"  {'Win rate':<28s}  {bt_lr['win_rate']:>16.1%}  {bt_lgb['win_rate']:>16.1%}")
    print(f"  {'Max drawdown ($)':<28s}  {bt_lr['max_dd']:>+16.2f}  {bt_lgb['max_dd']:>+16.2f}")
    lr_yn = f"{bt_lr['yes_trades']}/{bt_lr['no_trades']}"
    lgb_yn = f"{bt_lgb['yes_trades']}/{bt_lgb['no_trades']}"
    print(f"  {'YES / NO trades':<28s}  {lr_yn:>16s}  {lgb_yn:>16s}")
    print("=" * 70)

    # LightGBM feature importance
    print("\n  LightGBM feature importance (gain):")
    importances = booster.feature_importance(importance_type="gain")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True):
        print(f"    {feat:25s}  {imp:>10.1f}")


if __name__ == "__main__":
    main()
