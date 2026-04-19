"""
Logistic Regression Edge Backtest for BTC Up/Down Polymarket markets.

Trains a logistic regression on BTC features to predict P(Up), then
backtests an edge-based entry strategy against live orderbook data.
Entry rule: buy when model disagrees with market by more than delta + costs.
Exit rule: hold to expiry (binary resolution).

Usage:
    ./.venv/bin/python scripts/backtest_logreg_edge.py
    ./.venv/bin/python scripts/backtest_logreg_edge.py --delta 0.03 --bet-size 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.backtest.snapshot_dataset import build_btc_decision_dataset, load_btc_prices

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install: pip install scikit-learn"
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants ────────────────────────────────────────────────────────────────

BTC_PATH = os.path.join(_ROOT, "data", "btc_1s.csv")
ORDERBOOK_PATH = os.path.join(_ROOT, "data", "live_orderbook_snapshots.csv")

FEATURES = [
    "ret_15s", "ret_30s", "ret_60s", "rolling_vol_60s",
    "rsi_14", "dist_to_strike", "ma_12_gap", "time_to_expiry_sec",
]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logistic regression edge backtest")
    p.add_argument("--btc-path", default=BTC_PATH)
    p.add_argument("--orderbook-path", default=ORDERBOOK_PATH)
    p.add_argument("--bet-size", type=float, default=20.0, help="USDC per trade")
    p.add_argument("--delta", type=float, default=0.05, help="Min margin of safety")
    p.add_argument("--valid-fraction", type=float, default=0.20)
    p.add_argument("--row-interval", type=int, default=15, help="Decision row spacing (s)")
    p.add_argument("--output-dir", default=os.path.join(_ROOT, "data"))
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def _asof_price(ts_arr: np.ndarray, price_arr: np.ndarray, target: float) -> Optional[float]:
    idx = int(np.searchsorted(ts_arr, target, side="right")) - 1
    if idx < 0:
        return None
    return float(price_arr[idx])


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


# ── Phase 1: load data & derive outcomes ─────────────────────────────────────

def load_and_derive(
    btc_path: str, ob_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (btc_df, markets_df, ob_df)."""
    btc_df = load_btc_prices(btc_path)
    ob_df = pd.read_csv(ob_path)

    btc_ts = btc_df["ts"].to_numpy(dtype=float)
    btc_price = btc_df["price"].to_numpy(dtype=float)
    btc_min, btc_max = float(btc_ts[0]), float(btc_ts[-1])

    slots = sorted(ob_df["slot_ts"].unique())
    valid_slots = [int(s) for s in slots if s >= btc_min and s + 300 <= btc_max]

    rows: List[Dict[str, object]] = []
    for s in valid_slots:
        p_open = _asof_price(btc_ts, btc_price, float(s))
        p_close = _asof_price(btc_ts, btc_price, float(s + 300))
        if p_open is None or p_close is None:
            continue
        rows.append({
            "slot_ts": s,
            "outcome": "Up" if p_close >= p_open else "Down",
            "strike_price": p_open,
        })

    markets_df = pd.DataFrame(rows)
    print(f"Loaded {len(btc_df)} BTC ticks, {len(ob_df)} orderbook rows")
    print(f"Derived {len(markets_df)} contracts "
          f"(Up={sum(1 for r in rows if r['outcome']=='Up')}, "
          f"Down={sum(1 for r in rows if r['outcome']=='Down')})")
    return btc_df, markets_df, ob_df


# ── Phase 2: build features + merge orderbook ───────────────────────────────

def build_merged_dataset(
    btc_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    row_interval_sec: int,
) -> pd.DataFrame:
    """Build BTC decision dataset and merge with orderbook snapshots."""
    ds = build_btc_decision_dataset(markets_df, btc_df, row_interval_sec=row_interval_sec)

    # Prepare up-side orderbook. Carry forward depth + spread for slippage models.
    ob_up = ob_df[ob_df["side"] == "up"].copy()
    ob_up["ob_ts"] = ob_up["slot_ts"] + ob_up["elapsed_s"]
    _up_cols = {"best_bid": "up_bid", "best_ask": "up_ask", "mid": "up_mid"}
    if "ask_depth_3" in ob_up.columns:
        _up_cols["ask_depth_3"] = "up_ask_depth_3"
    if "spread" in ob_up.columns:
        _up_cols["spread"] = "up_spread"
    ob_up = ob_up.rename(columns=_up_cols)
    keep_up = ["slot_ts", "ob_ts"] + [v for v in _up_cols.values()]
    ob_up = ob_up[keep_up].sort_values("ob_ts")

    # Prepare down-side orderbook
    ob_down = ob_df[ob_df["side"] == "down"].copy()
    ob_down["ob_ts"] = ob_down["slot_ts"] + ob_down["elapsed_s"]
    _dn_cols = {"best_bid": "down_bid", "best_ask": "down_ask", "mid": "down_mid"}
    if "ask_depth_3" in ob_down.columns:
        _dn_cols["ask_depth_3"] = "down_ask_depth_3"
    if "spread" in ob_down.columns:
        _dn_cols["spread"] = "down_spread"
    ob_down = ob_down.rename(columns=_dn_cols)
    keep_down = ["slot_ts", "ob_ts"] + [v for v in _dn_cols.values()]
    ob_down = ob_down[keep_down].sort_values("ob_ts")

    # Asof-merge per contract
    merged_parts: List[pd.DataFrame] = []
    for contract_id in ds["contract_id"].unique():
        contract_rows = ds[ds["contract_id"] == contract_id].copy().sort_values("timestamp")
        up_slice = ob_up[ob_up["slot_ts"] == contract_id].copy()
        down_slice = ob_down[ob_down["slot_ts"] == contract_id].copy()

        if up_slice.empty or down_slice.empty:
            continue

        m = pd.merge_asof(
            contract_rows, up_slice.drop(columns=["slot_ts"]),
            left_on="timestamp", right_on="ob_ts", direction="backward",
        )
        m = pd.merge_asof(
            m.sort_values("timestamp"),
            down_slice.drop(columns=["slot_ts"]),
            left_on="timestamp", right_on="ob_ts", direction="backward",
            suffixes=("", "_down"),
        )
        merged_parts.append(m)

    if not merged_parts:
        raise SystemExit("No contracts with orderbook data found")

    merged = pd.concat(merged_parts, ignore_index=True).sort_values(
        ["contract_id", "timestamp"]
    ).reset_index(drop=True)

    # Derived columns
    merged["q_t"] = merged["up_mid"]
    merged["c_t"] = (merged["up_ask"] - merged["up_bid"]) / 2.0

    # Drop rows without orderbook data
    merged = merged.dropna(subset=["q_t", "c_t", "up_ask", "down_ask"]).reset_index(drop=True)
    print(f"Merged dataset: {len(merged)} rows across {merged['contract_id'].nunique()} contracts")
    return merged


# ── Phase 3+4: split & train ────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame, valid_fraction: float
) -> Tuple[LogisticRegression, StandardScaler, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Walk-forward train/test split at contract level, train logistic regression.

    Returns (model, scaler, train_df, test_df, p_hat_test).
    """
    contracts = sorted(df["contract_id"].unique())
    n_train = max(1, int(len(contracts) * (1.0 - valid_fraction)))
    train_contracts = set(contracts[:n_train])
    test_contracts = set(contracts[n_train:])

    train_df = df[df["contract_id"].isin(train_contracts)].copy()
    test_df = df[df["contract_id"].isin(test_contracts)].copy()

    print(f"Split: {len(train_contracts)} train contracts ({len(train_df)} rows), "
          f"{len(test_contracts)} test contracts ({len(test_df)} rows)")

    X_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df["target_up"].to_numpy(dtype=float)
    X_test = test_df[FEATURES].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train_s, y_train)

    p_hat_test = model.predict_proba(X_test_s)[:, 1]
    test_df = test_df.copy()
    test_df["p_hat"] = p_hat_test

    # Edges
    test_df["edge_yes"] = test_df["p_hat"] - test_df["q_t"] - test_df["c_t"]
    test_df["edge_no"] = test_df["q_t"] - test_df["p_hat"] - test_df["c_t"]

    return model, scaler, train_df, test_df, p_hat_test


# ── Phase 5: highest-edge entry selection ────────────────────────────────────

def find_trades(
    test_df: pd.DataFrame, delta: float, bet_size: float
) -> Tuple[List[Dict[str, object]], List[int]]:
    """For each test contract, find the highest-edge entry. Return trades and skipped slots."""
    trades: List[Dict[str, object]] = []
    skipped: List[int] = []

    for contract_id, grp in test_df.groupby("contract_id", sort=True):
        best_yes_idx = grp["edge_yes"].idxmax()
        best_no_idx = grp["edge_no"].idxmax()
        best_yes_edge = float(grp.loc[best_yes_idx, "edge_yes"])
        best_no_edge = float(grp.loc[best_no_idx, "edge_no"])

        # Pick the side with larger edge
        if best_yes_edge >= best_no_edge:
            side = "YES"
            edge = best_yes_edge
            row = grp.loc[best_yes_idx]
            entry_price = float(row["up_ask"])
            payout = 1.0 if int(row["target_up"]) == 1 else 0.0
        else:
            side = "NO"
            edge = best_no_edge
            row = grp.loc[best_no_idx]
            entry_price = float(row["down_ask"])
            payout = 1.0 if int(row["target_up"]) == 0 else 0.0

        if edge < delta:
            skipped.append(int(contract_id))
            continue

        if entry_price <= 0:
            skipped.append(int(contract_id))
            continue

        shares = bet_size / entry_price
        pnl = shares * (payout - entry_price)

        trades.append({
            "contract_id": int(contract_id),
            "side": side,
            "entry_ts": int(row["timestamp"]),
            "tte_at_entry": int(row["time_to_expiry_sec"]),
            "p_hat": float(row["p_hat"]),
            "q_t": float(row["q_t"]),
            "c_t": float(row["c_t"]),
            "edge": edge,
            "entry_price": entry_price,
            "payout": payout,
            "outcome": "Up" if int(row["target_up"]) == 1 else "Down",
            "shares": shares,
            "pnl": pnl,
            "win": pnl > 0,
        })

    return trades, skipped


# ── Phase 7: report ─────────────────────────────────────────────────────────

def print_report(
    model: LogisticRegression,
    scaler: StandardScaler,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trades: List[Dict[str, object]],
    skipped: List[int],
    delta: float,
    bet_size: float,
) -> None:
    """Print console report."""

    # Model diagnostics on test set
    y_test = test_df["target_up"].to_numpy(dtype=float)
    p_hat = test_df["p_hat"].to_numpy(dtype=float)
    preds = (p_hat >= 0.5).astype(int)
    acc = float(np.mean(preds == y_test))
    brier = _brier(y_test, p_hat)
    logloss = _log_loss(y_test, p_hat)

    print()
    print("=" * 64)
    print("  Logistic Regression Edge Backtest — Results")
    print("=" * 64)

    # Feature coefficients
    print()
    print("  Feature Coefficients (scaled):")
    coefs = model.coef_[0]
    for feat, coef in sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:<22s} {coef:+.4f}")
    print(f"    {'intercept':<22s} {model.intercept_[0]:+.4f}")

    # Model metrics
    print()
    print(f"  Model Metrics (test set, {len(test_df)} rows):")
    print(f"  {'─' * 50}")
    print(f"  Accuracy                : {acc:.1%}")
    print(f"  Brier score             : {brier:.4f}  (0.25=coin flip)")
    print(f"  Log-loss                : {logloss:.4f}")
    print(f"  Test Up/Down            : {int(y_test.sum())}/{int(len(y_test) - y_test.sum())}")

    # PnL summary
    n_contracts = test_df["contract_id"].nunique()
    n_traded = len(trades)
    n_skipped = len(skipped)

    print()
    print(f"  PnL Simulation (${bet_size:.0f}/trade, delta={delta})")
    print(f"  {'─' * 50}")
    print(f"  Test contracts          : {n_contracts}")
    print(f"  Contracts traded        : {n_traded}")
    print(f"  Contracts skipped       : {n_skipped}  (edge < delta)")

    if n_traded == 0:
        print("  No trades — lower delta or check model quality.")
        print("=" * 64)
        return

    pnl_arr = np.array([t["pnl"] for t in trades])
    wins = sum(1 for t in trades if t["win"])
    yes_count = sum(1 for t in trades if t["side"] == "YES")
    no_count = n_traded - yes_count
    edges = np.array([t["edge"] for t in trades])

    print(f"  Win rate                : {wins}/{n_traded} = {wins/n_traded:.1%}")
    print(f"  YES / NO trades         : {yes_count} / {no_count}")
    print(f"  Total PnL               : ${pnl_arr.sum():+.2f}")
    print(f"  Avg PnL per trade       : ${pnl_arr.mean():+.2f}")
    print(f"  Mean edge at entry      : {edges.mean():.4f}")
    print(f"  Max edge at entry       : {edges.max():.4f}")

    # Equity curve stats
    cum_pnl = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    print(f"  Max drawdown            : ${dd.min():+.2f}")

    print()
    print("  Trade Log:")
    print(f"  {'slot_ts':>12s}  {'side':>4s}  {'tte':>4s}  {'p_hat':>6s}  "
          f"{'q_t':>6s}  {'c_t':>5s}  {'edge':>6s}  {'entry':>5s}  "
          f"{'pay':>3s}  {'pnl':>8s}  {'out':>4s}")
    print(f"  {'─' * 78}")
    for t in trades:
        print(f"  {t['contract_id']:>12d}  {t['side']:>4s}  {t['tte_at_entry']:>4d}  "
              f"{t['p_hat']:>6.3f}  {t['q_t']:>6.3f}  {t['c_t']:>5.3f}  "
              f"{t['edge']:>6.3f}  {t['entry_price']:>5.2f}  "
              f"{t['payout']:>3.0f}  ${t['pnl']:>+7.2f}  {t['outcome']:>4s}")

    print("=" * 64)


# ── Phase 7b: visualizations ────────────────────────────────────────────────

def plot_results(
    test_df: pd.DataFrame,
    trades: List[Dict[str, object]],
    output_dir: str,
) -> None:
    """Generate and save backtest visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Logistic Regression Edge Backtest", fontsize=14, fontweight="bold")

    # 1. Equity curve
    ax = axes[0, 0]
    if trades:
        pnl_arr = [t["pnl"] for t in trades]
        cum_pnl = np.cumsum(pnl_arr)
        ax.plot(range(1, len(cum_pnl) + 1), cum_pnl, "b.-", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL ($)")
        ax.set_title("Equity Curve")
    else:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Equity Curve")

    # 2. Edge distribution at entry
    ax = axes[0, 1]
    if trades:
        edges = [t["edge"] for t in trades]
        ax.hist(edges, bins=max(3, len(edges) // 2), color="#2ecc71", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="zero edge")
        ax.set_xlabel("Edge at entry")
        ax.set_ylabel("Count")
        ax.set_title("Edge Distribution at Entry")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Edge Distribution")

    # 3. Calibration plot
    ax = axes[1, 0]
    y_test = test_df["target_up"].to_numpy(dtype=float)
    p_hat = test_df["p_hat"].to_numpy(dtype=float)
    n_bins = min(5, max(2, len(test_df) // 20))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_actuals = []
    for i in range(n_bins):
        mask = (p_hat >= bin_edges[i]) & (p_hat < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p_hat >= bin_edges[i]) & (p_hat <= bin_edges[i + 1])
        if mask.any():
            bin_centers.append(float(np.mean(p_hat[mask])))
            bin_actuals.append(float(np.mean(y_test[mask])))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.scatter(bin_centers, bin_actuals, s=60, zorder=5, color="#e74c3c", label="Model")
    if len(bin_centers) > 1:
        ax.plot(bin_centers, bin_actuals, "r-", alpha=0.5)
    ax.set_xlabel("Predicted P(Up)")
    ax.set_ylabel("Actual P(Up)")
    ax.set_title("Calibration Plot")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    # 4. p_hat vs q_t scatter
    ax = axes[1, 1]
    ax.scatter(test_df["q_t"], test_df["p_hat"], alpha=0.3, s=10, c=test_df["target_up"],
               cmap="RdYlGn", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="p_hat = q_t")
    ax.set_xlabel("Market price q_t (up_mid)")
    ax.set_ylabel("Model p_hat")
    ax.set_title("Model vs Market")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "logreg_backtest_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Phase 1
    btc_df, markets_df, ob_df = load_and_derive(args.btc_path, args.orderbook_path)
    if markets_df.empty:
        raise SystemExit("No overlapping contracts found")

    # Phase 2
    merged = build_merged_dataset(btc_df, markets_df, ob_df, args.row_interval)

    # Phase 3+4
    model, scaler, train_df, test_df, p_hat = train_model(merged, args.valid_fraction)

    # Phase 5+6
    trades, skipped = find_trades(test_df, args.delta, args.bet_size)

    # Phase 7
    print_report(model, scaler, train_df, test_df, trades, skipped, args.delta, args.bet_size)
    plot_results(test_df, trades, args.output_dir)


if __name__ == "__main__":
    main()
