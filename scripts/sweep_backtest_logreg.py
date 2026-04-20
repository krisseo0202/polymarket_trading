"""
Honest delta sweep for the toy logreg backtest.

Trains the logreg once on the sharded BTC + orderbook dataset, then sweeps
delta and reports ONE trading table under the realistic execution model:
first-eligible entry at tte >= --tte-min, with Polymarket taker fees
(7.2% × p(1-p)) and linear top-3 depth-walk slippage.

Usage:
    ./.venv/bin/python scripts/sweep_backtest_logreg.py
    ./.venv/bin/python scripts/sweep_backtest_logreg.py --bet-size 50 --tte-min 0
    ./.venv/bin/python scripts/sweep_backtest_logreg.py --taker-fee 0 --slippage none
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.backtest_logreg_edge import (
    build_merged_dataset, load_and_derive, train_model,
)


def _apply_slippage(best_ask: float, spread: float, depth_3: float,
                    shares_wanted: float, model: str):
    """Linear top-3 walk: fill_price = best_ask + (size/depth_3) * spread.

    Returns (fill_price, shares_filled). Partial fill when size > depth_3;
    excess capital is not deployed.
    """
    if model == "none" or not spread or not depth_3 or depth_3 <= 0:
        return float(best_ask), float(shares_wanted)
    size_frac = min(1.0, shares_wanted / float(depth_3))
    fill_price = float(best_ask) + size_frac * max(0.0, float(spread))
    shares_filled = min(shares_wanted, float(depth_3))
    return fill_price, shares_filled


def _polymarket_taker_fee(shares: float, price: float, rate: float) -> float:
    """Polymarket taker fee: rate × shares × p × (1 − p). Crypto rate = 0.072.
    https://docs.polymarket.com/trading/fees
    """
    p = max(0.0, min(1.0, float(price)))
    return float(rate) * float(shares) * p * (1.0 - p)


_SIDE_COLS = {
    "YES": ("edge_yes", "up_ask",   "up_spread",   "up_ask_depth_3",   1),
    "NO":  ("edge_no",  "down_ask", "down_spread", "down_ask_depth_3", 0),
}


def find_trades_first_eligible(test_df, delta: float, bet_size: float,
                               taker_fee_rate: float = 0.072,
                               slippage: str = "linear"):
    """Pick the FIRST moment per contract where edge ≥ delta — what a live bot does.

    Applies taker fee + depth-aware slippage at entry. Holds to settlement.
    Assumes test_df is pre-sorted by (contract_id, timestamp), which
    build_merged_dataset guarantees.
    """
    trades, skipped = [], []
    for cid, g in test_df.groupby("contract_id", sort=True):
        # Pick the side that triggers earliest (YES and NO evaluated in parallel).
        best_side, best_ts, best_row = None, float("inf"), None
        for side, (edge_col, *_rest) in _SIDE_COLS.items():
            hit = g[g[edge_col] >= delta]
            if hit.empty:
                continue
            ts = hit["timestamp"].min()
            if ts < best_ts:
                best_side, best_ts, best_row = side, ts, hit.iloc[0]
        if best_row is None:
            skipped.append(int(cid))
            continue

        _, ask_col, spread_col, depth_col, win_target = _SIDE_COLS[best_side]
        top_ask = float(best_row[ask_col])
        if top_ask <= 0:
            skipped.append(int(cid))
            continue
        spread = float(best_row.get(spread_col, 0.0) or 0.0)
        depth_3 = float(best_row.get(depth_col, 0.0) or 0.0)
        payout = 1.0 if int(best_row["target_up"]) == win_target else 0.0

        shares_wanted = bet_size / top_ask
        fill_price, shares_filled = _apply_slippage(
            top_ask, spread, depth_3, shares_wanted, slippage,
        )
        if shares_filled <= 0:
            skipped.append(int(cid))
            continue
        capital_used = shares_filled * fill_price
        fee = _polymarket_taker_fee(shares_filled, fill_price, taker_fee_rate)
        pnl = shares_filled * (payout - fill_price) - fee
        trades.append({
            "contract_id": int(cid), "side": best_side,
            "entry_price": fill_price, "top_ask": top_ask,
            "shares_wanted": shares_wanted, "shares_filled": shares_filled,
            "capital_used": capital_used, "payout": payout,
            "fee": fee, "pnl": pnl, "win": pnl > 0,
            "partial": shares_filled < shares_wanted - 1e-9,
        })
    return trades, skipped


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return ece


def _safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--btc-path", default=os.path.join(_ROOT, "data", "backtest_working", "btc_1s.csv"))
    p.add_argument("--orderbook-path", default=os.path.join(_ROOT, "data", "backtest_working", "live_orderbook_snapshots.csv"))
    p.add_argument("--bet-size", type=float, default=20.0)
    p.add_argument("--valid-fraction", type=float, default=0.20)
    p.add_argument("--row-interval", type=int, default=15)
    p.add_argument("--tte-min", type=int, default=240,
                   help="Minimum time-to-expiry (s) for entry. Default 240 = first minute of slot.")
    p.add_argument("--deltas", default="0.01,0.02,0.03,0.04,0.05,0.06")
    p.add_argument("--taker-fee", type=float, default=0.072,
                   help="Polymarket taker fee rate. Default 0.072 (crypto). 0 to disable.")
    p.add_argument("--slippage", choices=["none", "linear"], default="linear",
                   help="Slippage model. 'linear' walks top-3 depth. 'none' uses best_ask.")
    args = p.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]

    # Load, build, train (once).
    btc_df, markets_df, ob_df = load_and_derive(args.btc_path, args.orderbook_path)
    if markets_df.empty:
        raise SystemExit("No overlapping contracts")
    merged = build_merged_dataset(btc_df, markets_df, ob_df, args.row_interval)
    _model, _scaler, train_df, test_df, p_hat = train_model(merged, args.valid_fraction)

    y = test_df["target_up"].to_numpy(dtype=float)
    brier = brier_score_loss(y, p_hat)
    logloss = log_loss(y, p_hat, labels=[0.0, 1.0])
    auc = _safe_auc(y, p_hat)
    acc = float(((p_hat >= 0.5).astype(int) == y.astype(int)).mean())
    ece = _ece(y, p_hat)

    # Restrict to the decision zone (tte >= --tte-min).
    test_decision = test_df[test_df["time_to_expiry_sec"] >= args.tte_min].copy()

    print()
    print("=" * 78)
    print("  Toy Logreg Backtest")
    print("=" * 78)
    print(f"  Train / test contracts : {train_df['contract_id'].nunique()} / {test_df['contract_id'].nunique()}")
    print(f"  Decision-zone rows     : {len(test_decision):,}  (tte ≥ {args.tte_min}s)")
    print(f"  Test Up / Down         : {int(y.sum())} / {int(len(y)-y.sum())}")
    print()
    print(f"  Calibration (row-level, all tte):")
    print(f"    Brier    : {brier:.4f}   (0.25 = coin flip)")
    print(f"    Log Loss : {logloss:.4f}   (0.693 = coin flip)")
    print(f"    AUC      : {auc:.4f}   (0.5  = random)")
    print(f"    Accuracy : {acc:.1%}")
    print(f"    ECE      : {ece:.4f}")
    print()
    fee_str = "off" if args.taker_fee == 0 else f"{args.taker_fee*100:.1f}% × p(1-p)"
    print(f"  Trading : first-eligible entry · ${args.bet_size:.0f}/trade · "
          f"fees {fee_str} · slippage {args.slippage}")
    print("  " + "-" * 76)
    print(f"  {'delta':>6}  {'trades':>6}  {'part.':>5}  {'win%':>6}  "
          f"{'fees':>7}  {'PnL':>9}  {'maxDD':>9}  {'ROI%':>6}")
    print("  " + "-" * 76)
    for d in deltas:
        trades, skipped = find_trades_first_eligible(
            test_decision, d, args.bet_size,
            taker_fee_rate=args.taker_fee, slippage=args.slippage,
        )
        if not trades:
            print(f"  {d:>6.3f}  {'—':>6}")
            continue
        pnl = np.array([t["pnl"] for t in trades])
        wins = sum(1 for t in trades if t["win"])
        partial = sum(1 for t in trades if t.get("partial"))
        fees = sum(float(t.get("fee", 0.0)) for t in trades)
        capital = sum(float(t.get("capital_used", args.bet_size)) for t in trades) or 1.0
        cum = np.cumsum(pnl)
        dd = float((cum - np.maximum.accumulate(cum)).min())
        roi = pnl.sum() / capital * 100.0
        print(f"  {d:>6.3f}  {len(trades):>6}  {partial:>5}  {wins/len(trades)*100:>5.1f}%  "
              f"${fees:>5.2f}  ${pnl.sum():>+7.2f}  ${dd:>+7.2f}  {roi:>+5.2f}%")
    print("  " + "-" * 76)
    print()
    print("  One fold, single split. Do not treat ROI as signal without walk-forward.")
    print("=" * 78)


if __name__ == "__main__":
    main()
