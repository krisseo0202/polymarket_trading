"""
End-to-end XGBoost backtest pipeline for BTC Up/Down Polymarket markets.

Steps:
  1. Collect last N hours of resolved 5-minute BTC Up/Down markets from Polymarket
  2. Fetch BTC 1-minute klines from Binance for each slot
  3. Reconstruct feature snapshots via the existing feature builder
  4. Train XGBoost on an 80% walk-forward split
  5. Evaluate on holdout: accuracy, log-loss, Brier score, simulated PnL
  6. Save model artifacts to --model-dir

Usage:
    python scripts/backtest_xgb.py --hours 24
    python scripts/backtest_xgb.py --hours 1 --verbose      # quick sanity check
    python scripts/backtest_xgb.py \\
        --hours 24 \\
        --feature-groups btc_core,btc_vol,market_book,strike_time,indicators

Note:
    Binance 1-minute klines are used for BTC price history.  The FVG / TD Sequential
    indicator features require 20+ 5-second bars, which are not available from 1m klines.
    The 'indicators' group will be all-zero unless finer-grained BTC data is provided.
    Exclude it (the default) to avoid training on zero-padded noise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path setup — project root and scripts/ both need to be importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from collect_history import (  # noqa: E402  (after sys.path setup)
    SLOT_INTERVAL,
    current_slot_ts,
    fetch_market_for_slot,
    fetch_price_history,
    parse_strike_price,
)
from src.api.types import OrderBook, OrderBookEntry  # noqa: E402
from src.models.feature_builder import build_live_features  # noqa: E402
from src.models.schema import (  # noqa: E402
    DEFAULT_FEATURE_VALUES,
    DEFAULT_THRESHOLDS,
    FEATURE_COLUMNS,
    MODEL_NAME,
)

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "xgboost is required.  Install it:  pip install xgboost"
    ) from exc


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, List[str]] = {
    "btc_core": [
        "btc_mid", "btc_ret_15s", "btc_ret_30s", "btc_ret_60s",
    ],
    "btc_vol": [
        "btc_vol_30s", "btc_vol_60s",
    ],
    "market_book": [
        "yes_bid", "yes_ask", "yes_mid", "yes_spread", "yes_spread_pct",
        "yes_book_imbalance", "yes_ret_30s",
        "no_bid",  "no_ask",  "no_mid",  "no_spread",  "no_spread_pct",
        "no_book_imbalance",  "no_ret_30s",
    ],
    "strike_time": [
        "strike_price", "seconds_to_expiry", "moneyness", "distance_to_strike_bps",
    ],
    "indicators": [
        "active_bull_gap", "active_bear_gap", "latest_gap_distance_pct",
        "bull_setup", "bear_setup", "buy_cd", "sell_cd",
        "buy_9", "sell_9", "buy_13", "sell_13",
    ],
}

# Default: exclude market_book (skips 2 CLOB tick-fetch calls/slot, ~2× faster collection)
# and indicators (unavailable from 1-minute Binance klines).
DEFAULT_FEATURE_GROUPS = "btc_core,btc_vol,strike_time"

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# ---------------------------------------------------------------------------
# Data-collection helpers
# ---------------------------------------------------------------------------

def fetch_btc_klines(
    slot_ts: int,
    pred_offset_s: int = 90,
    context_s: int = 300,
) -> List[Tuple[float, float]]:
    """Fetch 1-minute BTCUSDT klines from Binance for the window
    [slot_ts - context_s, slot_ts + pred_offset_s].

    Returns List[(close_time_seconds, close_price)].
    Empty list on network error.
    """
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": (slot_ts - context_s) * 1000,
        "endTime":   (slot_ts + pred_offset_s) * 1000,
        "limit": 10,
    }
    try:
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        # kline format: [open_time_ms, open, high, low, close, vol, close_time_ms, ...]
        return [
            (float(k[6]) / 1000.0, float(k[4]))   # (close_time_s, close_price)
            for k in resp.json()
        ]
    except requests.RequestException as exc:
        print(f"  [WARN] Binance klines failed for slot {slot_ts}: {exc}", file=sys.stderr)
        return []


def make_synthetic_book(token_id: str, mid: float, spread_pct: float = 0.005) -> OrderBook:
    """Build a minimal OrderBook from a single mid-price observation.

    bid = mid * (1 - spread_pct/2),  ask = mid * (1 + spread_pct/2).
    Equal bid/ask sizes → book_imbalance = 0.0.
    """
    half = mid * (spread_pct / 2.0)
    bid = max(0.01, mid - half)
    ask = min(0.99, mid + half)
    return OrderBook(
        market_id="",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=500.0)],
        asks=[OrderBookEntry(price=ask, size=500.0)],
    )


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------

def build_snapshot(
    slot_ts: int,
    btc_klines: List[Tuple[float, float]],
    yes_ticks: List[dict],
    no_ticks: List[dict],
    market: dict,
    pred_offset_s: int,
) -> Optional[dict]:
    """Reconstruct a feature snapshot at t = slot_ts + pred_offset_s.

    Returns None if BTC price history is insufficient.
    """
    now_ts = float(slot_ts + pred_offset_s)

    # Strict lookahead prevention: only prices observed before now_ts
    btc_prices = [(ts, p) for ts, p in btc_klines if ts <= now_ts]
    if len(btc_prices) < 2:
        return None

    yes_history = [(float(t["t"]), float(t["p"])) for t in yes_ticks if float(t["t"]) <= now_ts]
    no_history  = [(float(t["t"]), float(t["p"])) for t in no_ticks  if float(t["t"]) <= now_ts]

    # Mid prices for synthetic books
    yes_mid = yes_history[-1][1] if yes_history else 0.50
    no_mid  = no_history[-1][1]  if no_history  else 0.50

    # Complementary fallback when one side has no ticks
    if not yes_history and no_history:
        yes_mid = max(0.01, min(0.99, 1.0 - no_mid))
    elif not no_history and yes_history:
        no_mid = max(0.01, min(0.99, 1.0 - yes_mid))

    return {
        "btc_prices":     btc_prices,
        "yes_book":       make_synthetic_book(market["up_token"],   yes_mid),
        "no_book":        make_synthetic_book(market["down_token"], no_mid),
        "yes_history":    yes_history,
        "no_history":     no_history,
        "question":       market["question"],
        "strike_price":   parse_strike_price(market["question"]),
        "slot_expiry_ts": float(slot_ts + SLOT_INTERVAL),
        "now_ts":         now_ts,
    }


# ---------------------------------------------------------------------------
# Phase 1+2: collect + build snapshots
# ---------------------------------------------------------------------------

@dataclass
class SlotRecord:
    slot_ts:        int
    slot_utc:       str
    label:          int            # 1 = Up, 0 = Down
    features:       Dict[str, float]
    feature_status: str
    yes_ask:        float
    no_ask:         float


def collect_and_build(
    hours: int,
    pred_offset_s: int,
    verbose: bool,
    selected_groups: Optional[List[str]] = None,
    rate_limit_s: float = 0.3,
) -> List[SlotRecord]:
    """Iterate slots newest → oldest, fetch data, build feature snapshots."""
    now_slot    = current_slot_ts()
    start_slot  = now_slot - SLOT_INTERVAL        # skip current open slot
    total_slots = (hours * 3600) // SLOT_INTERVAL
    end_slot    = start_slot - total_slots * SLOT_INTERVAL

    need_ticks = selected_groups is None or "market_book" in (selected_groups or [])
    records: List[SlotRecord] = []
    found = skipped = 0
    slot = start_slot

    while slot > end_slot:
        market = fetch_market_for_slot(slot)
        time.sleep(rate_limit_s)

        outcome = (market or {}).get("outcome")
        if not market or not market.get("closed") or outcome not in ("Up", "Down"):
            skipped += 1
            slot -= SLOT_INTERVAL
            continue

        label = 1 if outcome == "Up" else 0

        # Skip expensive CLOB tick fetches when market_book features are not used
        if need_ticks:
            yes_ticks = fetch_price_history(market["up_token"],   slot); time.sleep(rate_limit_s)
            no_ticks  = fetch_price_history(market["down_token"], slot); time.sleep(rate_limit_s)
        else:
            yes_ticks, no_ticks = [], []

        btc_klines = fetch_btc_klines(slot, pred_offset_s);          time.sleep(rate_limit_s)

        snapshot = build_snapshot(slot, btc_klines, yes_ticks, no_ticks, market, pred_offset_s)
        if snapshot is None:
            if verbose:
                print(f"  SKIP {market['slot_utc']} — insufficient BTC data (<2 klines before t+{pred_offset_s}s)")
            skipped += 1
            slot -= SLOT_INTERVAL
            continue

        result   = build_live_features(snapshot)
        yes_ask  = float(snapshot["yes_book"].asks[0].price)
        no_ask   = float(snapshot["no_book"].asks[0].price)
        found   += 1

        if verbose:
            print(f"  {market['slot_utc']}  {'Up  ' if label else 'Down'}  status={result.status}")

        records.append(SlotRecord(
            slot_ts=slot,
            slot_utc=market["slot_utc"],
            label=label,
            features=result.features,
            feature_status=result.status,
            yes_ask=yes_ask,
            no_ask=no_ask,
        ))

        progress = found + skipped
        if progress % 20 == 0 and not verbose:
            print(f"  Progress: {progress}/{total_slots}  valid={found}  skipped={skipped}")

        slot -= SLOT_INTERVAL

    print(f"  Done: {found} valid slots, {skipped} skipped out of {total_slots} total")
    return records


def build_dataframe(records: List[SlotRecord], selected_groups: List[str]) -> pd.DataFrame:
    """Assemble feature DataFrame.  Columns outside selected groups are zeroed."""
    active_cols: set = set()
    for g in selected_groups:
        active_cols.update(FEATURE_GROUPS.get(g, []))

    rows = []
    for r in records:
        row: dict = {
            "snapshot_ts":  r.slot_ts,
            "label":        r.label,
            # Use sim_ prefix to avoid collision with yes_ask/no_ask feature columns
            "sim_yes_ask":  r.yes_ask,
            "sim_no_ask":   r.no_ask,
        }
        for col in FEATURE_COLUMNS:
            raw = r.features.get(col, DEFAULT_FEATURE_VALUES.get(col, 0.0))
            row[col] = float(raw) if col in active_cols else 0.0
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("snapshot_ts").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Calibration helpers (match train_btc_updown_xgb.py exactly)
# ---------------------------------------------------------------------------

def _build_calibration(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    if len(preds) == 0:
        return {"bin_edges": [0.0, 1.0], "bin_values": [0.5]}
    edges = np.quantile(preds, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    unique_edges: List[float] = [float(edges[0])]
    for e in edges[1:]:
        if float(e) > unique_edges[-1]:
            unique_edges.append(float(e))
    if len(unique_edges) < 2:
        unique_edges = [0.0, 1.0]
    global_mean = float(np.mean(labels)) if len(labels) else 0.5
    values: List[float] = []
    for i in range(len(unique_edges) - 1):
        lo, hi = unique_edges[i], unique_edges[i + 1]
        if i == len(unique_edges) - 2:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        values.append(float(np.mean(labels[mask])) if np.any(mask) else global_mean)
    return {"bin_edges": unique_edges, "bin_values": values}


def _apply_calibration(preds: np.ndarray, calibration: dict) -> np.ndarray:
    edges  = calibration["bin_edges"]
    values = calibration["bin_values"]
    out    = np.empty_like(preds, dtype=float)
    for i, p in enumerate(preds):
        idx     = int(np.searchsorted(edges, p, side="right") - 1)
        idx     = max(0, min(idx, len(values) - 1))
        out[i]  = float(values[idx])
    return out


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# Phase 3+4: train and save
# ---------------------------------------------------------------------------

def train_and_save(
    df: pd.DataFrame,
    val_split: float,
    num_round: int,
    output_dir: str,
    selected_groups: List[str],
) -> Tuple["xgb.Booster", dict, np.ndarray, np.ndarray, pd.DataFrame]:
    """Walk-forward split → train XGBoost → calibrate → save artifacts.

    Returns (booster, calibration, X_valid, y_valid, valid_meta_df).
    valid_meta_df has yes_ask / no_ask columns aligned to validation rows.
    """
    y = df["label"].to_numpy(dtype=float)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    split_idx = max(1, int(len(df) * (1.0 - val_split)))

    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]
    valid_meta = df[["sim_yes_ask", "sim_no_ask"]].iloc[split_idx:].reset_index(drop=True)

    print(f"  Training on {len(X_train)} rows, validating on {len(X_valid)} rows ...")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=FEATURE_COLUMNS)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "eta":              0.05,
        "max_depth":        4,
        "subsample":        0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "seed":             42,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )

    raw_valid      = booster.predict(dvalid)
    calibration    = _build_calibration(raw_valid, y_valid)
    cal_valid      = _apply_calibration(raw_valid, calibration)

    os.makedirs(output_dir, exist_ok=True)
    model_path   = os.path.join(output_dir, "btc_updown_xgb.json")
    feature_path = os.path.join(output_dir, "btc_updown_xgb_features.json")
    meta_path    = os.path.join(output_dir, "btc_updown_xgb_meta.json")

    booster.save_model(model_path)
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    metadata = {
        "model_name":    MODEL_NAME,
        "model_version": f"{MODEL_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "feature_columns": FEATURE_COLUMNS,
        "feature_groups":  selected_groups,
        "thresholds":      DEFAULT_THRESHOLDS,
        "calibration":     calibration,
        "metrics": {
            "train_rows":          int(len(X_train)),
            "valid_rows":          int(len(X_valid)),
            "raw_log_loss":        _log_loss(y_valid, raw_valid),
            "calibrated_log_loss": _log_loss(y_valid, cal_valid),
            "raw_brier":           _brier(y_valid, raw_valid),
            "calibrated_brier":    _brier(y_valid, cal_valid),
            "positive_rate":       float(np.mean(y)),
        },
        "training": {
            "feature_groups": selected_groups,
            "val_fraction":   val_split,
            "num_round":      num_round,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved artifacts → {output_dir}/")
    print(f"    logloss (calibrated) = {metadata['metrics']['calibrated_log_loss']:.5f}")
    print(f"    brier   (calibrated) = {metadata['metrics']['calibrated_brier']:.5f}")
    return booster, calibration, X_valid, y_valid, valid_meta


# ---------------------------------------------------------------------------
# Phase 5: evaluate
# ---------------------------------------------------------------------------

def evaluate(
    booster: "xgb.Booster",
    calibration: dict,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_meta: pd.DataFrame,
    bet_size: float,
) -> dict:
    """Run model on holdout, compute all metrics."""
    dvalid     = xgb.DMatrix(X_valid, feature_names=FEATURE_COLUMNS)
    raw_preds  = booster.predict(dvalid)
    cal_preds  = np.clip(_apply_calibration(raw_preds, calibration), 1e-9, 1.0 - 1e-9)

    pred_yes   = cal_preds >= 0.5
    actual_yes = y_valid == 1

    # Simulated PnL
    pnl_list: List[float] = []
    for i in range(len(cal_preds)):
        if pred_yes[i]:
            entry  = float(valid_meta.iloc[i]["sim_yes_ask"])
            payout = 1.0 if actual_yes[i] else 0.0
        else:
            entry  = float(valid_meta.iloc[i]["sim_no_ask"])
            payout = 1.0 if not actual_yes[i] else 0.0
        shares = bet_size / max(entry, 0.01)
        pnl_list.append(shares * (payout - entry))

    tp = int(np.sum( pred_yes &  actual_yes))
    fp = int(np.sum( pred_yes & ~actual_yes))
    tn = int(np.sum(~pred_yes & ~actual_yes))
    fn = int(np.sum(~pred_yes &  actual_yes))

    # Calibration curve (5 probability bins)
    cal_curve: List[dict] = []
    for i in range(5):
        lo, hi = i / 5.0, (i + 1) / 5.0
        mask = (cal_preds >= lo) & (cal_preds < hi) if i < 4 else (cal_preds >= lo)
        if np.any(mask):
            cal_curve.append({
                "range":        f"[{lo:.1f},{hi:.1f})",
                "n":            int(np.sum(mask)),
                "mean_prob_yes": float(np.mean(cal_preds[mask])),
                "actual_up_rate": float(np.mean(y_valid[mask])),
            })

    pnl_arr = np.array(pnl_list)
    return {
        "n":            len(y_valid),
        "accuracy":     float(np.mean(pred_yes == actual_yes)),
        "log_loss":     _log_loss(y_valid, cal_preds),
        "brier":        _brier(y_valid, cal_preds),
        "confusion":    {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "precision":    tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall":       tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "total_pnl":    float(np.sum(pnl_arr)),
        "pnl_per_trade": float(np.mean(pnl_arr)) if len(pnl_arr) else 0.0,
        "calibration":  cal_curve,
    }


def print_report(metrics: dict, feature_groups: List[str], bet_size: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  BTCUpDownXGBModel — Backtest Report")
    print(f"  Feature groups: {', '.join(feature_groups)}")
    print(sep)
    print(f"Holdout rows    : {metrics['n']}")
    print(f"Accuracy        : {metrics['accuracy']:.1%}")
    print(f"Log-loss        : {metrics['log_loss']:.5f}   (random ≈ 0.693)")
    print(f"Brier score     : {metrics['brier']:.5f}   (random ≈ 0.250)")
    print(f"Precision       : {metrics['precision']:.1%}   (of YES calls, how many were Up)")
    print(f"Recall          : {metrics['recall']:.1%}   (of Up outcomes caught as YES)")
    c = metrics["confusion"]
    print(f"\nConfusion Matrix :")
    print(f"                    Pred YES   Pred NO")
    print(f"  Actual Up    :     {c['TP']:>5}     {c['FN']:>5}")
    print(f"  Actual Down  :     {c['FP']:>5}     {c['TN']:>5}")
    print(f"\nSimulated PnL  (${bet_size:.0f}/trade) :")
    print(f"  Total        : ${metrics['total_pnl']:+.2f}")
    print(f"  Per trade    : ${metrics['pnl_per_trade']:+.4f}")
    if metrics["calibration"]:
        print(f"\nCalibration (prob_yes vs actual Up rate) :")
        for b in metrics["calibration"]:
            bar = "#" * int(b["actual_up_rate"] * 20)
            print(
                f"  {b['range']:>12}  n={b['n']:>3}  "
                f"pred={b['mean_prob_yes']:.3f}  actual={b['actual_up_rate']:.3f}  {bar}"
            )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end XGBoost backtest for BTC Up/Down Polymarket markets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Hours of history to collect",
    )
    parser.add_argument(
        "--feature-groups", type=str, default=DEFAULT_FEATURE_GROUPS,
        help=(
            f"Comma-separated feature groups to include in training. "
            f"Available: {', '.join(FEATURE_GROUPS)}. "
            f"Excluded groups are zeroed out (schema stays fixed)."
        ),
    )
    parser.add_argument(
        "--pred-offset", type=int, default=90,
        help="Seconds into the 5-min slot at which to make the prediction",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of data (from the tail) held out for evaluation",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Directory for saved model artifacts",
    )
    parser.add_argument(
        "--num-round", type=int, default=200,
        help="XGBoost boosting rounds",
    )
    parser.add_argument(
        "--bet-size", type=float, default=20.0,
        help="Simulated bet size in USDC for PnL calculation",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-slot collection details",
    )
    args = parser.parse_args()

    # Validate feature groups
    selected_groups = [g.strip() for g in args.feature_groups.split(",") if g.strip()]
    unknown = [g for g in selected_groups if g not in FEATURE_GROUPS]
    if unknown:
        raise SystemExit(
            f"Unknown feature group(s): {unknown}\nAvailable: {list(FEATURE_GROUPS)}"
        )

    print("=== XGBoost Backtest Pipeline ===")
    print(f"Hours         : {args.hours}")
    print(f"Pred offset   : t+{args.pred_offset}s into each 5-min slot")
    print(f"Val split     : {args.val_split:.0%} holdout from tail")
    print(f"Feature groups: {', '.join(selected_groups)}")

    # --- Phase 1+2: collect and build ---
    print(f"\n[1/4] Collecting {args.hours}h of history and building snapshots ...")
    records = collect_and_build(args.hours, args.pred_offset, args.verbose, selected_groups)

    if len(records) < 10:
        raise SystemExit(
            f"Only {len(records)} valid slots collected — need at least 10 to train. "
            "Try --hours 48 or check your network connection."
        )

    df = build_dataframe(records, selected_groups)
    print(f"[2/4] Feature DataFrame: {len(df)} rows × {len(FEATURE_COLUMNS)} columns")
    if len(df) < 30:
        print(
            f"  [WARN] Only {len(df)} rows — metrics will be noisy. "
            "Consider --hours 48 for more data."
        )

    # --- Phase 3+4: train ---
    print(f"\n[3/4] Training XGBoost ({args.num_round} rounds) ...")
    booster, calibration, X_valid, y_valid, valid_meta = train_and_save(
        df, args.val_split, args.num_round, args.model_dir, selected_groups
    )

    if len(X_valid) == 0:
        raise SystemExit("No validation rows after split. Reduce --val-split or collect more data.")

    # --- Phase 5: evaluate ---
    print(f"\n[4/4] Evaluating on {len(X_valid)} holdout rows ...")
    metrics = evaluate(booster, calibration, X_valid, y_valid, valid_meta, args.bet_size)
    print_report(metrics, selected_groups, args.bet_size)


if __name__ == "__main__":
    main()
