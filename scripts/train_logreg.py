"""
Train a logistic regression model for BTC Up/Down Polymarket probability trading.

Usage:
    python scripts/train_logreg.py --markets data/backtest_td_rsi_results.csv
    python scripts/train_logreg.py --markets data/backtest_td_rsi_results.csv --btc data/btc_1s.csv
    python scripts/train_logreg.py --markets data/backtest_td_rsi_results.csv --fetch-btc

If --btc is not given and --fetch-btc is set, BTC 1-minute candles are fetched
from Coinbase for the slot time range and cached to data/btc_1m_cache.csv.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest.snapshot_dataset import (
    build_btc_decision_dataset,
    load_btc_prices,
    load_market_history,
)
from src.backtest.data_loader import DataLoader
from src.models.logreg_model import LR_FEATURES, LogRegModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BTC Up/Down LogReg model")
    parser.add_argument("--markets", required=True, help="Resolved markets CSV (needs slot_ts, outcome columns)")
    parser.add_argument("--btc", default=None, help="BTC price CSV/JSONL (ts, price columns)")
    parser.add_argument("--fetch-btc", action="store_true", help="Fetch BTC 1m candles from Coinbase if --btc not given")
    parser.add_argument("--output-dir", default="models/logreg", help="Directory for model artifacts")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="Holdout fraction (walk-forward)")
    parser.add_argument("--row-interval", type=int, default=15, help="Seconds between decision rows per slot")
    parser.add_argument("--version", default="logreg_v1", help="Model version string")
    return parser.parse_args()


def _load_btc(args, markets_df: pd.DataFrame) -> pd.DataFrame:
    """Load or fetch BTC prices covering the market slot range."""
    if args.btc:
        return load_btc_prices(args.btc)

    # Try cached file
    cache_path = os.path.join("data", "btc_1m_cache.csv")
    start_ts = int(markets_df["slot_ts"].min()) - 300  # 5 min before first slot
    end_ts = int(markets_df["slot_ts"].max()) + 600    # 5 min after last slot

    if os.path.exists(cache_path):
        df = load_btc_prices(cache_path)
        if not df.empty and df["ts"].min() <= start_ts and df["ts"].max() >= end_ts:
            print(f"Using cached BTC data from {cache_path} ({len(df)} rows)")
            return df

    if not args.fetch_btc:
        print(
            "No BTC price data found. Use --btc <file> or --fetch-btc to download from Coinbase.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching BTC 1m candles from Coinbase ({start_ts} to {end_ts})...")
    loader = DataLoader()
    candles = loader._fetch_coinbase_1m(start_ts, end_ts)
    if candles.empty:
        print("Failed to fetch BTC candles from Coinbase.", file=sys.stderr)
        sys.exit(1)

    # Convert OHLC to (ts, price) format and cache
    btc_df = pd.DataFrame({
        "ts": candles.index.astype(np.int64) // 10**9,
        "price": candles["close"].astype(float),
    }).sort_values("ts").reset_index(drop=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    btc_df.to_csv(cache_path, index=False)
    print(f"Cached {len(btc_df)} BTC candles to {cache_path}")
    return btc_df


def main() -> None:
    args = _parse_args()

    # Load resolved markets
    markets_df = load_market_history(args.markets)
    valid_markets = markets_df[markets_df["outcome"].isin(["Up", "Down"])]
    print(f"Loaded {len(valid_markets)} resolved markets from {args.markets}")

    if len(valid_markets) < 20:
        print("Too few resolved markets to train. Need at least 20.", file=sys.stderr)
        sys.exit(1)

    # Load BTC prices
    btc_df = _load_btc(args, valid_markets)
    print(f"BTC prices: {len(btc_df)} rows")

    # Build decision dataset
    print(f"Building decision dataset (interval={args.row_interval}s)...")
    dataset = build_btc_decision_dataset(
        valid_markets, btc_df, row_interval_sec=args.row_interval,
    )
    if dataset.empty:
        print("Dataset is empty — BTC prices may not overlap with market slots.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset: {len(dataset)} rows, {dataset['contract_id'].nunique()} contracts")

    # Check feature columns exist
    missing = [f for f in LR_FEATURES if f not in dataset.columns]
    if missing:
        print(f"Missing features in dataset: {missing}", file=sys.stderr)
        sys.exit(1)

    X = dataset[LR_FEATURES].to_numpy(dtype=float)
    y = dataset["target_up"].to_numpy(dtype=int)

    # Walk-forward split by contract_id (not random)
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
    print("Training logistic regression...")
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
    print(f"Valid base rate: {y_valid.mean():.3f}")

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
