"""
Build a local snapshot dataset from probability tick logs and resolved market outcomes.

Usage:
    ./.venv/bin/python scripts/build_probability_snapshot_dataset.py
    ./.venv/bin/python scripts/build_probability_snapshot_dataset.py \
        --btc-file data/btc_prices.csv \
        --output data/snapshots_local.parquet
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backtest import (
    build_snapshot_dataset,
    load_btc_prices,
    load_market_history,
    load_probability_ticks,
    save_snapshot_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local snapshot dataset from logged probability ticks")
    parser.add_argument("--market-csv", default="data/btc_updown_5m.csv", help="Resolved market history CSV")
    parser.add_argument("--prob-ticks", default="data/probability_ticks.jsonl", help="Probability tick JSONL path")
    parser.add_argument("--btc-file", default="", help="Optional local BTC price file (csv/jsonl)")
    parser.add_argument("--btc-window", type=int, default=300, help="BTC history lookback window in seconds")
    parser.add_argument("--output", default="data/snapshots_local.csv", help="Output dataset path (.csv or .parquet)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    markets_df = load_market_history(args.market_csv)
    prob_ticks_df = load_probability_ticks(args.prob_ticks)
    btc_df = load_btc_prices(args.btc_file) if args.btc_file else None

    dataset = build_snapshot_dataset(
        markets_df=markets_df,
        prob_ticks_df=prob_ticks_df,
        btc_df=btc_df,
        btc_window_seconds=args.btc_window,
    )
    if dataset.empty:
        raise SystemExit("No rows were built. Check your input files and slot overlap.")

    save_snapshot_dataset(dataset, args.output)
    status_counts = dataset["feature_status"].value_counts().to_dict() if "feature_status" in dataset.columns else {}
    print(f"Saved dataset: {args.output}")
    print(f"Rows: {len(dataset)}")
    if status_counts:
        print(f"Feature status counts: {status_counts}")


if __name__ == "__main__":
    main()
