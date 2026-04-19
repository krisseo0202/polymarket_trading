"""
Build a local snapshot dataset from probability tick logs and resolved market outcomes.

Usage:
    ./.venv/bin/python scripts/build_probability_snapshot_dataset.py
    ./.venv/bin/python scripts/build_probability_snapshot_dataset.py \
        --btc-file data/btc_prices.csv \
        --output data/snapshots_local.parquet \
        --warmup-days 3 \
        --enable-binance-warmup

Multi-TF warmup
---------------
`--enable-binance-warmup` pulls ≥`--warmup-days` days of 1s BTC history from
Binance REST so the multi-timeframe indicator features (rsi_*, ut_*_, td_*_)
across 1m/3m/5m/15m/30m/60m/4h are fully warmed at every snapshot. Without it,
long-TF features remain at 0.0 for snapshots whose warmup horizon exceeds the
local BTC file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.backtest import (
    build_snapshot_dataset,
    load_btc_prices,
    load_market_history,
    load_probability_ticks,
    save_snapshot_dataset,
)
from src.utils.btc_warmup import DEFAULT_WARMUP_DAYS, warmup_btc_history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local snapshot dataset from logged probability ticks")
    parser.add_argument("--market-csv", default="data/btc_updown_5m.csv", help="Resolved market history CSV")
    parser.add_argument("--prob-ticks", default="data/probability_ticks.jsonl", help="Probability tick JSONL path")
    parser.add_argument("--btc-file", default="", help="Optional local BTC price file (csv/jsonl)")
    parser.add_argument("--btc-window", type=int, default=300, help="BTC history lookback window in seconds (300s = standard backtest; bump to 86400×warmup-days when multi-TF features are needed)")
    parser.add_argument(
        "--enable-binance-warmup", action="store_true",
        help="Fetch Binance 1s history so multi-TF indicator features are warm for every snapshot.",
    )
    parser.add_argument(
        "--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS,
        help=f"Days of BTC history to guarantee before the first snapshot (default {DEFAULT_WARMUP_DAYS}).",
    )
    parser.add_argument("--output", default="data/snapshots_local.csv", help="Output dataset path (.csv or .parquet)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("build_snapshots")

    args = _parse_args()
    markets_df = load_market_history(args.market_csv)
    prob_ticks_df = load_probability_ticks(args.prob_ticks)
    btc_df = load_btc_prices(args.btc_file) if args.btc_file else None

    btc_window_s = args.btc_window
    if args.enable_binance_warmup:
        btc_df = _warmup_and_reshape(btc_df, prob_ticks_df, args.warmup_days, log)
        # The multi-TF feature computer needs the full warmup window, so stretch
        # btc_window_seconds to cover it. (Existing per-snapshot features like
        # btc_ret_60s and btc_vol_30s still read only their own short slice.)
        btc_window_s = max(btc_window_s, args.warmup_days * 86400)
        log.info("Multi-TF warmup active: btc_window_seconds bumped to %d", btc_window_s)

    dataset = build_snapshot_dataset(
        markets_df=markets_df,
        prob_ticks_df=prob_ticks_df,
        btc_df=btc_df,
        btc_window_seconds=btc_window_s,
    )
    if dataset.empty:
        raise SystemExit("No rows were built. Check your input files and slot overlap.")

    save_snapshot_dataset(dataset, args.output)
    status_counts = dataset["feature_status"].value_counts().to_dict() if "feature_status" in dataset.columns else {}
    log.info("Saved dataset: %s", args.output)
    log.info("Rows: %d", len(dataset))
    if status_counts:
        log.info("Feature status counts: %s", status_counts)


def _warmup_and_reshape(
    btc_df,
    prob_ticks_df: pd.DataFrame,
    warmup_days: int,
    log: logging.Logger,
):
    """Expand BTC coverage to ≥warmup_days before the earliest snapshot.

    Converts the warmed-up OHLCV frame to the (ts, close)-style frame that
    ``build_snapshot_dataset`` expects (columns: ``ts, price``).
    """
    if prob_ticks_df.empty:
        return btc_df

    first_snapshot_ts = float(prob_ticks_df["ts"].min())
    last_snapshot_ts = float(prob_ticks_df["ts"].max())
    need_start_ts = first_snapshot_ts - warmup_days * 86400

    # build_snapshot_dataset expects (ts, price) columns. The warmup helper
    # returns OHLCV. Convert both the existing local df (if any) and the
    # warmup output into the OHLCV schema so they can be merged by the helper.
    existing_ohlcv = _to_ohlcv(btc_df)
    warmed = warmup_btc_history(
        existing=existing_ohlcv,
        need_start_ts=need_start_ts,
        need_end_ts=last_snapshot_ts,
        logger=log,
    )
    log.info(
        "BTC warmup: %d → %d bars (need %s..%s)",
        0 if existing_ohlcv is None else len(existing_ohlcv),
        len(warmed), need_start_ts, last_snapshot_ts,
    )

    # Reshape back into (ts, price) for the existing dataset builder.
    return pd.DataFrame({"ts": warmed["timestamp"], "price": warmed["close"]})


def _to_ohlcv(btc_df):
    """Promote a (ts, price) frame into an OHLCV frame so warmup_btc_history
    (which merges on the OHLCV schema) can work with it."""
    if btc_df is None or btc_df.empty:
        return None
    return pd.DataFrame({
        "timestamp": btc_df["ts"].astype("int64"),
        "open": btc_df["price"].astype(float),
        "high": btc_df["price"].astype(float),
        "low": btc_df["price"].astype(float),
        "close": btc_df["price"].astype(float),
        "volume": 1.0,
    })


if __name__ == "__main__":
    main()
