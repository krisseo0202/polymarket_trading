"""Sample live features into a CSV for inspection.

Connects to the same BTC feed + Polymarket order books that the bot
uses, calls ``build_live_features`` every ``--interval`` seconds for
``--duration`` seconds, and writes one CSV row per sample.

Usage::

    .venv/bin/python scripts/sample_features_live.py
    .venv/bin/python scripts/sample_features_live.py --duration 30 --interval 5
    .venv/bin/python scripts/sample_features_live.py --warmup --duration 30 --interval 3

Notes
-----
- Paper-trading mode against real market data (no orders placed).
- Without ``--warmup``, multi-TF features that need long history (60m,
  240m) start at 0.0 and gradually warm. Add ``--warmup`` to seed the
  BTC feed from Binance REST first (~10s extra startup).
- Output goes to ``data/feature_samples_<UTC ts>.csv`` by default.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.client import PolymarketClient
from src.api.types import OrderBook, OrderBookEntry
from src.models.feature_builder import build_live_features, parse_strike_price
from src.models.schema import FEATURE_COLUMNS
from src.models.slot_path_state import (
    SlotPathState,
    advance_from_snapshot,
    features_from_snapshot,
)
from src.utils.btc_feed import BtcPriceFeed
from src.utils.market_utils import find_updown_market


_SLOT_S = 300


def _setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("sample_features_live")


def _fetch_book(client: PolymarketClient, token_id: str) -> Optional[OrderBook]:
    try:
        return client.get_order_book(token_id)
    except Exception as exc:
        logging.warning("get_order_book(%s) failed: %s", token_id, exc)
        return None


def _build_one_sample(
    *,
    client: PolymarketClient,
    btc_feed: BtcPriceFeed,
    market: Dict[str, str],
    slot_state: SlotPathState,
    slot_state_ts_ref: List[int],
    btc_window_s: int,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Build one snapshot, run build_live_features, return the row dict.

    ``slot_state_ts_ref`` is a 1-elem list so we can pass-by-reference into
    ``advance_from_snapshot`` without making slot_state itself stateful
    about its own ts.
    """
    yes_book = _fetch_book(client, market["yes_token_id"])
    no_book = _fetch_book(client, market["no_token_id"])
    if yes_book is None or no_book is None:
        logger.warning("missing order book — skipping sample")
        return None

    btc_prices = btc_feed.get_recent_prices(btc_window_s)
    if not btc_prices:
        logger.warning("btc_feed has no prices yet — skipping sample")
        return None

    now_ts = float(btc_prices[-1][0])
    # The active BTC 5m slot expiry is the next 300s boundary after now.
    slot_expiry_ts = (int(now_ts // _SLOT_S) + 1) * _SLOT_S
    slot_open_ts = slot_expiry_ts - _SLOT_S

    # Strike resolution. The bot normally uses Chainlink's slot-open
    # price; for this offline-style sampler we fall back to the BTC tick
    # at slot_open (or the closest one ≤ slot_open). Same approach
    # cycle_runner.slot_state takes when Chainlink is stale.
    strike = parse_strike_price(market.get("question", "") or "")
    if strike is None:
        for ts, px, *_ in btc_prices:
            if float(ts) <= slot_open_ts:
                strike = float(px)
            else:
                break

    snapshot: Dict[str, Any] = {
        "btc_prices": btc_prices,
        "yes_book": yes_book,
        "no_book": no_book,
        "yes_history": [],
        "no_history": [],
        "question": market.get("question", ""),
        "strike_price": strike,
        "slot_expiry_ts": slot_expiry_ts,
        "now_ts": now_ts,
    }

    # Family C path features — fold ticks into the slot accumulator, then
    # query for the current slot's high/low/crosses.
    slot_state_ts_ref[0] = advance_from_snapshot(
        slot_state, slot_state_ts_ref[0], snapshot,
    )
    snapshot["slot_path_features"] = features_from_snapshot(slot_state, snapshot)

    built = build_live_features(snapshot)
    if not built.ready:
        logger.info("features not ready: %s", built.status)

    # Row = sample metadata + every feature column.
    row: Dict[str, Any] = {
        "wall_ts": now_ts,
        "wall_iso": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
        "slot_ts": slot_expiry_ts - _SLOT_S,
        "slot_expiry_ts": slot_expiry_ts,
        "tte_s": max(0.0, slot_expiry_ts - now_ts),
        "feature_status": built.status,
        "btc_prices_in_buffer": len(btc_prices),
        "question": market.get("question", "")[:80],
        "strike_price": strike or 0.0,
    }
    for col in FEATURE_COLUMNS:
        row[col] = built.features.get(col, 0.0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample live features into a CSV for inspection.",
    )
    parser.add_argument("--duration", type=int, default=30,
                        help="Total seconds to sample (default 30).")
    parser.add_argument("--interval", type=int, default=5,
                        help="Seconds between samples (default 5).")
    parser.add_argument("--warmup", action="store_true",
                        help="Seed BTC feed from Binance REST so multi-TF "
                             "features start populated. Adds ~3min startup.")
    parser.add_argument("--warmup-days", type=int, default=5,
                        help="Days of Binance 1s history to load. The 240m "
                             "TF needs ≥100h (≈5d) before its 25-bar minimum "
                             "is met. Production bot uses 3 days, which is "
                             "why long-TF features are zero in live runs.")
    parser.add_argument("--btc-symbol", default="btcusdt")
    parser.add_argument("--btc-exchange", default="binance")
    parser.add_argument("--btc-window-s", type=int, default=432000,
                        help="BTC tick window passed to build_live_features. "
                             "Default 432000s = 5d so all 7 multi-TF blocks "
                             "warm. Production uses 14400s (4h) at "
                             "logreg_edge.py:577 — known issue.")
    parser.add_argument(
        "--output", default=None,
        help="CSV path. Default: data/feature_samples_<UTC_TS>.csv",
    )
    parser.add_argument("--keywords", nargs="*", default=["Bitcoin", "Up or Down"])
    parser.add_argument("--slug-prefix", default="btc-updown-5m")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = _setup_logger(args.log_level)
    logger.info("starting BTC feed: %s on %s", args.btc_symbol, args.btc_exchange)
    btc_feed = BtcPriceFeed(
        symbol=args.btc_symbol, exchange=args.btc_exchange, logger=logger,
    ).start()

    if args.warmup:
        logger.info("warming up BTC feed from Binance REST (%d days)…", args.warmup_days)
        try:
            n = btc_feed.warmup_from_binance(days=args.warmup_days)
            logger.info("warmup loaded %d ticks", n)
        except Exception as exc:
            logger.warning("warmup failed (continuing without): %s", exc)

    # Wait briefly for the feed to receive its first ticks if no warmup.
    if not args.warmup:
        logger.info("waiting 5s for BTC feed to receive first ticks…")
        time.sleep(5.0)

    logger.info("discovering active BTC 5-min market…")
    market = find_updown_market(
        keywords=args.keywords, min_volume=0, logger=logger,
        slug_prefix=args.slug_prefix,
    )
    if not market:
        logger.error("no active market found; aborting.")
        btc_feed.stop()
        sys.exit(1)
    logger.info("market: %s", market.get("question", ""))
    logger.info("yes_token=%s no_token=%s",
                market["yes_token_id"][:12], market["no_token_id"][:12])

    client = PolymarketClient(paper_trading=True)

    out_path = Path(args.output) if args.output else (
        _ROOT / "data" / f"feature_samples_{int(time.time())}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slot_state = SlotPathState()
    slot_state_ts_ref = [0]
    rows: List[Dict[str, Any]] = []
    deadline = time.time() + args.duration

    sample_n = 0
    while time.time() < deadline:
        sample_n += 1
        loop_start = time.time()
        row = _build_one_sample(
            client=client, btc_feed=btc_feed, market=market,
            slot_state=slot_state, slot_state_ts_ref=slot_state_ts_ref,
            btc_window_s=args.btc_window_s, logger=logger,
        )
        if row:
            rows.append(row)
            logger.info(
                "sample %d  tte=%5.0fs  btc_mid=%.2f  rsi_1m=%.1f  td_5m_setup=%+d  "
                "fair_value=%.3f  status=%s",
                sample_n, row["tte_s"], row.get("btc_mid", 0.0),
                row.get("rsi_1m", 0.0), int(row.get("td_5m_setup", 0)),
                row.get("fair_value_p_up", 0.0), row["feature_status"],
            )

        # Sleep the remainder of the interval; never longer than the
        # remaining duration.
        elapsed = time.time() - loop_start
        sleep_for = max(0.0, args.interval - elapsed)
        sleep_for = min(sleep_for, max(0.0, deadline - time.time()))
        if sleep_for > 0:
            time.sleep(sleep_for)

    btc_feed.stop()

    if not rows:
        logger.error("no rows captured.")
        sys.exit(2)

    logger.info("captured %d rows; writing %s", len(rows), out_path)
    fieldnames: Sequence[str] = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("done. inspect with: head %s | csvlook  (or open in any spreadsheet)", out_path)


if __name__ == "__main__":
    main()
