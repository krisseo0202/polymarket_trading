"""
Live intrabar snapshot collector for crypto 5-min Up/Down markets.

Records per-snapshot ML features to data/snapshots.jsonl at configurable
cadence (default 5s).  Runs standalone alongside or without the trading bot.

Each snapshot record contains:
  slot_ts, snapshot_ts, up_token, down_token,
  strike, strike_source, btc_now, btc_source,
  yes_bid, yes_ask, yes_mid, yes_spread, yes_imbalance,
  yes_bids, yes_asks (full depth dicts {price: size}),
  yes_bid_depth, yes_ask_depth, yes_n_levels,
  no_bid, no_ask, no_mid, no_spread, no_imbalance,
  no_bids, no_asks (full depth dicts {price: size}),
  no_bid_depth, no_ask_depth, no_n_levels,
  realized_vol_30s, realized_vol_60s

  bid/ask are VWAP of the top-N levels by volume, not raw top-of-book
  (which is always 0.01/0.99 dust on these markets).

At each slot rollover an outcome sentinel is appended:
  {"type": "outcome", "slot_ts": <N>, "outcome": "Up"|"Down"}

ML pipeline joins snapshot records to outcome on slot_ts.

Usage:
    python scripts/collect_snapshots.py
    python scripts/collect_snapshots.py --interval 10 --output data/snapshots.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import only lightweight utils — avoid src.engine/__init__.py and src.models/__init__.py
# which eagerly load the full trading stack (strategies, indicators, XGB models).
# SlotContext.slot_for() and _realized_vol() are inlined below to break the chain.
from src.utils.crypto_feed import CryptoPriceFeed
from src.utils.chainlink_feed import ChainlinkFeed

import math
import numpy as np
from typing import Sequence

_SLOT_INTERVAL_S = 300


def _slot_for(ts: float) -> int:
    """Return the slot-start timestamp for a given unix timestamp."""
    return int(math.floor(ts / _SLOT_INTERVAL_S) * _SLOT_INTERVAL_S)


def _realized_vol(prices: Sequence, now_ts: float, window_s: int) -> float:
    """Realized volatility (std of log-returns) over a rolling window."""
    cutoff = now_ts - window_s
    window = [float(entry[1]) for entry in prices if float(entry[0]) >= cutoff]
    if len(window) < 3:
        return 0.0
    arr = np.asarray(window, dtype=float)
    returns = np.diff(np.log(arr))
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1))

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

_TOP_N_LEVELS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BookSummary:
    # Volume-weighted top-N prices (the real market, not 0.01/0.99 dust)
    bid: Optional[float]       # VWAP of top-N bid levels
    ask: Optional[float]       # VWAP of top-N ask levels
    mid: Optional[float]       # (bid + ask) / 2
    spread: Optional[float]    # ask - bid
    imbalance: Optional[float] # (bid_depth - ask_depth) / total at top-N
    # Full book depth
    bids: Optional[Dict[str, str]] = None  # {price: size}
    asks: Optional[Dict[str, str]] = None  # {price: size}
    bid_depth: Optional[float] = None      # total bid size
    ask_depth: Optional[float] = None      # total ask size
    n_bid_levels: int = 0
    n_ask_levels: int = 0


def _parse_json_field(raw) -> list:
    """Parse a field that may be a JSON-encoded string or already a list."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    return raw or []


def _determine_outcome(outcome_prices: list, closed: bool) -> Optional[str]:
    if not closed or len(outcome_prices) < 2:
        return None
    try:
        if float(outcome_prices[0]) > 0.9:
            return "Up"
        if float(outcome_prices[1]) > 0.9:
            return "Down"
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# MarketDiscovery
# ---------------------------------------------------------------------------

class MarketDiscovery:
    """Slot-keyed cached wrapper around the Gamma API market lookup."""

    def __init__(
        self,
        slug_prefix: str = "btc-updown-5m",
        gamma_url: str = GAMMA_API,
        timeout: int = 15,
        logger: Optional[logging.Logger] = None,
    ):
        self._slug_prefix = slug_prefix
        self._gamma_url = gamma_url
        self._timeout = timeout
        self._logger = logger or logging.getLogger("market_discovery")
        self._cached_slot_ts: Optional[int] = None
        self._cached_market: Optional[Dict] = None

    def get_current_market(self, slot_ts: int) -> Optional[Dict]:
        """Return market metadata for slot_ts, using cache if slot hasn't changed."""
        if self._cached_slot_ts == slot_ts and self._cached_market is not None:
            return self._cached_market
        market = self._fetch(slot_ts)
        self._cached_slot_ts = slot_ts
        self._cached_market = market
        return market

    def get_outcome_for_slot(self, slot_ts: int) -> Optional[str]:
        """Force-fetch outcome for a (just-ended) slot. Not cached."""
        market = self._fetch(slot_ts)
        if market is None:
            return None
        return market.get("outcome")

    def invalidate(self) -> None:
        self._cached_slot_ts = None
        self._cached_market = None

    def _fetch(self, slot_ts: int) -> Optional[Dict]:
        slug = f"{self._slug_prefix}-{slot_ts}"
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{self._gamma_url}/events",
                    params={"slug": slug},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                events = resp.json()
                if not events:
                    return None
                event = events[0]
                markets = event.get("markets", [])
                if not markets:
                    return None
                market = markets[0]

                outcome_prices = _parse_json_field(market.get("outcomePrices", ""))
                clob_token_ids = _parse_json_field(market.get("clobTokenIds", ""))
                closed = market.get("closed", False)
                outcome = _determine_outcome(outcome_prices, closed)

                return {
                    "slot_ts": slot_ts,
                    "up_token":   clob_token_ids[0] if len(clob_token_ids) > 0 else None,
                    "down_token": clob_token_ids[1] if len(clob_token_ids) > 1 else None,
                    "question":   market.get("question", event.get("title", "")),
                    "closed":     closed,
                    "outcome":    outcome,
                }
            except requests.RequestException as exc:
                if attempt < 2:
                    time.sleep(2.0)
                else:
                    self._logger.warning(f"Gamma API failed for slot {slot_ts}: {exc}")
        return None


# ---------------------------------------------------------------------------
# BookFetcher
# ---------------------------------------------------------------------------

class BookFetcher:
    """Fetches full CLOB orderbook for one or two tokens and computes top-N imbalance."""

    def __init__(
        self,
        clob_url: str = CLOB_API,
        top_n: int = _TOP_N_LEVELS,
        timeout: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self._clob_url = clob_url
        self._top_n = top_n
        self._timeout = timeout
        self._logger = logger or logging.getLogger("book_fetcher")

    def fetch(self, token_id: str) -> Optional[BookSummary]:
        try:
            resp = requests.get(
                f"{self._clob_url}/book",
                params={"token_id": token_id},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._logger.debug(f"Book fetch failed for {token_id[:12]}: {exc}")
            return None

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids and not asks:
            return None

        # Build full dict + total depth in one pass, then VWAP from sorted top-N
        def _process_side(levels: list, top_n: int, reverse: bool = False):
            full = {}
            sz_total = 0.0
            for entry in levels:
                full[entry["price"]] = entry["size"]
                sz_total += float(entry.get("size", 0))
            # Sort by price: bids descending (best=highest), asks ascending (best=lowest)
            sorted_prices = sorted(full.keys(), key=float, reverse=reverse)
            vwap_num = 0.0
            vwap_den = 0.0
            for p in sorted_prices[:top_n]:
                sz = float(full[p])
                vwap_num += float(p) * sz
                vwap_den += sz
            vwap = vwap_num / vwap_den if vwap_den > 0 else None
            return full, sz_total, vwap, vwap_den

        # Bids: best = highest price → reverse=True
        # Asks: best = lowest price  → reverse=False
        full_bids, bid_depth, vwap_bid, bid_top_sz = _process_side(bids, self._top_n, reverse=True)
        full_asks, ask_depth, vwap_ask, ask_top_sz = _process_side(asks, self._top_n, reverse=False)

        mid = (vwap_bid + vwap_ask) / 2 if vwap_bid is not None and vwap_ask is not None else None
        spread = (vwap_ask - vwap_bid) if vwap_bid is not None and vwap_ask is not None else None
        denom = bid_top_sz + ask_top_sz
        imbalance = (bid_top_sz - ask_top_sz) / denom if denom > 0 else None

        return BookSummary(
            bid=vwap_bid, ask=vwap_ask, mid=mid, spread=spread, imbalance=imbalance,
            bids=full_bids, asks=full_asks,
            bid_depth=bid_depth, ask_depth=ask_depth,
            n_bid_levels=len(bids), n_ask_levels=len(asks),
        )

    def fetch_pair(
        self, yes_token: str, no_token: str
    ) -> Tuple[Optional[BookSummary], Optional[BookSummary]]:
        """Fetch both tokens in parallel."""
        yes_result: Optional[BookSummary] = None
        no_result: Optional[BookSummary] = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_yes = pool.submit(self.fetch, yes_token)
            f_no  = pool.submit(self.fetch, no_token)
            try:
                yes_result = f_yes.result()
            except Exception:
                pass
            try:
                no_result = f_no.result()
            except Exception:
                pass
        return yes_result, no_result


# ---------------------------------------------------------------------------
# SnapshotLogger
# ---------------------------------------------------------------------------

class SnapshotLogger:
    """Append-only JSONL writer for snapshot and outcome sentinel records."""

    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def write_snapshot(self, record: dict) -> None:
        line = json.dumps(record, default=str)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def write_outcome_sentinel(self, slot_ts: int, outcome: str) -> None:
        record = {"type": "outcome", "slot_ts": slot_ts, "outcome": outcome}
        self.write_snapshot(record)


# ---------------------------------------------------------------------------
# SnapshotCollector
# ---------------------------------------------------------------------------

class SnapshotCollector:
    """Main loop: discover market → fetch book → log snapshot → handle rollover."""

    def __init__(self, config: dict):
        self._interval = config.get("snapshot_interval_s", 5)
        self._logger = config.get("logger") or logging.getLogger("snapshot_collector")

        slug_prefix = config.get("slug_prefix", "btc-updown-5m")
        price_symbol = config.get("price_symbol", "BTC-USD")
        price_exchange = config.get("price_exchange", "coinbase")
        chainlink_symbol = config.get("chainlink_symbol", "btc/usd")

        self._price_feed = CryptoPriceFeed(
            symbol=price_symbol, exchange=price_exchange, logger=self._logger,
        )
        self._chainlink_feed = ChainlinkFeed(
            symbol=chainlink_symbol, logger=self._logger,
        )
        self._discovery = MarketDiscovery(slug_prefix=slug_prefix, logger=self._logger)
        self._book_fetcher = BookFetcher(logger=self._logger)
        self._snap_logger = SnapshotLogger(config.get("output", "data/snapshots.jsonl"))

        self._stop_evt = threading.Event()
        self._last_slot_ts: Optional[int] = None

    def run(self) -> None:
        self._price_feed.start()
        self._chainlink_feed.start()
        self._wait_for_feeds()

        self._logger.info("Snapshot collector started")
        try:
            while not self._stop_evt.is_set():
                now = time.time()
                slot_ts = _slot_for(now)

                if self._last_slot_ts is not None and slot_ts != self._last_slot_ts:
                    self._on_slot_rollover(self._last_slot_ts)

                self._last_slot_ts = slot_ts

                market = self._discovery.get_current_market(slot_ts)
                if market is None:
                    self._logger.debug(f"No market found for slot {slot_ts}, skipping snapshot")
                else:
                    self._collect_snapshot(slot_ts, market)

                self._stop_evt.wait(timeout=self._interval)
        finally:
            self._price_feed.stop()
            self._chainlink_feed.stop()
            self._logger.info("Snapshot collector stopped")

    def stop(self) -> None:
        self._stop_evt.set()

    def _wait_for_feeds(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            btc_ready = self._price_feed.get_latest_mid() is not None
            cl_ready  = self._chainlink_feed.get_latest() is not None
            if btc_ready and cl_ready:
                self._logger.info("Feeds ready")
                return
            time.sleep(0.5)
        if self._price_feed.get_latest_mid() is None:
            self._logger.warning("Price feed not ready after startup wait")
        if self._chainlink_feed.get_latest() is None:
            self._logger.warning("Chainlink feed not ready after startup wait")

    def _collect_snapshot(self, slot_ts: int, market: dict) -> None:
        up_token   = market.get("up_token")
        down_token = market.get("down_token")
        if not up_token or not down_token:
            self._logger.debug("Market has no token IDs, skipping")
            return

        snapshot_ts = time.time()

        yes_book, no_book = self._book_fetcher.fetch_pair(up_token, down_token)

        price_now  = self._price_feed.get_latest_mid()
        price_src  = self._price_feed.exchange if price_now is not None else None
        price_history = self._price_feed.get_recent_prices(300)

        # If exchange feed unavailable, try Chainlink latest
        if price_now is None:
            cl = self._chainlink_feed.get_latest()
            if cl is not None:
                price_now = cl.price
                price_src = "chainlink"

        # Strike: Chainlink slot-open for this specific slot. Prefer the
        # carry-forward getter — Chainlink only pushes on ≥0.5% deviation, so
        # calm slots receive zero ticks and the last-known price IS the
        # slot-open price (by contract). Without this, 1%+ of snapshots would
        # drop strike and be filtered from training.
        strike: Optional[float] = None
        strike_src = "unknown"
        slot_open = self._chainlink_feed.get_slot_open_or_carry_forward(slot_ts)
        if slot_open is not None and slot_open.slot_ts == slot_ts:
            strike = slot_open.price
            fresh = self._chainlink_feed.get_slot_open_price()
            if fresh is not None and fresh.slot_ts == slot_ts:
                strike_src = "chainlink"
            else:
                strike_src = "chainlink_carryforward"

        vol_30s = _realized_vol(price_history, snapshot_ts, 30) if len(price_history) >= 3 else None
        vol_60s = _realized_vol(price_history, snapshot_ts, 60) if len(price_history) >= 3 else None

        record = {
            "slot_ts":          slot_ts,
            "snapshot_ts":      snapshot_ts,
            "up_token":         up_token,
            "down_token":       down_token,
            "strike":           strike,
            "strike_source":    strike_src,
            "btc_now":          price_now,
            "btc_source":       price_src,
            # YES (Up) book
            "yes_bid":          yes_book.bid       if yes_book else None,
            "yes_ask":          yes_book.ask       if yes_book else None,
            "yes_mid":          yes_book.mid       if yes_book else None,
            "yes_spread":       yes_book.spread    if yes_book else None,
            "yes_imbalance":    yes_book.imbalance if yes_book else None,
            "yes_bids":         yes_book.bids      if yes_book else None,
            "yes_asks":         yes_book.asks      if yes_book else None,
            "yes_bid_depth":    yes_book.bid_depth if yes_book else None,
            "yes_ask_depth":    yes_book.ask_depth if yes_book else None,
            "yes_n_levels":     (yes_book.n_bid_levels + yes_book.n_ask_levels) if yes_book else 0,
            # NO (Down) book
            "no_bid":           no_book.bid        if no_book else None,
            "no_ask":           no_book.ask        if no_book else None,
            "no_mid":           no_book.mid        if no_book else None,
            "no_spread":        no_book.spread     if no_book else None,
            "no_imbalance":     no_book.imbalance  if no_book else None,
            "no_bids":          no_book.bids       if no_book else None,
            "no_asks":          no_book.asks       if no_book else None,
            "no_bid_depth":     no_book.bid_depth  if no_book else None,
            "no_ask_depth":     no_book.ask_depth  if no_book else None,
            "no_n_levels":      (no_book.n_bid_levels + no_book.n_ask_levels) if no_book else 0,
            # BTC realized vol
            "realized_vol_30s": vol_30s,
            "realized_vol_60s": vol_60s,
        }
        self._snap_logger.write_snapshot(record)

    def _on_slot_rollover(self, ended_slot_ts: int) -> None:
        self._logger.info(f"Slot {ended_slot_ts} ended — querying outcome")
        self._discovery.invalidate()
        # Small wait for Gamma API to reflect the closed market
        time.sleep(3.0)
        outcome = self._discovery.get_outcome_for_slot(ended_slot_ts)
        if outcome:
            self._snap_logger.write_outcome_sentinel(ended_slot_ts, outcome)
            self._logger.info(f"Outcome sentinel: slot={ended_slot_ts} outcome={outcome}")
        else:
            self._logger.warning(
                f"Outcome not yet available for slot {ended_slot_ts} — "
                "join from btc_updown_5m.csv as fallback"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_asset_config(asset: str) -> dict:
    """Load per-asset config from config/config.yaml, with sensible BTC defaults."""
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        return raw.get("assets", {}).get(asset.upper(), {})
    except Exception:
        return {}


_ASSET_DEFAULTS = {
    "BTC":  {"slug_prefix": "btc-updown-5m",  "price_symbol": "BTC-USD",  "chainlink_symbol": "btc/usd"},
    "ETH":  {"slug_prefix": "eth-updown-5m",  "price_symbol": "ETH-USD",  "chainlink_symbol": "eth/usd"},
    "SOL":  {"slug_prefix": "sol-updown-5m",  "price_symbol": "SOL-USD",  "chainlink_symbol": "sol/usd"},
    "DOGE": {"slug_prefix": "doge-updown-5m", "price_symbol": "DOGE-USD", "chainlink_symbol": "doge/usd"},
    "XRP":  {"slug_prefix": "xrp-updown-5m",  "price_symbol": "XRP-USD",  "chainlink_symbol": "xrp/usd"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live intrabar snapshot collector")
    parser.add_argument("--asset", default="BTC",
                        help="Asset to collect: BTC, ETH, SOL, DOGE, XRP (default: BTC)")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between snapshots")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: data/snapshots_{asset}.jsonl)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    asset = args.asset.upper()
    output = args.output or f"data/snapshots_{asset.lower()}.jsonl"

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger(f"collect_snapshots.{asset.lower()}")

    # Merge config file + hardcoded defaults
    asset_cfg = {**_ASSET_DEFAULTS.get(asset, _ASSET_DEFAULTS["BTC"]), **_load_asset_config(asset)}

    config = {
        "snapshot_interval_s": args.interval,
        "output": output,
        "logger": logger,
        "slug_prefix": asset_cfg.get("slug_prefix", f"{asset.lower()}-updown-5m"),
        "price_symbol": asset_cfg.get("price_symbol", f"{asset}-USD"),
        "price_exchange": asset_cfg.get("price_exchange", "coinbase"),
        "chainlink_symbol": asset_cfg.get("chainlink_symbol", f"{asset.lower()}/usd"),
    }
    collector = SnapshotCollector(config)

    def _handle_signal(signum, frame):
        logger.info("Shutdown requested")
        collector.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    collector.run()


if __name__ == "__main__":
    main()
