"""
Live intrabar snapshot collector for crypto 5-min Up/Down markets.

Records per-snapshot ML features to data/snapshots.jsonl at configurable
cadence (default 5s).  Runs standalone alongside or without the trading bot.

Each snapshot record contains:
  slot_ts, snapshot_ts, strike, strike_source, btc_now, btc_source,
  yes_bid, yes_ask, yes_mid, yes_spread, yes_imbalance,
  no_bid, no_ask, no_mid, no_spread, no_imbalance,
  realized_vol_30s, realized_vol_60s

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.engine.slot_state import SlotContext
from src.models.feature_builder import _realized_vol
from src.utils.crypto_feed import CryptoPriceFeed
from src.utils.chainlink_feed import ChainlinkFeed

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

_TOP_N_LEVELS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BookSummary:
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    imbalance: Optional[float]


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
        # CLOB returns bids descending, asks ascending — take top_n each side
        top_bids = bids[: self._top_n]
        top_asks = asks[: self._top_n]

        if not top_bids and not top_asks:
            return None

        best_bid = float(top_bids[0]["price"]) if top_bids else None
        best_ask = float(top_asks[0]["price"]) if top_asks else None
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None

        bid_sz = sum(float(b.get("size", 0)) for b in top_bids)
        ask_sz = sum(float(a.get("size", 0)) for a in top_asks)
        denom = bid_sz + ask_sz
        imbalance = (bid_sz - ask_sz) / denom if denom > 0 else None

        return BookSummary(
            bid=best_bid, ask=best_ask, mid=mid, spread=spread, imbalance=imbalance
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
                slot_ts = SlotContext.slot_for(now)

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

        # Strike: Chainlink slot-open for this specific slot
        strike: Optional[float] = None
        strike_src = "unknown"
        slot_open = self._chainlink_feed.get_slot_open_price()
        if slot_open is not None and slot_open.slot_ts == slot_ts:
            strike = slot_open.price
            strike_src = "chainlink"

        vol_30s = _realized_vol(price_history, snapshot_ts, 30) if len(price_history) >= 3 else None
        vol_60s = _realized_vol(price_history, snapshot_ts, 60) if len(price_history) >= 3 else None

        record = {
            "slot_ts":          slot_ts,
            "snapshot_ts":      snapshot_ts,
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
            # NO (Down) book
            "no_bid":           no_book.bid        if no_book else None,
            "no_ask":           no_book.ask        if no_book else None,
            "no_mid":           no_book.mid        if no_book else None,
            "no_spread":        no_book.spread     if no_book else None,
            "no_imbalance":     no_book.imbalance  if no_book else None,
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
