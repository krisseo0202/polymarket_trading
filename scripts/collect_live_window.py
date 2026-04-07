#!/usr/bin/env python3
"""
Continuously capture Polymarket orderbook snapshots for each active
BTC Up/Down 5-minute market window.

Appends rows to a CSV with columns:
  slot_ts, elapsed_s, side, best_bid, best_ask, bid_depth_3, ask_depth_3, spread, mid

Usage:
  python scripts/collect_live_window.py
  python scripts/collect_live_window.py --output data/my_snapshots.csv --interval 3
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

import requests

GAMMA_API = "https://gamma-api.polymarket.com/events"
CLOB_BOOK = "https://clob.polymarket.com/book"

CSV_COLUMNS = [
    "slot_ts",
    "elapsed_s",
    "side",
    "best_bid",
    "best_ask",
    "bid_depth_3",
    "ask_depth_3",
    "spread",
    "mid",
]

_shutdown = False


def _handle_sigint(sig, frame):
    global _shutdown
    print("\nShutting down gracefully...")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_sigint)


def current_slot_ts() -> int:
    """Return the start-of-slot unix timestamp for the current 5-min window."""
    now = int(time.time())
    return now - (now % 300)


def discover_market(slot_ts: int):
    """
    Query Gamma API for the BTC Up/Down 5-min market at the given slot.

    Returns dict mapping "up" and "down" to their CLOB token IDs,
    or None if the market is not found.
    """
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        resp = requests.get(GAMMA_API, params={"slug": slug}, timeout=10)
        resp.raise_for_status()
        events = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  [warn] Gamma API error for {slug}: {exc}")
        return None

    if not events:
        print(f"  [info] No market found for slug={slug}")
        return None

    event = events[0]
    markets = event.get("markets", [])
    if not markets:
        print(f"  [info] Event has no markets for slug={slug}")
        return None

    market = markets[0]
    try:
        token_ids = json.loads(market["clobTokenIds"])
        outcomes = json.loads(market["outcomes"])
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"  [warn] Failed to parse market tokens: {exc}")
        return None

    token_map = {}
    for outcome, tid in zip(outcomes, token_ids):
        token_map[outcome.lower()] = tid

    if "up" not in token_map or "down" not in token_map:
        print(f"  [warn] Unexpected outcomes: {outcomes}")
        return None

    return token_map


def fetch_book(token_id: str):
    """
    Fetch the CLOB orderbook for a token.

    Returns (bids, asks) where each is a list of (price, size) tuples,
    sorted best-first (bids descending, asks ascending).
    Returns ([], []) on error.
    """
    try:
        resp = requests.get(CLOB_BOOK, params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  [warn] Book fetch error: {exc}")
        return [], []

    bids = []
    for entry in data.get("bids", []):
        price = float(entry.get("price") or entry.get("p", 0))
        size = float(entry.get("size") or entry.get("s", 0))
        bids.append((price, size))
    bids.sort(key=lambda x: x[0], reverse=True)

    asks = []
    for entry in data.get("asks", []):
        price = float(entry.get("price") or entry.get("p", 0))
        size = float(entry.get("size") or entry.get("s", 0))
        asks.append((price, size))
    asks.sort(key=lambda x: x[0])

    return bids, asks


def extract_snapshot(side: str, bids, asks, slot_ts: int, elapsed_s: int) -> dict:
    """Build a single CSV row dict from orderbook data."""
    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    bid_depth_3 = sum(s for _, s in bids[:3])
    ask_depth_3 = sum(s for _, s in asks[:3])
    spread = round(best_ask - best_bid, 6) if (bids and asks) else 0.0
    mid = round((best_bid + best_ask) / 2, 6) if (bids and asks) else 0.0

    return {
        "slot_ts": slot_ts,
        "elapsed_s": elapsed_s,
        "side": side,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth_3": round(bid_depth_3, 4),
        "ask_depth_3": round(ask_depth_3, 4),
        "spread": spread,
        "mid": mid,
    }


def ensure_csv_header(path: Path):
    """Write CSV header if the file doesn't exist or is empty."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_rows(path: Path, rows: list[dict]):
    """Append snapshot rows to the CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        for row in rows:
            writer.writerow(row)


def run(output: str, interval: float):
    output_path = Path(output)
    ensure_csv_header(output_path)

    print(f"Collecting orderbook snapshots -> {output_path.resolve()}")
    print(f"Poll interval: {interval}s")
    print("Press Ctrl+C to stop.\n")

    last_slot = None
    token_map = None

    while not _shutdown:
        slot_ts = current_slot_ts()

        # New slot -- discover market
        if slot_ts != last_slot:
            print(f"--- Slot {slot_ts} ---")
            token_map = discover_market(slot_ts)
            last_slot = slot_ts
            if token_map is None:
                print("  Skipping slot (market not found). Waiting for next slot...")
                next_slot = slot_ts + 300
                while time.time() < next_slot and not _shutdown:
                    time.sleep(1)
                continue
            print(f"  Up token:   {token_map['up'][:16]}...")
            print(f"  Down token: {token_map['down'][:16]}...")

        if token_map is None:
            time.sleep(1)
            continue

        # Check if still within current slot
        now = time.time()
        elapsed_s = int(now - slot_ts)
        if elapsed_s >= 300:
            # Slot ended, loop will pick up the next one
            continue

        # Fetch books for both sides
        rows = []
        for side in ("up", "down"):
            bids, asks = fetch_book(token_map[side])
            row = extract_snapshot(side, bids, asks, slot_ts, elapsed_s)
            rows.append(row)

        append_rows(output_path, rows)

        up_row = rows[0]
        down_row = rows[1]
        print(
            f"  t+{elapsed_s:>3}s  "
            f"Up mid={up_row['mid']:.4f} spd={up_row['spread']:.4f}  "
            f"Dn mid={down_row['mid']:.4f} spd={down_row['spread']:.4f}"
        )

        # Sleep until next poll
        sleep_until = now + interval
        while time.time() < sleep_until and not _shutdown:
            time.sleep(0.25)

    print(f"\nDone. Snapshots saved to {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect live Polymarket BTC 5-min orderbook snapshots"
    )
    parser.add_argument(
        "--output",
        default="data/live_orderbook_snapshots.csv",
        help="Output CSV path (default: data/live_orderbook_snapshots.csv)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1)",
    )
    args = parser.parse_args()
    run(args.output, args.interval)


if __name__ == "__main__":
    main()
