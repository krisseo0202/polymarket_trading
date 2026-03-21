"""
Collect historical BTC 5-min Up/Down market data from Polymarket.

Usage:
    python scripts/collect_history.py --hours 24
    python scripts/collect_history.py --hours 168 --output data/btc_updown_5m.csv
    python scripts/collect_history.py --hours 24 --ticks   # also collect intra-market prices
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

SLOT_INTERVAL = 300  # 5 minutes in seconds


def current_slot_ts() -> int:
    """Return the most recent 5-minute boundary timestamp."""
    now = int(time.time())
    return now - (now % SLOT_INTERVAL)


def fetch_market_for_slot(slot_ts: int) -> Optional[Dict]:
    """Fetch market data for a given 5-minute slot from the gamma API.

    Returns dict with market metadata, or None if not found.
    """
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"slug": slug},
            timeout=15,
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

        # Parse outcome prices
        outcome_prices_raw = market.get("outcomePrices", "")
        if isinstance(outcome_prices_raw, str):
            # Format: '["0.505","0.495"]' or already a list
            try:
                import json
                outcome_prices = json.loads(outcome_prices_raw)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = []
        else:
            outcome_prices = outcome_prices_raw

        # Parse token IDs
        clob_token_ids_raw = market.get("clobTokenIds", "")
        if isinstance(clob_token_ids_raw, str):
            try:
                import json
                clob_token_ids = json.loads(clob_token_ids_raw)
            except (json.JSONDecodeError, TypeError):
                clob_token_ids = []
        else:
            clob_token_ids = clob_token_ids_raw or []

        # Determine outcome (Up or Down won)
        closed = market.get("closed", False)
        outcome = None
        if closed and len(outcome_prices) >= 2:
            try:
                up_final = float(outcome_prices[0])
                down_final = float(outcome_prices[1])
                if up_final > 0.9:
                    outcome = "Up"
                elif down_final > 0.9:
                    outcome = "Down"
            except (ValueError, TypeError):
                pass

        up_token = clob_token_ids[0] if len(clob_token_ids) > 0 else ""
        down_token = clob_token_ids[1] if len(clob_token_ids) > 1 else ""

        return {
            "slot_ts": slot_ts,
            "slot_utc": datetime.fromtimestamp(slot_ts, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "question": market.get("question", event.get("title", "")),
            "up_token": up_token,
            "down_token": down_token,
            "outcome": outcome,
            "volume": float(market.get("volume", 0)),
            "outcome_prices": outcome_prices,
            "closed": closed,
        }
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch slot {slot_ts}: {e}", file=sys.stderr)
        return None


def fetch_price_history(token_id: str, slot_ts: int) -> List[Dict]:
    """Fetch price history for a token, filtered to the 5-min window.

    Returns list of {t: unix_ts, p: price}.
    """
    if not token_id:
        return []

    try:
        resp = requests.get(
            f"{CLOB_API}/prices-history",
            params={"market": token_id, "interval": "max"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history", [])

        # Filter to entries within the 5-min window
        slot_start = slot_ts
        slot_end = slot_ts + SLOT_INTERVAL
        return [
            {"t": int(entry["t"]), "p": float(entry["p"])}
            for entry in history
            if slot_start <= int(entry["t"]) <= slot_end
        ]
    except requests.RequestException:
        return []


def parse_strike_price(question: str) -> Optional[float]:
    """Extract strike price from question text like 'Will BTC > $84,500 ...'."""
    m = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)", question)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def collect(hours_back: int, output_path: str, collect_ticks: bool = False):
    """Main collection loop: iterate slots backwards and save to CSV."""
    now_slot = current_slot_ts()
    # Skip the current (potentially unclosed) slot
    start_slot = now_slot - SLOT_INTERVAL
    total_slots = (hours_back * 3600) // SLOT_INTERVAL
    end_slot = start_slot - (total_slots * SLOT_INTERVAL)

    print(f"Collecting {total_slots} slots ({hours_back}h) from "
          f"{datetime.fromtimestamp(end_slot + SLOT_INTERVAL, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} to "
          f"{datetime.fromtimestamp(start_slot, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")

    # Prepare output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "slot_ts", "slot_utc", "question", "up_token", "down_token",
        "outcome", "volume", "up_price_start", "up_price_end",
        "down_price_start", "down_price_end", "strike_price",
    ]

    ticks_path = output_path.replace(".csv", "_ticks.csv")
    tick_fieldnames = ["slot_ts", "token", "t", "price"]

    rows = []
    tick_rows = []
    found = 0
    skipped = 0

    slot = start_slot
    while slot > end_slot:
        market = fetch_market_for_slot(slot)
        if market is None:
            skipped += 1
            slot -= SLOT_INTERVAL
            time.sleep(0.2)
            continue

        found += 1
        strike = parse_strike_price(market["question"])
        outcome_prices = market["outcome_prices"]

        # Determine start/end prices from outcome_prices
        up_price_end = float(outcome_prices[0]) if len(outcome_prices) > 0 else None
        down_price_end = float(outcome_prices[1]) if len(outcome_prices) > 1 else None

        # Fetch intra-market tick data for start prices
        up_ticks = []
        down_ticks = []
        if market["up_token"]:
            up_ticks = fetch_price_history(market["up_token"], slot)
            time.sleep(0.2)
        if market["down_token"]:
            down_ticks = fetch_price_history(market["down_token"], slot)
            time.sleep(0.2)

        up_price_start = up_ticks[0]["p"] if up_ticks else None
        down_price_start = down_ticks[0]["p"] if down_ticks else None

        row = {
            "slot_ts": market["slot_ts"],
            "slot_utc": market["slot_utc"],
            "question": market["question"],
            "up_token": market["up_token"],
            "down_token": market["down_token"],
            "outcome": market["outcome"] or "",
            "volume": market["volume"],
            "up_price_start": up_price_start or "",
            "up_price_end": up_price_end or "",
            "down_price_start": down_price_start or "",
            "down_price_end": down_price_end or "",
            "strike_price": strike or "",
        }
        rows.append(row)

        if collect_ticks:
            for tick in up_ticks:
                tick_rows.append({
                    "slot_ts": slot, "token": "up",
                    "t": tick["t"], "price": tick["p"],
                })
            for tick in down_ticks:
                tick_rows.append({
                    "slot_ts": slot, "token": "down",
                    "t": tick["t"], "price": tick["p"],
                })

        progress = found + skipped
        if progress % 10 == 0 or progress == total_slots:
            print(f"  Progress: {progress}/{total_slots} slots processed, "
                  f"{found} found, {skipped} not found")

        slot -= SLOT_INTERVAL
        time.sleep(0.2)

    # Sort rows by slot_ts ascending
    rows.sort(key=lambda r: r["slot_ts"])

    # Write market summary CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Wrote {len(rows)} markets to {output_path}")

    # Write ticks CSV if requested
    if collect_ticks and tick_rows:
        tick_rows.sort(key=lambda r: (r["slot_ts"], r["token"], r["t"]))
        with open(ticks_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tick_fieldnames)
            writer.writeheader()
            writer.writerows(tick_rows)
        print(f"Wrote {len(tick_rows)} tick entries to {ticks_path}")

    # Quick summary
    if rows:
        up_wins = sum(1 for r in rows if r["outcome"] == "Up")
        down_wins = sum(1 for r in rows if r["outcome"] == "Down")
        unresolved = sum(1 for r in rows if r["outcome"] == "")
        volumes = [r["volume"] for r in rows if r["volume"] > 0]
        avg_vol = sum(volumes) / len(volumes) if volumes else 0
        print(f"\nSummary: {len(rows)} markets | Up: {up_wins} | Down: {down_wins} | "
              f"Unresolved: {unresolved} | Avg volume: ${avg_vol:,.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect historical BTC 5-min Up/Down market data from Polymarket"
    )
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Hours of history to collect (default: 24)"
    )
    parser.add_argument(
        "--output", type=str, default="data/btc_updown_5m.csv",
        help="Output CSV path (default: data/btc_updown_5m.csv)"
    )
    parser.add_argument(
        "--ticks", action="store_true",
        help="Also collect intra-market price ticks"
    )
    args = parser.parse_args()
    collect(args.hours, args.output, args.ticks)


if __name__ == "__main__":
    main()
