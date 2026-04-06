"""
Fetch historical price data for a specific BTC 5-min Up/Down market on Polymarket.

Usage:
    python scripts/fetch_market_history.py
    python scripts/fetch_market_history.py --slot-ts 1775407500
    python scripts/fetch_market_history.py --date "2026-04-05 12:45" --tz ET
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

TZ_OFFSETS = {"ET": -4, "CT": -5, "MT": -6, "PT": -7, "UTC": 0}


def compute_slot_ts(date_str: str, tz_name: str) -> int:
    """Parse a date string like '2026-04-05 12:45' in the given timezone and return a 5-min-aligned UTC timestamp."""
    offset = TZ_OFFSETS.get(tz_name.upper(), 0)
    tz = timezone(timedelta(hours=offset))
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    ts = int(dt.timestamp())
    return ts - (ts % 300)


def fetch_event(slot_ts: int):
    """Fetch event and markets from Gamma API by slug."""
    slug = f"btc-updown-5m-{slot_ts}"
    print(f"Looking up slug: {slug}")
    resp = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=15)
    resp.raise_for_status()
    events = resp.json()
    if not events:
        print(f"No event found for slot_ts={slot_ts}", file=sys.stderr)
        sys.exit(1)
    return events[0]


def find_market(markets, keyword):
    """Find market whose question or groupItemTitle contains keyword."""
    for m in markets:
        text = m.get("groupItemTitle", "") + " " + m.get("question", "")
        if keyword.lower() in text.lower():
            return m
    return None


def fetch_price_history(token_id: str):
    """Fetch price history ticks from CLOB API."""
    resp = requests.get(
        f"{CLOB_API}/prices-history",
        params={"market": token_id, "interval": "max"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("history", [])


def main():
    parser = argparse.ArgumentParser(description="Fetch BTC 5-min Up/Down price history from Polymarket")
    parser.add_argument("--slot-ts", type=int, help="5-minute slot UTC timestamp (e.g. 1775407500)")
    parser.add_argument("--date", type=str, help="Date and time, e.g. '2026-04-05 12:45'")
    parser.add_argument("--tz", type=str, default="ET", help="Timezone: ET, CT, MT, PT, UTC (default: ET)")
    args = parser.parse_args()

    if args.slot_ts:
        slot_ts = args.slot_ts
    elif args.date:
        slot_ts = compute_slot_ts(args.date, args.tz)
    else:
        # Default: April 5, 2026 12:45 PM ET
        slot_ts = compute_slot_ts("2026-04-05 12:45", "ET")

    print(f"Slot timestamp: {slot_ts}")
    print(f"Slot UTC: {datetime.fromtimestamp(slot_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # 1. Fetch event
    event = fetch_event(slot_ts)
    markets = event.get("markets", [])
    print(f"Event: {event.get('title', 'N/A')}")
    print(f"Markets found: {len(markets)}")
    print()

    # 2. Extract Up and Down token IDs
    #    BTC Up/Down markets typically have a single market with outcomes=["Up","Down"]
    #    and clobTokenIds=[up_token, down_token] in matching order.
    market = markets[0]
    outcomes = json.loads(market["outcomes"]) if isinstance(market.get("outcomes"), str) else market.get("outcomes", [])
    token_ids = json.loads(market["clobTokenIds"]) if isinstance(market.get("clobTokenIds"), str) else market.get("clobTokenIds", [])

    if len(outcomes) < 2 or len(token_ids) < 2:
        print(f"Unexpected market structure: outcomes={outcomes}, tokens={token_ids}", file=sys.stderr)
        sys.exit(1)

    # Map outcome name to token ID
    outcome_map = dict(zip(outcomes, token_ids))
    up_token = outcome_map.get("Up")
    down_token = outcome_map.get("Down")

    print(f"Up token:   {up_token}")
    print(f"Down token: {down_token}")
    print()

    # 4. Fetch price history
    if up_token:
        up_history = fetch_price_history(up_token)
        print(f"Up price history: {len(up_history)} ticks")
        if up_history:
            print(f"  First: t={up_history[0]['t']} p={up_history[0]['p']}")
            print(f"  Last:  t={up_history[-1]['t']} p={up_history[-1]['p']}")
    else:
        up_history = []
        print("Up token not found, skipping price history")

    if down_token:
        down_history = fetch_price_history(down_token)
        print(f"Down price history: {len(down_history)} ticks")
        if down_history:
            print(f"  First: t={down_history[0]['t']} p={down_history[0]['p']}")
            print(f"  Last:  t={down_history[-1]['t']} p={down_history[-1]['p']}")
    else:
        down_history = []
        print("Down token not found, skipping price history")

    # 5. Show outcome prices
    print()
    prices_raw = market.get("outcomePrices", "")
    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    closed = market.get("closed", False)
    print(f"Market: closed={closed}, outcomes={outcomes}, outcomePrices={prices}")


if __name__ == "__main__":
    main()
