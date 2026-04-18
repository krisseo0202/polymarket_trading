"""Market utility functions for trading"""

import email.utils
import json
import math
import time
from typing import List, Optional, Dict

import requests

from ..api.client import PolymarketClient
from ..api.types import OrderBook

# Corrected clock offset: server_time - local_time, updated on each slug lookup.
_server_offset: float = 0.0


def get_server_time() -> float:
    """Return local time corrected by the last-observed Polymarket server offset."""
    return time.time() + _server_offset

def _contains_all(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return all(k.lower() in t for k in keywords)

def get_active_events_with_token_ids(
    min_volume: float = 100000,                 # more realistic default
    event_keywords: Optional[List[str]] = None, # e.g. ["btc"]
    market_keywords: Optional[List[str]] = None # e.g. ["up", "down"] or ["up", "down", "5", "minute"]
) -> list:
    """
    Fetch active events with their token IDs, optionally filtering by keywords.
    - event_keywords are matched against event['title']
    - market_keywords are matched against market['question'] (and event title as fallback)
    """
    url = "https://gamma-api.polymarket.com/events"
    all_events = []

    params = {"active": "true", "closed": "false", "limit": 100, "offset": 0}
    # NOTE: gamma supports pagination; if you need full coverage, loop over offset.

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    events = response.json()
    if not events:
        return []

    event_keywords = event_keywords or []
    market_keywords = market_keywords or []

    for event in events:
        title = event.get("title", "")
        if event_keywords and not _contains_all(title, event_keywords):
            continue

        total_volume = 0.0
        markets_with_tokens = []

        for market in event.get("markets", []):
            volume = float(market.get("volume", 0) or 0)
            total_volume += volume

            question = market.get("question", "")
            # Filter at market level (most important)
            if market_keywords:
                # check question first, then fallback to combined text
                combined = f"{title} {question}"
                if not _contains_all(combined, market_keywords):
                    continue

            # Parse clobTokenIds safely (it’s often a JSON string)
            raw = market.get("clobTokenIds", "[]")
            try:
                clob_token_ids = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                clob_token_ids = []

            markets_with_tokens.append({
                "market_id": market.get("id"),
                "question": question,
                "yes_token_id": clob_token_ids[0] if len(clob_token_ids) > 0 else None,
                "no_token_id": clob_token_ids[1] if len(clob_token_ids) > 1 else None,
                "neg_risk": market.get("negRisk", False),
                "tick_size": market.get("orderPriceMinTickSize", 0.01),
                "volume": volume,
            })

        # Only keep events that have at least one matching market + pass volume
        if markets_with_tokens and total_volume >= min_volume:
            all_events.append({
                "event_id": event.get("id"),
                "title": title,
                "total_volume": total_volume,
                "markets": markets_with_tokens
            })

    return all_events

def round_to_tick(price: float, tick: float) -> float:
    """
    Round price to nearest tick.
    
    Args:
        price: Price to round
        tick: Tick size
        
    Returns:
        Rounded price
    """
    if tick <= 0:
        tick = 0.001
    decimals = len(str(tick).split(".")[-1]) if "." in str(tick) else 3
    return round(round(price / tick) * tick, decimals)


def get_tick_size_fallback(token_id: str) -> float:
    """
    Get tick size for a token (fallback implementation).
    You can wire your tick-size API later.
    
    Args:
        token_id: Token ID
        
    Returns:
        Default tick size (0.001)
    """
    # Keep it simple. You can wire your tick-size API later.
    return 0.001


def get_mid_price(order_book: OrderBook) -> Optional[float]:
    """
    Extract mid price from order book.
    Prefers last_price if available, else midpoint of best bid/ask.
    
    Args:
        order_book: Order book object
        
    Returns:
        Mid price or None if unavailable
    """
    # Prefer last_price if your wrapper provides it, else midpoint of best bid/ask
    if getattr(order_book, "last_price", None):
        return order_book.last_price
    bids = getattr(order_book, "bids", None) or []
    asks = getattr(order_book, "asks", None) or []
    if bids and asks:
        return (bids[0].price + asks[0].price) / 2.0
    if bids:
        return bids[0].price
    if asks:
        return asks[0].price
    return None


def spread_pct(bid: float, ask: float) -> float:
    """Spread as a fraction of mid price. Returns inf when mid is zero."""
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else float("inf")


def parse_event_market(event: dict) -> Optional[Dict]:
    """Extract market_id / yes_token_id / no_token_id from a gamma-api event dict."""
    for m in event.get("markets", []):
        tids = json.loads(m.get("clobTokenIds", "[]"))
        yes_id = tids[0] if len(tids) > 0 else m.get("yes_token_id")
        no_id  = tids[1] if len(tids) > 1 else m.get("no_token_id")
        if yes_id and no_id:
            return {
                "market_id": m.get("id") or m.get("market_id", ""),
                "yes_token_id": yes_id,
                "no_token_id": no_id,
                "question": m.get("question", event.get("title", "")),
            }
    return None


def find_updown_market(
    keywords: List[str],
    min_volume: int,
    logger,
    slug_prefix: str = "btc-updown-5m",
) -> Optional[Dict]:
    """
    Discover the current Up/Down 5-minute market for any asset.

    Discovery order:
      1. Slug-based lookup: gamma-api /events?slug={slug_prefix}-{slot}
      2. Keyword search across gamma-api events (min_volume filter).
      3. Keyword search across gamma-api /markets endpoint.

    Returns dict {market_id, yes_token_id, no_token_id, question} or None.
    """
    global _server_offset
    lower_kws = [k.lower() for k in keywords]

    # 1. Slug-based lookup (fastest, most precise)
    slot = int(math.floor(get_server_time() / 300) * 300)
    slug = f"{slug_prefix}-{slot}"
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        date_hdr = resp.headers.get("Date")
        if date_hdr:
            try:
                _server_offset = (
                    email.utils.parsedate_to_datetime(date_hdr).timestamp() - time.time()
                )
            except Exception:
                pass
        data = resp.json()
        events = data if isinstance(data, list) else [data]
        for event in events:
            result = parse_event_market(event)
            if result:
                logger.info(
                    f"Found market via slug '{slug}': "
                    f"yes={result['yes_token_id'][:12]}..."
                )
                return result
    except Exception as e:
        logger.warning(f"Slug lookup failed ({slug}): {e}")

    # 2. Keyword search across events
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"active": "true", "closed": "false", "limit": 100, "offset": 0},
            timeout=10,
        )
        resp.raise_for_status()
        for event in resp.json():
            title = event.get("title", "").lower()
            total_vol = sum(
                float(m.get("volume", 0) or 0) for m in event.get("markets", [])
            )
            if total_vol < min_volume:
                continue
            if all(kw in title for kw in lower_kws):
                result = parse_event_market(event)
                if result:
                    logger.info(
                        f"Found market via keyword search '{event['title']}': "
                        f"yes={result['yes_token_id'][:12]}..."
                    )
                    return result
    except Exception as e:
        logger.warning(f"Keyword/events search failed: {e}")

    # 3. gamma-api /markets endpoint
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 100, "offset": 0},
            timeout=10,
        )
        resp.raise_for_status()
        for m in resp.json():
            question = m.get("question", "").lower()
            if all(kw in question for kw in lower_kws):
                tids = json.loads(m.get("clobTokenIds", "[]"))
                yes_id = tids[0] if len(tids) > 0 else None
                no_id  = tids[1] if len(tids) > 1 else None
                if yes_id and no_id:
                    logger.info(
                        f"Found market via gamma-api/markets: '{m['question']}'"
                    )
                    return {
                        "market_id": m.get("id", ""),
                        "yes_token_id": yes_id,
                        "no_token_id": no_id,
                        "question": m.get("question", ""),
                    }
    except Exception as e:
        logger.warning(f"gamma-api/markets search failed: {e}")

    logger.warning(f"Up/Down market not found this cycle (slug_prefix={slug_prefix})")
    return None


def cancel_if_exists(client: PolymarketClient, order_id: Optional[str], dry_run: bool) -> None:
    """
    Cancel an order if it exists (safe wrapper).
    
    Args:
        client: Polymarket API client
        order_id: Order ID to cancel
        dry_run: If True, don't actually cancel
    """
    if not order_id:
        return
    if dry_run:
        return
    try:
        client.cancel_order(order_id)
    except Exception:
        pass
