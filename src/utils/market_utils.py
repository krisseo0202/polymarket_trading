"""Market utility functions for trading"""

from typing import List, Optional, Dict
import requests
import json
from ..api.client import PolymarketClient
from ..api.types import OrderBook

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
