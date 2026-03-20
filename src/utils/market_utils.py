"""Market utility functions for trading"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import requests
import json

from ..api.client import PolymarketClient
from ..api.types import OrderBook


# ── Constants ─────────────────────────────────────────────────────────────────

_GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
_CLOB_BOOK_URL     = "https://clob.polymarket.com/book"

_log = logging.getLogger(__name__)


# ── fetch_market_odds ─────────────────────────────────────────────────────────

def fetch_market_odds(
    market_id: str,
    *,
    retries: int = 3,
    backoff_base: float = 1.0,
    timeout: float = 10.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float]:
    """
    Return the current implied probabilities for the UP (YES) and DOWN (NO)
    outcomes of a Polymarket binary market.

    Implied probability is defined as the mid-price of each outcome token's
    best bid/ask:  mid = (best_bid + best_ask) / 2.

    On Polymarket every binary market has exactly two CLOB outcome tokens:
        clobTokenIds[0]  →  YES / "Up"
        clobTokenIds[1]  →  NO  / "Down"

    Args:
        market_id:    Numeric Gamma market ID (string), e.g. ``"12345"``.
                      Obtain one from ``get_active_events_with_token_ids()``.
        retries:      Maximum number of attempts per HTTP request (default 3).
        backoff_base: Base delay in seconds for exponential backoff (default 1).
                      Delay after attempt k  =  backoff_base * 2^k   (capped 30s).
        timeout:      Per-request timeout in seconds (default 10).
        logger:       Optional logger; falls back to the module-level logger.

    Returns:
        ``(up_prob, down_prob)`` — both floats in [0.0, 1.0].

    Raises:
        ValueError:  Market not found, or token IDs are missing.
        RuntimeError: All retry attempts exhausted for a network call.

    Example::

        up, down = fetch_market_odds("12345")
        print(f"YES={up:.2%}  NO={down:.2%}")
    """
    log = logger or _log

    # ── Step 1: resolve token IDs from the Gamma markets API ─────────────────
    # Gamma returns a list even when querying by id.
    gamma_data = _fetch_with_retry(
        url=_GAMMA_MARKETS_URL,
        params={"id": market_id},
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        label=f"gamma/markets?id={market_id}",
        logger=log,
    )

    if not gamma_data:
        raise ValueError(f"Market {market_id!r} not found in Gamma API")

    market = gamma_data[0] if isinstance(gamma_data, list) else gamma_data

    # clobTokenIds is sometimes a JSON string, sometimes a list
    raw_ids = market.get("clobTokenIds", "[]")
    try:
        token_ids: List[str] = json.loads(raw_ids) if isinstance(raw_ids, str) else (raw_ids or [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse clobTokenIds for market {market_id}: {exc}") from exc

    if len(token_ids) < 2:
        raise ValueError(
            f"Market {market_id} has {len(token_ids)} token ID(s); expected 2 (YES + NO). "
            f"Raw clobTokenIds: {raw_ids!r}"
        )

    yes_token_id, no_token_id = token_ids[0], token_ids[1]
    log.debug(f"Market {market_id}: YES={yes_token_id[:12]}… NO={no_token_id[:12]}…")

    # ── Step 2: fetch order books for both tokens ─────────────────────────────
    yes_book = _fetch_with_retry(
        url=_CLOB_BOOK_URL,
        params={"token_id": yes_token_id},
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        label=f"clob/book YES token",
        logger=log,
    )
    no_book = _fetch_with_retry(
        url=_CLOB_BOOK_URL,
        params={"token_id": no_token_id},
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        label=f"clob/book NO token",
        logger=log,
    )

    # ── Step 3: compute mid-prices ────────────────────────────────────────────
    up_prob   = _book_mid(yes_book, label="YES")
    down_prob = _book_mid(no_book,  label="NO")

    log.info(
        f"Market {market_id} odds — "
        f"UP(YES)={up_prob:.4f}  DOWN(NO)={down_prob:.4f}  "
        f"sum={up_prob + down_prob:.4f}"
    )
    return up_prob, down_prob


# ── Private helpers ───────────────────────────────────────────────────────────

def _fetch_with_retry(
    url: str,
    params: dict,
    timeout: float,
    retries: int,
    backoff_base: float,
    label: str,
    logger: logging.Logger,
) -> dict:
    """
    GET ``url`` with ``params``, retrying up to ``retries`` times on any
    network or HTTP error using exponential backoff.

    Returns the parsed JSON body.
    Raises ``RuntimeError`` if all attempts fail.
    """
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            logger.warning(f"[{label}] Timeout on attempt {attempt + 1}/{retries}")

        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            logger.warning(f"[{label}] Connection error on attempt {attempt + 1}/{retries}: {exc}")

        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else "?"
            logger.warning(f"[{label}] HTTP {status} on attempt {attempt + 1}/{retries}")
            # 4xx errors (bad market ID, etc.) are not worth retrying
            if exc.response is not None and exc.response.status_code < 500:
                raise RuntimeError(f"[{label}] HTTP {status} — {exc}") from exc

        except (ValueError, KeyError) as exc:
            # JSON decode failure or malformed response — not a transient error
            raise RuntimeError(f"[{label}] Bad response body: {exc}") from exc

        if attempt < retries - 1:
            delay = min(backoff_base * (2 ** attempt), 30.0)
            logger.info(f"[{label}] Retrying in {delay:.1f}s …")
            time.sleep(delay)

    raise RuntimeError(
        f"[{label}] All {retries} attempts failed. Last error: {last_exc}"
    ) from last_exc


def _book_mid(book: dict, label: str) -> float:
    """
    Extract the mid-price from a CLOB /book response dict.

    Priority:
      1. (best_bid + best_ask) / 2  — most accurate
      2. best_bid alone              — if asks side is empty
      3. best_ask alone              — if bids side is empty

    Raises ``ValueError`` if the book is completely empty.
    """
    def _best(entries: list, side: str) -> Optional[float]:
        """Return the best price from a list of {price/p, size/s} entries."""
        if not entries:
            return None
        try:
            # CLOB uses either "price"/"size" or compact "p"/"s" keys
            prices = [float(e.get("price") or e.get("p", 0)) for e in entries]
            if side == "bid":
                return max(prices)   # best bid = highest
            else:
                return min(prices)   # best ask = lowest
        except (TypeError, ValueError):
            return None

    best_bid = _best(book.get("bids", []), "bid")
    best_ask = _best(book.get("asks", []), "ask")

    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    if best_bid is not None:
        _log.debug(f"{label} book: asks empty, using best_bid={best_bid:.4f} as mid")
        return best_bid
    if best_ask is not None:
        _log.debug(f"{label} book: bids empty, using best_ask={best_ask:.4f} as mid")
        return best_ask

    raise ValueError(f"{label} order book is completely empty — cannot compute mid-price")

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
