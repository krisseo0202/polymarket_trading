"""
ws_feed.py — Async WebSocket price feed (Binance best bid/ask + depth)

Usage:
    from src.utils.ws_feed import subscribe_price_ws
    stop = subscribe_price_ws("binance", "btcusdt", print)
    time.sleep(5)
    stop.set()
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import websockets


@dataclass
class Tick:
    exchange: str
    symbol: str
    best_bid: Optional[float]          # None if depth-only message
    best_ask: Optional[float]
    bids: List[Tuple[float, float]]    # [(price, qty), ...]
    asks: List[Tuple[float, float]]
    ts: float                          # time.monotonic()


def _build_url(exchange: str, symbol: str) -> str:
    if exchange == "binance":
        return (
            f"wss://stream.binance.com:9443/stream"
            f"?streams={symbol}@bookTicker/{symbol}@depth5@100ms"
        )
    raise ValueError(f"Unsupported exchange: {exchange!r}")


def _parse(exchange: str, symbol: str, raw: str) -> Optional[Tick]:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None

    data = msg.get("data", msg)
    stream = msg.get("stream", "")
    ts = time.monotonic()

    if "bookTicker" in stream or ("b" in data and "a" in data and "bids" not in data):
        return Tick(
            exchange=exchange,
            symbol=symbol,
            ts=ts,
            best_bid=float(data["b"]) if data.get("b") else None,
            best_ask=float(data["a"]) if data.get("a") else None,
            bids=[],
            asks=[],
        )
    if "depth" in stream or "bids" in data:
        return Tick(
            exchange=exchange,
            symbol=symbol,
            ts=ts,
            best_bid=None,
            best_ask=None,
            bids=[(float(p), float(q)) for p, q in data.get("bids", [])],
            asks=[(float(p), float(q)) for p, q in data.get("asks", [])],
        )
    return None


async def _ws_listener(
    url: str,
    exchange: str,
    symbol: str,
    on_tick: Callable[[Tick], None],
    stop_event: threading.Event,
    logger,
) -> None:
    backoff = 1.0
    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                if logger:
                    logger.info(f"WS connected: {exchange}/{symbol}")
                backoff = 1.0
                async for raw in ws:
                    if stop_event.is_set():
                        return
                    tick = _parse(exchange, symbol, raw)
                    if tick:
                        try:
                            on_tick(tick)
                        except Exception as cb_err:
                            if logger:
                                logger.warning(f"on_tick callback error: {cb_err}")
        except Exception as e:
            if stop_event.is_set():
                return
            if logger:
                logger.warning(
                    f"WS error ({e.__class__.__name__}: {e}), "
                    f"reconnecting in {backoff:.0f}s"
                )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


def subscribe_price_ws(
    exchange: str,
    symbol: str,
    on_tick: Callable[[Tick], None],
    logger=None,
) -> threading.Event:
    """
    Stream best bid/ask + order-book depth from `exchange` for `symbol`.

    Runs an asyncio event loop in a daemon thread. Auto-reconnects with
    exponential backoff (1s → 60s cap).

    Args:
        exchange: "binance" (extensible via _build_url)
        symbol:   e.g. "btcusdt" (case-insensitive, normalised to lower)
        on_tick:  callable receiving a Tick on every message
        logger:   optional logger for info/warning messages

    Returns:
        threading.Event — call .set() to stop the feed.
    """
    symbol = symbol.lower()
    url = _build_url(exchange, symbol)
    stop_event = threading.Event()

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _ws_listener(url, exchange, symbol, on_tick, stop_event, logger)
            )
        finally:
            loop.close()

    threading.Thread(
        target=_run,
        daemon=True,
        name=f"ws-{exchange}-{symbol}",
    ).start()

    return stop_event
