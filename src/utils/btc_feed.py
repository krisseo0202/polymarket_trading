"""
BTC/USDT live price feed via Binance WebSocket bookTicker stream.

Provides a thread-safe BtcPriceFeed that runs a background WebSocket
connection and exposes the latest bid/ask/mid for dashboard display.
"""

import asyncio
import collections
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


@dataclass(frozen=True)
class BookSnapshot:
    """Immutable snapshot of best bid/ask from Binance."""
    bid: float
    ask: float
    mid: float
    exchange_ts: Optional[float]  # None for Binance bookTicker
    local_ts: float               # time.time() at receipt


_STALE_WARN_S = 2.0
_RECONNECT_S = 5.0
_BUFFER_S = 300.0
_BACKOFF_INIT = 1.0
_BACKOFF_MAX = 60.0


class BtcPriceFeed:
    """Live BTC price feed from Binance bookTicker WebSocket."""

    def __init__(
        self,
        symbol: str = "btcusdt",
        logger: Optional[logging.Logger] = None,
        stale_warn_s: float = _STALE_WARN_S,
        reconnect_s: float = _RECONNECT_S,
        buffer_s: float = _BUFFER_S,
    ):
        self._symbol = symbol.lower()
        self._logger = logger or logging.getLogger("btc_feed")
        self._stale_warn_s = stale_warn_s
        self._reconnect_s = reconnect_s
        self._buffer_s = buffer_s

        self._lock = threading.Lock()
        self._latest: Optional[BookSnapshot] = None
        self._buffer: Deque[Tuple[float, float]] = collections.deque()
        self._stop_evt = threading.Event()
        self.reconnect_count: int = 0

        self._ws_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

    def start(self) -> "BtcPriceFeed":
        """Start WebSocket and watchdog threads. Returns self for chaining."""
        self._stop_evt.clear()
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        return self

    def stop(self) -> None:
        """Signal threads to exit (non-blocking)."""
        self._stop_evt.set()

    # ── Public getters ────────────────────────────────────────────────────

    def get_latest_book(self) -> Optional[BookSnapshot]:
        with self._lock:
            return self._latest

    def get_latest_mid(self) -> Optional[float]:
        with self._lock:
            return self._latest.mid if self._latest else None

    def get_feed_age_ms(self) -> Optional[float]:
        with self._lock:
            if self._latest is None:
                return None
            return (time.time() - self._latest.local_ts) * 1000

    def is_healthy(self) -> bool:
        age = self.get_feed_age_ms()
        if age is None:
            return False
        return age < self._stale_warn_s * 1000

    def get_recent_prices(self, window_s: int = 300) -> List[Tuple[float, float]]:
        """Return (timestamp, mid_price) pairs for the last N seconds."""
        cutoff = time.time() - window_s
        with self._lock:
            return [(ts, mid) for ts, mid in self._buffer if ts >= cutoff]

    # ── Message handling ──────────────────────────────────────────────────

    def _handle_message(self, raw: str) -> None:
        """Parse a raw JSON bookTicker message."""
        try:
            data = json.loads(raw)
            bid = float(data["b"])
            ask = float(data["a"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._logger.debug(f"Bad message: {e}")
            return

        mid = (bid + ask) / 2
        now = time.time()
        snap = BookSnapshot(bid=bid, ask=ask, mid=mid, exchange_ts=None, local_ts=now)

        with self._lock:
            self._latest = snap
            self._buffer.append((now, mid))
            # Trim old entries
            cutoff = now - self._buffer_s
            while self._buffer and self._buffer[0][0] < cutoff:
                self._buffer.popleft()

    # ── WebSocket loop ────────────────────────────────────────────────────

    def _ws_loop(self) -> None:
        """Run async WebSocket in a dedicated event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_connect_loop())
        except Exception as e:
            self._logger.error(f"WS loop crashed: {e}")
        finally:
            loop.close()

    async def _ws_connect_loop(self) -> None:
        """Connect, receive messages, and reconnect on failure."""
        try:
            import websockets
        except ImportError:
            self._logger.error("websockets package not installed (pip install websockets)")
            return

        url = f"wss://stream.binance.com:9443/ws/{self._symbol}@bookTicker"
        backoff = _BACKOFF_INIT
        first_connect = True

        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10
                ) as ws:
                    self._logger.info(f"Connected to {url}")
                    if not first_connect:
                        self.reconnect_count += 1
                    first_connect = False
                    backoff = _BACKOFF_INIT

                    while not self._stop_evt.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            self._handle_message(raw)
                        except asyncio.TimeoutError:
                            continue

            except Exception as e:
                if self._stop_evt.is_set():
                    break
                self._logger.warning(f"WS error: {e}, reconnecting in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    # ── Watchdog ──────────────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Poll feed age and warn/reconnect if stale."""
        while not self._stop_evt.is_set():
            age_ms = self.get_feed_age_ms()
            if age_ms is not None:
                age_s = age_ms / 1000
                if age_s > self._reconnect_s:
                    self._logger.warning(f"Feed stale ({age_s:.1f}s), will reconnect")
                elif age_s > self._stale_warn_s:
                    self._logger.debug(f"Feed age {age_s:.1f}s (warn threshold)")
            self._stop_evt.wait(0.5)
