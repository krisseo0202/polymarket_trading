"""
Generic cryptocurrency live price feed via WebSocket — supports Coinbase and Binance.US.

Works for any asset available on the exchange: BTC-USD, ETH-USD, SOL-USD, etc.
This is the generalized version of BtcPriceFeed — same thread-safe interface:
    get_latest_mid(), get_recent_prices(), is_healthy(), get_latest_book()
"""

import asyncio
import bisect
import collections
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, List, Literal, Optional, Tuple

Exchange = Literal["coinbase", "binance_us"]

_STALE_WARN_S = 2.0
_RECONNECT_S = 5.0
_BUFFER_S = 300.0
_BACKOFF_INIT = 1.0
_BACKOFF_MAX = 60.0


@dataclass(frozen=True)
class BookSnapshot:
    """Immutable snapshot of best bid/ask."""
    bid: float
    ask: float
    mid: float
    exchange_ts: Optional[float]  # exchange-provided timestamp (seconds); None if unavailable
    local_ts: float               # time.time() at receipt


class CryptoPriceFeed:
    """
    Live cryptocurrency price feed with pluggable exchange backend.

    Args:
        symbol:   Product symbol for the chosen exchange.
                  Coinbase:   "ETH-USD", "SOL-USD", "BTC-USD", etc.
                  Binance.US: "ethusd", "solusd", "btcusd", etc.
        exchange: "coinbase" (default) | "binance_us"
    """

    # Default symbols per exchange for common assets
    COINBASE_SYMBOLS = {
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "BTC": "BTC-USD",
    }
    BINANCE_SYMBOLS = {
        "ETH": "ethusd",
        "SOL": "solusd",
        "BTC": "btcusd",
    }

    def __init__(
        self,
        symbol: str = "ETH-USD",
        exchange: Exchange = "coinbase",
        logger: Optional[logging.Logger] = None,
        stale_warn_s: float = _STALE_WARN_S,
        reconnect_s: float = _RECONNECT_S,
        buffer_s: float = _BUFFER_S,
    ):
        if exchange not in ("coinbase", "binance_us"):
            raise ValueError(f"exchange must be 'coinbase' or 'binance_us', got {exchange!r}")

        self._exchange: Exchange = exchange
        self._symbol = symbol
        # Derive a short asset name for logging (e.g., "ETH-USD" -> "eth")
        self._asset_tag = symbol.split("-")[0].lower() if "-" in symbol else symbol.lower()
        self._logger = logger or logging.getLogger(f"crypto_feed.{self._asset_tag}")
        self._stale_warn_s = stale_warn_s
        self._reconnect_s = reconnect_s
        self._buffer_s = buffer_s

        self._lock = threading.Lock()
        self._latest: Optional[BookSnapshot] = None
        self._buffer: Deque[Tuple[float, float, float]] = collections.deque()
        self._stop_evt = threading.Event()
        self.reconnect_count: int = 0

        self._ws_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def exchange(self) -> str:
        return self._exchange

    def start(self) -> "CryptoPriceFeed":
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

    # -- Public getters --------------------------------------------------------

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

    def get_recent_prices(self, window_s: int = 300) -> List[Tuple[float, float, float]]:
        """Return (timestamp, mid_price, tick_count) triples for the last N seconds."""
        cutoff = time.time() - window_s
        with self._lock:
            if not self._buffer:
                return []
            idx = bisect.bisect_left(self._buffer, (cutoff,))
            return list(self._buffer)[idx:]

    # -- Shared update helper --------------------------------------------------

    def _update(self, bid: float, ask: float, mid: float, exchange_ts: Optional[float]) -> None:
        now = time.time()
        snap = BookSnapshot(bid=bid, ask=ask, mid=mid, exchange_ts=exchange_ts, local_ts=now)
        buf_ts = exchange_ts if exchange_ts is not None else now
        with self._lock:
            self._latest = snap
            self._buffer.append((buf_ts, mid, 1.0))
            cutoff = now - self._buffer_s
            while self._buffer and self._buffer[0][0] < cutoff:
                self._buffer.popleft()

    # -- WebSocket loop --------------------------------------------------------

    def _ws_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_connect_loop())
        except Exception as e:
            self._logger.error(f"WS loop crashed: {e}")
        finally:
            loop.close()

    async def _ws_connect_loop(self) -> None:
        try:
            import websockets  # noqa: F401
        except ImportError:
            self._logger.error("websockets package not installed (pip install websockets)")
            return

        backoff = _BACKOFF_INIT
        first_connect = True

        while not self._stop_evt.is_set():
            try:
                if self._exchange == "coinbase":
                    await self._run_coinbase()
                else:
                    await self._run_binance_us()

                if not first_connect:
                    self.reconnect_count += 1
                first_connect = False
                backoff = _BACKOFF_INIT

            except Exception as e:
                if self._stop_evt.is_set():
                    break
                self._logger.warning(
                    f"[{self._exchange}/{self._asset_tag}] WS error: {e}, reconnecting in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    # -- Coinbase WebSocket ----------------------------------------------------

    async def _run_coinbase(self) -> None:
        import websockets

        uri = "wss://ws-feed.exchange.coinbase.com"
        subscribe = {
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": [self._symbol]}],
        }

        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
            self._logger.info(f"[coinbase] Connected — subscribing to {self._symbol} ticker")
            await ws.send(json.dumps(subscribe))

            while not self._stop_evt.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    with self._lock:
                        latest_ts = self._latest.local_ts if self._latest else None
                    if latest_ts is not None and (time.time() - latest_ts) > self._reconnect_s:
                        raise ConnectionError(
                            f"[{self._exchange}/{self._asset_tag}] No price data for "
                            f"{self._reconnect_s:.0f}s — reconnecting"
                        )
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") != "ticker":
                    continue

                bid_s = msg.get("best_bid")
                ask_s = msg.get("best_ask")
                if not bid_s or not ask_s:
                    continue

                bid = float(bid_s)
                ask = float(ask_s)
                mid = (bid + ask) / 2.0

                ts_raw = msg.get("time")
                exchange_ts: Optional[float] = None
                if ts_raw:
                    try:
                        exchange_ts = datetime.fromisoformat(
                            ts_raw.replace("Z", "+00:00")
                        ).timestamp()
                    except ValueError:
                        pass

                self._update(bid, ask, mid, exchange_ts)

    # -- Binance.US WebSocket --------------------------------------------------

    async def _run_binance_us(self) -> None:
        import websockets

        symbol = self._symbol.lower()
        uri = f"wss://stream.binance.us:9443/ws/{symbol}@bookTicker"

        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
            self._logger.info(f"[binance_us] Connected — {symbol}@bookTicker")

            while not self._stop_evt.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    with self._lock:
                        latest_ts = self._latest.local_ts if self._latest else None
                    if latest_ts is not None and (time.time() - latest_ts) > self._reconnect_s:
                        raise ConnectionError(
                            f"[{self._exchange}/{self._asset_tag}] No price data for "
                            f"{self._reconnect_s:.0f}s — reconnecting"
                        )
                    continue

                try:
                    msg = json.loads(raw)
                    bid = float(msg["b"])
                    ask = float(msg["a"])
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self._logger.debug(f"[binance_us/{self._asset_tag}] Bad message: {e}")
                    continue

                mid = (bid + ask) / 2.0
                self._update(bid, ask, mid, None)

    # -- Watchdog --------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        last_health_log = 0.0
        last_stale_warn = 0.0
        _HEALTH_LOG_INTERVAL_S = 60.0
        _STALE_WARN_INTERVAL_S = 30.0
        while not self._stop_evt.is_set():
            age_ms = self.get_feed_age_ms()
            now = time.time()
            if age_ms is not None:
                age_s = age_ms / 1000
                if age_s > self._reconnect_s:
                    if now - last_stale_warn >= _STALE_WARN_INTERVAL_S:
                        self._logger.warning(
                            f"[{self._exchange}/{self._asset_tag}] Price data stale ({age_s:.1f}s) — "
                            f"recv loop will reconnect"
                        )
                        last_stale_warn = now
                elif age_s > self._stale_warn_s:
                    self._logger.debug(
                        f"[{self._exchange}/{self._asset_tag}] Feed age {age_s:.1f}s (warn threshold)"
                    )
            if now - last_health_log >= _HEALTH_LOG_INTERVAL_S:
                self._logger.debug(
                    f"[{self._exchange}/{self._asset_tag}] health: age={age_ms:.0f}ms "
                    f"reconnects={self.reconnect_count}"
                    if age_ms is not None
                    else f"[{self._exchange}/{self._asset_tag}] health: no data yet reconnects={self.reconnect_count}"
                )
                last_health_log = now
            self._stop_evt.wait(0.5)
