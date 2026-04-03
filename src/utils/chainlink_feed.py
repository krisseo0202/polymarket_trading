"""
Chainlink BTC/USD price feed via Polymarket RTDS WebSocket.

Subscribes to the ``crypto_prices_chainlink`` topic and captures the first
price in each 5-minute slot as the authoritative "price to beat" for
BTC Up/Down market settlement.

The feed is complementary to the Binance ``BtcPriceFeed`` — Binance provides
high-frequency mid-price for features and charts, while this feed provides
the settlement-aligned reference price only.
"""

import asyncio
import collections
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple


@dataclass(frozen=True)
class ChainlinkSnapshot:
    """Immutable snapshot of a single Chainlink price tick."""
    symbol: str
    price: float
    exchange_ts: float   # timestamp from RTDS payload
    local_ts: float      # time.time() at receipt


@dataclass(frozen=True)
class SlotOpenPrice:
    """The first Chainlink price captured at/after a 5-min slot boundary."""
    slot_ts: int        # floor(now/300)*300
    price: float
    captured_at: float  # local time when captured


_STALE_WARN_S = 90.0   # Chainlink updates on 0.5% deviation; quiet periods can last minutes
_RECONNECT_S = 120.0  # only reconnect if no messages (including PONGs) for 2 min
_BUFFER_S = 600.0
_BACKOFF_INIT = 1.0
_BACKOFF_MAX = 60.0
_PING_INTERVAL_S = 5.0

_RTDS_URL = "wss://ws-live-data.polymarket.com"
_RTDS_TOPIC = "crypto_prices_chainlink"
_RTDS_TOPIC_FALLBACK = "crypto_prices"


class ChainlinkFeed:
    """Live Chainlink BTC/USD feed from Polymarket RTDS WebSocket."""

    def __init__(
        self,
        symbol: str = "btc/usd",
        slot_interval_s: int = 300,
        logger: Optional[logging.Logger] = None,
        stale_warn_s: float = _STALE_WARN_S,
        reconnect_s: float = _RECONNECT_S,
        buffer_s: float = _BUFFER_S,
    ):
        self._symbol = symbol.lower()
        self._slot_interval_s = slot_interval_s
        self._logger = logger or logging.getLogger("chainlink_feed")
        self._stale_warn_s = stale_warn_s
        self._reconnect_s = reconnect_s
        self._buffer_s = buffer_s

        self._lock = threading.Lock()
        self._latest: Optional[ChainlinkSnapshot] = None
        self._slot_open: Optional[SlotOpenPrice] = None
        self._buffer: Deque[Tuple[float, float]] = collections.deque()
        self._stop_evt = threading.Event()
        self.reconnect_count: int = 0

        self._ws_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

    def start(self) -> "ChainlinkFeed":
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

    def get_latest(self) -> Optional[ChainlinkSnapshot]:
        with self._lock:
            return self._latest

    def get_slot_open_price(self) -> Optional[SlotOpenPrice]:
        with self._lock:
            return self._slot_open

    def get_reference_price(self) -> Optional[float]:
        """Return the slot-open price (price to beat) if available."""
        with self._lock:
            if self._slot_open is not None:
                return self._slot_open.price
            return None

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

    def is_connecting(self) -> bool:
        """Return True if the feed has started but not yet received any message.

        Distinguishes the startup window from a feed that was live and went stale.
        """
        with self._lock:
            return self._latest is None

    def get_recent_prices(self, window_s: int = 300) -> List[Tuple[float, float]]:
        """Return (timestamp, price) pairs for the last N seconds."""
        cutoff = time.time() - window_s
        with self._lock:
            return [(ts, p) for ts, p in self._buffer if ts >= cutoff]

    def get_earliest_slot_price(self, slot_ts: int) -> Optional[float]:
        """Return the earliest buffered price at or after slot_ts.

        Used as a fallback when _slot_open is missing for the current slot but
        the feed received ticks after the slot boundary (e.g. mid-slot reconnect).
        The buffer is chronological, so the first match is the earliest tick.
        """
        with self._lock:
            for ts, price in self._buffer:
                if ts >= slot_ts:
                    return price
        return None

    # ── Message handling ──────────────────────────────────────────────────

    def _handle_message(self, raw: str) -> None:
        """Parse a raw JSON RTDS message."""
        if not raw or not str(raw).strip():
            return
        if str(raw).strip().upper() == "PONG":
            return

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self._logger.debug(f"Chainlink: bad JSON")
            return

        # RTDS messages may arrive in the legacy [topic, payload] format or
        # the current {topic, type, timestamp, payload} format.
        message: Any = None
        topic = ""
        if isinstance(data, list) and len(data) >= 2:
            topic = str(data[0])
            message = data[1]
        elif isinstance(data, dict):
            topic = str(data.get("topic", ""))
            message = data
        else:
            return

        payload = message.get("payload", message) if isinstance(message, dict) else message
        ticks = self._extract_ticks(topic, payload, message)
        if not ticks:
            return

        with self._lock:
            for symbol, price, exchange_ts in ticks:
                self._record_tick(symbol, price, exchange_ts, time.time())

    def _extract_ticks(
        self,
        topic: str,
        payload: Any,
        message: Any,
    ) -> List[Tuple[str, float, float]]:
        """Extract one or more price ticks from an RTDS message."""
        topic_str = str(topic or "")
        if topic_str and topic_str not in {_RTDS_TOPIC, _RTDS_TOPIC_FALLBACK}:
            return []
        if not isinstance(payload, dict):
            return []

        symbol = str(payload.get("symbol", "")).lower()

        # Subscription acknowledgements currently arrive as a snapshot with
        # payload = {"symbol": "...", "data": [{"timestamp": ..., "value": ...}, ...]}
        if isinstance(payload.get("data"), list):
            if symbol != self._symbol:
                return []
            ticks: List[Tuple[str, float, float]] = []
            for point in payload["data"]:
                if not isinstance(point, dict):
                    continue
                try:
                    price = float(point["value"])
                    exchange_ts = self._normalize_timestamp(point["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                ticks.append((symbol, price, exchange_ts))
            return ticks

        try:
            if symbol != self._symbol:
                return []
            price = float(payload["value"])
            raw_ts = payload.get("timestamp")
            if raw_ts is None and isinstance(message, dict):
                raw_ts = message.get("timestamp")
            exchange_ts = self._normalize_timestamp(raw_ts)
        except (KeyError, TypeError, ValueError) as e:
            self._logger.debug(f"Chainlink: bad payload: {e}")
            return []
        return [(symbol, price, exchange_ts)]

    @staticmethod
    def _normalize_timestamp(ts: Any) -> float:
        """Normalize RTDS timestamps to seconds."""
        if ts is None:
            return time.time()
        value = float(ts)
        if value > 1e11:
            value /= 1000.0
        return value

    def _record_tick(
        self,
        symbol: str,
        price: float,
        exchange_ts: float,
        local_ts: float,
    ) -> None:
        """Record a single normalized tick under the caller's lock."""
        snap = ChainlinkSnapshot(
            symbol=symbol,
            price=price,
            exchange_ts=exchange_ts,
            local_ts=local_ts,
        )
        self._latest = snap
        self._buffer.append((local_ts, price))

        cutoff = local_ts - self._buffer_s
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        interval = self._slot_interval_s
        current_slot = int(math.floor(exchange_ts / interval) * interval)
        if self._slot_open is None or self._slot_open.slot_ts != current_slot:
            self._slot_open = SlotOpenPrice(
                slot_ts=current_slot,
                price=price,
                captured_at=local_ts,
            )
            self._logger.info(
                f"New slot {current_slot}: Chainlink open price = "
                f"${price:,.2f}"
            )

    # ── WebSocket loop ────────────────────────────────────────────────────

    def _ws_loop(self) -> None:
        """Run async WebSocket in a dedicated event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_connect_loop())
        except Exception as e:
            self._logger.error(f"Chainlink WS loop crashed: {e}")
        finally:
            loop.close()

    async def _ws_connect_loop(self) -> None:
        """Connect, subscribe, receive messages, and reconnect on failure."""
        try:
            import websockets
        except ImportError:
            self._logger.error("websockets package not installed (pip install websockets)")
            return

        backoff = _BACKOFF_INIT
        first_connect = True

        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(
                    _RTDS_URL, ping_interval=20, ping_timeout=10
                ) as ws:
                    self._logger.info(f"Connected to Chainlink RTDS ({_RTDS_URL})")
                    if not first_connect:
                        self.reconnect_count += 1
                    first_connect = False
                    backoff = _BACKOFF_INIT
                    last_message_at = time.monotonic()

                    # RTDS currently expects action/subscriptions and a
                    # JSON-string filter for Chainlink symbols.
                    subscribe_msg = json.dumps({
                        "action": "subscribe",
                        "subscriptions": [
                            {
                                "topic": _RTDS_TOPIC,
                                "type": "*",
                                "filters": json.dumps({"symbol": self._symbol}),
                            }
                        ],
                    })
                    await ws.send(subscribe_msg)
                    self._logger.info(
                        f"Subscribed to Chainlink {self._symbol}"
                    )

                    async def _ping_loop() -> None:
                        while not self._stop_evt.is_set():
                            await asyncio.sleep(_PING_INTERVAL_S)
                            await ws.send("PING")

                    ping_task = asyncio.create_task(_ping_loop())
                    while not self._stop_evt.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            last_message_at = time.monotonic()
                            self._handle_message(raw)
                        except asyncio.TimeoutError:
                            if (time.monotonic() - last_message_at) > self._reconnect_s:
                                self._logger.warning(
                                    f"Chainlink feed quiet for {self._reconnect_s:.0f}s; reconnecting"
                                )
                                break
                            continue
                    ping_task.cancel()
                    try:
                        await ping_task
                    except BaseException:
                        pass

            except Exception as e:
                if self._stop_evt.is_set():
                    break
                self._logger.warning(
                    f"Chainlink WS error: {e}, reconnecting in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    # ── Watchdog ──────────────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Poll feed age and warn if stale."""
        while not self._stop_evt.is_set():
            age_ms = self.get_feed_age_ms()
            if age_ms is not None:
                age_s = age_ms / 1000
                if age_s > self._reconnect_s:
                    self._logger.warning(
                        f"Chainlink feed stale ({age_s:.1f}s), will reconnect"
                    )
                elif age_s > self._stale_warn_s:
                    self._logger.debug(
                        f"Chainlink feed age {age_s:.1f}s (warn threshold)"
                    )
            self._stop_evt.wait(0.5)
