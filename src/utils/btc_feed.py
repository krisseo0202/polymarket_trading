"""
btc_feed.py — Production BTC/USDT real-time price feed via Binance WebSocket.

Connects to the Binance bookTicker stream for best bid/ask updates, maintains
a thread-safe state store, a rolling 5-minute price buffer, stale-data
detection, and automatic reconnection.

Dependency:
    pip install websockets

Usage:
    feed = BtcPriceFeed()
    feed.start()
    mid = feed.get_latest_mid()   # None until first tick arrives
    feed.stop()
"""

import asyncio
import collections
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import websockets


# ── Constants ─────────────────────────────────────────────────────────────────

# Binance bookTicker: fires on every best-bid or best-ask change.
# Docs: https://binance-docs.github.io/apidocs/spot/en/#individual-symbol-book-ticker-streams
_WS_URL_TEMPLATE = "wss://stream.binance.com:9443/ws/{symbol}@bookTicker"

_STALE_WARN_S  = 2.0    # log a warning if no message for this many seconds
_RECONNECT_S   = 5.0    # force a reconnect if no message for this many seconds
_BUFFER_S      = 300.0  # rolling window kept in memory (5 minutes)
_BACKOFF_INIT  = 1.0    # first reconnect wait (seconds)
_BACKOFF_MAX   = 60.0   # cap on exponential backoff


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BookSnapshot:
    """
    Immutable snapshot of the best bid/ask at a single point in time.

    Note: Binance bookTicker does not include a server-side timestamp, so
    ``exchange_ts`` is None. Use ``local_ts`` (time.time() at receipt) for
    latency-sensitive logic.
    """
    bid: float                    # best bid price (USD)
    ask: float                    # best ask price (USD)
    mid: float                    # (bid + ask) / 2
    exchange_ts: Optional[float]  # server timestamp (None for bookTicker)
    local_ts: float               # time.time() when this message was received


# ── Main class ────────────────────────────────────────────────────────────────

class BtcPriceFeed:
    """
    Real-time BTC/USDT price feed backed by Binance bookTicker WebSocket.

    Threading model:
        Main thread  ──►  calls public API (get_latest_mid, etc.)
        ws thread    ──►  asyncio loop running _ws_loop()
        watchdog     ──►  checks staleness every 0.5 s; triggers reconnect

    All shared state is protected by a single threading.Lock.

    Example::

        feed = BtcPriceFeed()
        feed.start()
        time.sleep(2)
        print(feed.get_latest_mid())
        feed.stop()
    """

    def __init__(
        self,
        symbol: str = "btcusdt",
        logger: Optional[logging.Logger] = None,
        stale_warn_s: float = _STALE_WARN_S,
        reconnect_s: float = _RECONNECT_S,
        buffer_s: float = _BUFFER_S,
    ) -> None:
        """
        Args:
            symbol:      Binance symbol (case-insensitive). Default: "btcusdt".
            logger:      Optional logger. Creates one named ``btc_feed`` if omitted.
            stale_warn_s: Seconds of silence before a stale-feed warning is logged.
            reconnect_s:  Seconds of silence before forcing a reconnect.
            buffer_s:     Length of the rolling mid-price buffer in seconds.
        """
        self._symbol      = symbol.lower()
        self._url         = _WS_URL_TEMPLATE.format(symbol=self._symbol)
        self._log         = logger or logging.getLogger("btc_feed")
        self._stale_warn_s = stale_warn_s
        self._reconnect_s  = reconnect_s
        self._buffer_s     = buffer_s

        # ── Shared state (always accessed under _lock) ────────────────────
        self._lock   = threading.Lock()
        self._latest: Optional[BookSnapshot]      = None
        self._buffer: collections.deque           = collections.deque()
        # deque items: (local_ts: float, mid: float)

        # ── Control events ────────────────────────────────────────────────
        self._stop_event      = threading.Event()
        self._reconnect_event = threading.Event()  # watchdog → ws thread

        # ── Observable counters (read by dashboard / external monitors) ───
        self.reconnect_count: int  = 0     # incremented on every reconnect (not first connect)
        self._connected_once: bool = False  # guards the above so first connect is not counted

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "BtcPriceFeed":
        """Start the WebSocket and watchdog threads. Returns self for chaining."""
        self._stop_event.clear()
        self._start_ws_thread()
        self._start_watchdog_thread()
        self._log.info(f"BtcPriceFeed started ({self._symbol})")
        return self

    def stop(self) -> None:
        """Signal all background threads to exit. Non-blocking."""
        self._stop_event.set()
        self._log.info("BtcPriceFeed stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_latest_book(self) -> Optional[BookSnapshot]:
        """Return the most recent BookSnapshot, or None if no data has arrived."""
        with self._lock:
            return self._latest

    def get_latest_mid(self) -> Optional[float]:
        """Return the latest mid-price (USD), or None if no data has arrived."""
        with self._lock:
            return self._latest.mid if self._latest else None

    def get_feed_age_ms(self) -> Optional[float]:
        """
        Milliseconds elapsed since the last received message.
        Returns None if no message has been received yet.
        """
        with self._lock:
            if self._latest is None:
                return None
            return (time.time() - self._latest.local_ts) * 1_000.0

    def is_healthy(self) -> bool:
        """
        True if a message has been received within the stale-warning threshold.
        False if no data has arrived yet or the feed is stale.
        """
        age_ms = self.get_feed_age_ms()
        if age_ms is None:
            return False
        return age_ms < self._stale_warn_s * 1_000.0

    def get_recent_prices(self, seconds: float) -> List[Tuple[float, float]]:
        """
        Return all buffered (local_ts, mid) pairs from the last ``seconds`` seconds.

        List is ordered oldest-first. Each entry is a tuple of:
            (local_ts: float, mid: float)
        """
        cutoff = time.time() - seconds
        with self._lock:
            return [(ts, mid) for ts, mid in self._buffer if ts >= cutoff]

    # ── Internal: message parsing ─────────────────────────────────────────────

    def _handle_message(self, raw: str) -> None:
        """Parse one raw bookTicker message and update shared state."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._log.warning(f"JSON parse error: {exc}")
            return

        bid_s = data.get("b")
        ask_s = data.get("a")
        if not bid_s or not ask_s:
            # Not a bookTicker frame (e.g. subscription ack). Skip silently.
            return

        try:
            bid = float(bid_s)
            ask = float(ask_s)
        except ValueError as exc:
            self._log.warning(f"Price conversion error: {exc} | raw={raw[:120]}")
            return

        mid  = (bid + ask) / 2.0
        now  = time.time()
        snap = BookSnapshot(bid=bid, ask=ask, mid=mid, exchange_ts=None, local_ts=now)

        with self._lock:
            self._latest = snap
            self._buffer.append((now, mid))
            # Trim entries older than the rolling window
            cutoff = now - self._buffer_s
            while self._buffer and self._buffer[0][0] < cutoff:
                self._buffer.popleft()

    # ── Internal: WebSocket asyncio loop ─────────────────────────────────────

    def _start_ws_thread(self) -> None:
        """Launch the asyncio event loop in a daemon thread."""
        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._ws_loop())
            finally:
                loop.close()

        threading.Thread(
            target=_run,
            daemon=True,
            name=f"btcfeed-ws-{self._symbol}",
        ).start()

    async def _ws_loop(self) -> None:
        """
        Outer reconnect loop. Connects to Binance, reads messages until the
        watchdog requests a reconnect or the stop event fires, then
        waits with exponential backoff before reconnecting.
        """
        backoff = _BACKOFF_INIT

        while not self._stop_event.is_set():
            self._reconnect_event.clear()
            try:
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._log.info(f"WS connected: {self._url}")
                    if self._connected_once:
                        self.reconnect_count += 1
                    self._connected_once = True
                    backoff = _BACKOFF_INIT  # successful connection resets backoff

                    await self._ws_read_loop(ws)

            except Exception as exc:
                if self._stop_event.is_set():
                    return
                self._log.warning(
                    f"WS error ({type(exc).__name__}: {exc}) — "
                    f"reconnecting in {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    async def _ws_read_loop(self, ws) -> None:
        """
        Inner read loop for an established WebSocket connection.
        Exits when the stop event fires or the watchdog sets the reconnect event.
        Uses a 1-second recv() timeout so both events are checked regularly.
        """
        while not self._stop_event.is_set():
            if self._reconnect_event.is_set():
                self._log.warning("Watchdog triggered reconnect — dropping connection")
                return

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                self._handle_message(raw)
            except asyncio.TimeoutError:
                # No message in 1 s — loop back to check control events.
                continue

    # ── Internal: watchdog thread ─────────────────────────────────────────────

    def _start_watchdog_thread(self) -> None:
        """
        Daemon thread that polls feed age every 0.5 s.
        Logs a warning at _stale_warn_s and forces a reconnect at _reconnect_s.
        """
        def _watch() -> None:
            stale_logged = False

            while not self._stop_event.is_set():
                time.sleep(0.5)
                age_ms = self.get_feed_age_ms()

                if age_ms is None:
                    # No data yet — nothing to check.
                    continue

                age_s = age_ms / 1_000.0

                if age_s >= self._reconnect_s:
                    self._log.warning(
                        f"No message for {age_s:.1f}s — signalling reconnect"
                    )
                    self._reconnect_event.set()
                    stale_logged = False  # reset so we log again after reconnect
                elif age_s >= self._stale_warn_s:
                    if not stale_logged:
                        self._log.warning(f"Stale feed: no message for {age_s:.1f}s")
                        stale_logged = True
                else:
                    stale_logged = False

        threading.Thread(
            target=_watch,
            daemon=True,
            name=f"btcfeed-watchdog-{self._symbol}",
        ).start()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Runnable demo. Prints feed health once per second for 30 seconds.

        python src/utils/btc_feed.py
        python -m src.utils.btc_feed
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("btc_feed.demo")

    feed = BtcPriceFeed(logger=log)
    feed.start()
    log.info("Waiting for first tick …  (Ctrl-C to quit)")

    try:
        for tick in range(30):
            time.sleep(1)
            book    = feed.get_latest_book()
            age_ms  = feed.get_feed_age_ms()
            buf_60s = feed.get_recent_prices(60)

            if book is None:
                log.info(f"[{tick+1:02d}s] — no data yet")
                continue

            log.info(
                f"[{tick+1:02d}s] "
                f"bid={book.bid:>10,.2f}  "
                f"ask={book.ask:>10,.2f}  "
                f"mid={book.mid:>10,.2f}  "
                f"age={age_ms:>5.0f}ms  "
                f"healthy={feed.is_healthy()}  "
                f"60s_buf={len(buf_60s)} ticks"
            )
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()


if __name__ == "__main__":
    main()
