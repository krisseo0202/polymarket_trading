"""TDRSIStrategy — TD Sequential + RSI entry rule wrapped as a bot Strategy.

Decision logic lives in td_rsi_decide.decide(). This class handles:
  - BTC OHLC construction from btc_feed
  - TD setup count and RSI computation per timeframe
  - Signal emission with fixed position sizing
  - Exit via base-class check_exit() (stop-loss + time-limit)

Run with:
    python bot.py --strategy td_rsi --paper
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Signal, Strategy
from .td_rsi_decide import decide
from ..api.types import OrderBook
from ..models.feature_builder import _prices_to_ohlc
from ..utils.market_utils import get_mid_price, round_to_tick

from indicators.base import IndicatorConfig
from indicators.td_sequential import TDSequentialIndicator

_TDS = TDSequentialIndicator(IndicatorConfig("TDSeq", {}))
_RSI_PERIOD = 14


# ── helpers (mirrors scripts/backtest_td_rsi.py) ──────────────────────────────

def _compute_rsi(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _signed_td(ohlc: pd.DataFrame) -> int:
    """Signed TD setup count for the latest bar.
    negative → buy setup (TD Down), positive → sell setup (TD Up), 0 → flat.
    """
    if len(ohlc) < 6:
        return 0
    vals = _TDS.compute(ohlc).values
    bull = int(vals["bullish_setup_count"][-1])
    bear = int(vals["bearish_setup_count"][-1])
    if bull > 0 and bull >= bear:
        return -bull
    if bear > 0:
        return bear
    return 0


def _resample_ohlc(ohlc_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}min"
    r = ohlc_1m.resample(rule)
    return pd.DataFrame({
        "open":  r["open"].first(),
        "high":  r["high"].max(),
        "low":   r["low"].min(),
        "close": r["close"].last(),
    }).dropna(how="all")


# ── strategy class ────────────────────────────────────────────────────────────

class TDRSIStrategy(Strategy):
    """Pure technical entry: TD Sequential setup exhaustion + RSI confirmation.

    Weights: 30s=4 (unavailable from btc_feed), 1m=3, 3m=2, 5m=1
    Minimum weighted score to trade: 4
    No Kelly sizing — fixed position_size_usdc from config.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        btc_feed=None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="td_rsi", config=config)
        self.logger   = logger or logging.getLogger(__name__)
        self.btc_feed = btc_feed

        self.position_size_usdc: float = float(config.get("position_size_usdc", 10.0))
        self.stop_loss_pct:      float = float(config.get("stop_loss_pct", 0.15))
        self.max_hold_seconds:   int   = int(config.get("max_hold_seconds", 270))
        self.profit_target_pct:  float = 999.0   # hold to expiry; no take-profit

        # Observable state (read by dashboard)
        self.last_decision:  str           = "NO_TRADE"
        self.last_td_1m:     Optional[int] = None
        self.last_rsi_1m:    Optional[float] = None
        self.last_td_3m:     Optional[int] = None
        self.last_rsi_3m:    Optional[float] = None
        self.last_td_5m:     Optional[int] = None
        self.last_rsi_5m:    Optional[float] = None
        self.last_tte:       Optional[float] = None

    # ── core interface ────────────────────────────────────────────────────────

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        if not self._yes_token_id or not self._no_token_id:
            return []

        order_books: Dict[str, OrderBook] = market_data.get("order_books", {})
        positions  = market_data.get("positions", [])
        by_token   = {p.token_id: p for p in positions}
        now_mono   = time.monotonic()

        self._auto_recover_position(by_token, now_mono)

        # Exit check takes priority
        if self.active_token_id is not None:
            exit_sig = self.check_exit(order_books, by_token, now_mono)
            if exit_sig:
                self._reset_position_state()
                return [exit_sig]
            return []

        # Build snapshot
        snapshot = self._build_snapshot(market_data)
        if snapshot is None:
            return []

        decision = decide(snapshot)
        self.last_decision = decision

        if decision == "NO_TRADE":
            return []

        outcome  = "YES" if decision == "BUY_YES" else "NO"
        token_id = self._yes_token_id if outcome == "YES" else self._no_token_id
        book     = order_books.get(token_id)
        if book is None:
            return []

        ask = float(book.asks[0].price) if book.asks else None
        if ask is None:
            mid = get_mid_price(book)
            if mid is None:
                return []
            ask = mid
        if ask <= 0 or ask >= 1:
            return []

        size = round(self.position_size_usdc / ask, 2)
        if size <= 0:
            return []

        tick = book.tick_size or 0.001
        self.active_token_id  = token_id
        self.entry_price      = ask
        self.entry_timestamp  = now_mono
        self.entry_size       = size

        tte = snapshot.get("time_to_expiry_seconds", 0)
        return [Signal(
            market_id  = self._market_id,
            outcome    = outcome,
            action     = "BUY",
            confidence = 0.7,
            price      = round_to_tick(ask, tick),
            size       = size,
            reason     = (
                f"td_rsi decision={decision} tte={tte:.0f}s "
                f"td1m={self.last_td_1m} rsi1m={self.last_rsi_1m:.1f}"
                if self.last_rsi_1m is not None
                else f"td_rsi decision={decision} tte={tte:.0f}s"
            ),
        )]

    def should_enter(self, signal: Signal) -> bool:
        return self.validate_signal(signal) and signal.confidence >= self.min_confidence

    # ── snapshot builder ──────────────────────────────────────────────────────

    def _build_snapshot(self, market_data: Dict[str, Any]) -> Optional[dict]:
        """Fetch BTC prices, build OHLC, compute TD + RSI, return snapshot dict."""
        btc_prices = []
        if self.btc_feed is not None and getattr(self.btc_feed, "is_healthy", lambda: False)():
            btc_prices = getattr(self.btc_feed, "get_recent_prices", lambda w=5400: [])(5400)

        if not btc_prices:
            self.logger.debug("td_rsi: no BTC price data available")
            return None

        slot_expiry_ts = float(market_data.get("slot_expiry_ts") or 0.0)
        now_wall       = time.time()
        tte            = max(0.0, slot_expiry_ts - now_wall)
        self.last_tte  = tte

        ohlc_1m = _prices_to_ohlc(btc_prices, bar_seconds=60)
        if len(ohlc_1m) < 6:
            self.logger.debug("td_rsi: insufficient 1m bars (%d)", len(ohlc_1m))
            return None

        snap: dict = {"time_to_expiry_seconds": tte}

        for tf_label, ohlc in [
            ("1m", ohlc_1m),
            ("3m", _resample_ohlc(ohlc_1m, 3)),
            ("5m", _resample_ohlc(ohlc_1m, 5)),
        ]:
            if len(ohlc) < 6:
                continue
            td  = _signed_td(ohlc)
            rsi_series = _compute_rsi(ohlc["close"])
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
            if rsi is not None and np.isnan(rsi):
                rsi = None
            snap[f"td_{tf_label}"]  = td
            snap[f"rsi_{tf_label}"] = rsi

        self.last_td_1m  = snap.get("td_1m")
        self.last_rsi_1m = snap.get("rsi_1m")
        self.last_td_3m  = snap.get("td_3m")
        self.last_rsi_3m = snap.get("rsi_3m")
        self.last_td_5m  = snap.get("td_5m")
        self.last_rsi_5m = snap.get("rsi_5m")

        return snap
