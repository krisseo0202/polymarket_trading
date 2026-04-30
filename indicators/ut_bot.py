"""UT Bot Alert indicator (ATR trailing stop with EMA filter).

Replicates the TradingView "UT Bot Alerts" by QuantNomad:
  - ATR-based trailing stop
  - Buy signal when price crosses above trailing stop
  - Sell signal when price crosses below trailing stop
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorConfig, IndicatorResult


class UTBotIndicator(Indicator):
    def __init__(self, config: IndicatorConfig):
        super().__init__(config)
        self.atr_period: int = self.params.get("atr_period", 10)
        self.key_value: float = self.params.get("key_value", 1.0)
        self.ema_period: int = self.params.get("ema_period", 1)

    def compute(self, ohlc: pd.DataFrame, timeframe: str = "unknown") -> IndicatorResult:
        self.validate_ohlc(ohlc)
        n = len(ohlc)

        close = ohlc["close"].astype(float).to_numpy()
        high = ohlc["high"].astype(float).to_numpy()
        low = ohlc["low"].astype(float).to_numpy()

        # EMA of close (default period=1 means raw close)
        src = self._ema(close, self.ema_period)

        # ATR — Wilder's smoothing (RMA), matching TradingView's atr() which
        # internally calls rma(). Alpha = 1 / period (vs EMA's 2 / (period+1))
        # so ATR here is slower-responding than a plain EMA, producing the
        # same values Pine's UT Bot reads on TradingView charts.
        tr = np.empty(n, dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr = self._rma(tr, self.atr_period)
        n_loss = self.key_value * atr

        # Trailing stop
        trail = np.zeros(n, dtype=float)
        trail[0] = src[0]
        for i in range(1, n):
            if src[i] > trail[i - 1] and src[i - 1] > trail[i - 1]:
                trail[i] = max(trail[i - 1], src[i] - n_loss[i])
            elif src[i] < trail[i - 1] and src[i - 1] < trail[i - 1]:
                trail[i] = min(trail[i - 1], src[i] + n_loss[i])
            elif src[i] > trail[i - 1]:
                trail[i] = src[i] - n_loss[i]
            else:
                trail[i] = src[i] + n_loss[i]

        # Signals: cross above/below trailing stop
        buy = np.zeros(n, dtype=bool)
        sell = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if src[i] > trail[i] and src[i - 1] <= trail[i - 1]:
                buy[i] = True
            elif src[i] < trail[i] and src[i - 1] >= trail[i - 1]:
                sell[i] = True

        # Continuous features
        # Signed % gap to trailing stop: positive = price above trail (bullish)
        ut_distance = (close - trail) / close

        # Same gap normalized by ATR (avoid divide-by-zero on flat ATR)
        safe_atr = np.where(atr > 0, atr, np.nan)
        ut_atr_distance = (close - trail) / safe_atr

        # Bars since last buy / sell signal (sentinel 999 before first signal)
        bars_since_buy = np.full(n, 999, dtype=int)
        bars_since_sell = np.full(n, 999, dtype=int)
        last_buy = None
        last_sell = None
        for i in range(n):
            if buy[i]:
                last_buy = i
            if sell[i]:
                last_sell = i
            if last_buy is not None:
                bars_since_buy[i] = i - last_buy
            if last_sell is not None:
                bars_since_sell[i] = i - last_sell

        return IndicatorResult(
            indicator_name="UTBot",
            timeframe=timeframe,
            values={
                "trail": trail,
                "buy": buy,
                "sell": sell,
                "ut_distance": ut_distance,
                "ut_atr_distance": ut_atr_distance,
                "bars_since_buy": bars_since_buy,
                "bars_since_sell": bars_since_sell,
            },
            signals=[],
        )

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        if period <= 1:
            return arr.copy()
        out = np.empty_like(arr, dtype=float)
        alpha = 2.0 / (period + 1)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _rma(arr: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing (a.k.a. RMA) — matches TradingView's rma().

        Seed: SMA of first ``period`` values (Pine behavior). Early bars
        [0..period-2] use a running SMA so the output never contains NaN and
        downstream code can treat the whole series as a valid number. From
        index ``period-1`` onward, applies ``α = 1/period`` recursively.
        """
        n = len(arr)
        if period <= 1 or n == 0:
            return arr.copy()
        out = np.empty(n, dtype=float)
        alpha = 1.0 / period

        if n < period:
            csum = np.cumsum(arr, dtype=float)
            for i in range(n):
                out[i] = csum[i] / (i + 1)
            return out

        # Warmup: running SMA for indices [0..period-1], with true SMA-seed at period-1.
        csum = np.cumsum(arr[:period], dtype=float)
        for i in range(period - 1):
            out[i] = csum[i] / (i + 1)
        out[period - 1] = csum[-1] / period

        for i in range(period, n):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out
