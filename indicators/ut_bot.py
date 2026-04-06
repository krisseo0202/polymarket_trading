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

        # ATR
        tr = np.empty(n, dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr = self._ema(tr, self.atr_period)
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

        return IndicatorResult(
            indicator_name="UTBot",
            timeframe=timeframe,
            values={
                "trail": trail,
                "buy": buy,
                "sell": sell,
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
