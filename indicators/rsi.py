"""Classic Wilder RSI indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorConfig, IndicatorResult


class RSIIndicator(Indicator):
    """Relative Strength Index with Wilder's smoothing (alpha = 1 / period).

    Output:
        values["rsi"] — np.ndarray of length N; NaN for the first `period` bars
                        where Wilder's seed average isn't yet available.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(config)
        self.period: int = int(self.params.get("period", 14))
        if self.period < 2:
            raise ValueError("RSI period must be >= 2")

    def compute(self, ohlc: pd.DataFrame, timeframe: str = "unknown") -> IndicatorResult:
        self.validate_ohlc(ohlc)
        close = ohlc["close"].astype(float).to_numpy()
        rsi = _wilder_rsi(close, self.period)
        rsi_centered = rsi - 50.0
        return IndicatorResult(
            indicator_name="RSI",
            timeframe=timeframe,
            values={"rsi": rsi, "rsi_centered": rsi_centered},
            signals=[],
        )


def _wilder_rsi(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    rsi = np.full(n, np.nan, dtype=float)
    if n <= period:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    rsi[period] = _rsi_from_avg(avg_gain, avg_loss)

    for i in range(period + 1, n):
        # Wilder's smoothing: avg_new = (avg_prev * (period - 1) + current) / period
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rsi[i] = _rsi_from_avg(avg_gain, avg_loss)

    return rsi


def _rsi_from_avg(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
