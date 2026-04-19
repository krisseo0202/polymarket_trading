"""Tests for RSI indicator — golden series match and edge cases."""

import numpy as np
import pandas as pd
import pytest

from indicators.base import IndicatorConfig
from indicators.rsi import RSIIndicator, _wilder_rsi


def _frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
    })


def test_rsi_matches_wilder_golden_14():
    # Canonical Wilder RSI(14) example. Source: Wilder, "New Concepts in
    # Technical Trading Systems" (1978), p. 65 table. Expected first RSI
    # at index 14 ≈ 70.53, second ≈ 66.32.
    closes = [
        44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955,
        45.4245, 45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820,
        46.2820, 46.0028, 46.0328, 46.4116, 46.2222, 45.6439, 46.2122,
    ]
    rsi = _wilder_rsi(np.asarray(closes, dtype=float), period=14)
    # NaN for first 14 bars; first real value at index 14.
    assert np.all(np.isnan(rsi[:14]))
    assert rsi[14] == pytest.approx(70.53, abs=0.05)
    assert rsi[15] == pytest.approx(66.32, abs=0.1)


def test_rsi_all_up_returns_100():
    closes = np.linspace(100.0, 200.0, 30)
    rsi = _wilder_rsi(closes, period=14)
    # Strictly rising → losses==0 → RSI=100 once warmed.
    assert rsi[-1] == 100.0


def test_rsi_all_down_returns_0():
    closes = np.linspace(200.0, 100.0, 30)
    rsi = _wilder_rsi(closes, period=14)
    assert rsi[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_flat_returns_50():
    closes = np.full(30, 100.0)
    rsi = _wilder_rsi(closes, period=14)
    # zero gain AND zero loss → treat as neutral 50 (not 100).
    assert rsi[-1] == 50.0


def test_rsi_output_in_range_on_noise():
    rng = np.random.default_rng(42)
    closes = 100.0 + rng.normal(0, 1, size=200).cumsum()
    rsi = _wilder_rsi(closes, period=14)
    real = rsi[~np.isnan(rsi)]
    assert ((real >= 0) & (real <= 100)).all()


def test_rsi_nan_when_insufficient_bars():
    rsi = _wilder_rsi(np.array([1.0, 2.0, 3.0]), period=14)
    assert np.all(np.isnan(rsi))


def test_rsi_indicator_wrapper_returns_result():
    closes = np.linspace(100.0, 150.0, 30)
    ind = RSIIndicator(IndicatorConfig(name="RSI", params={"period": 14}))
    result = ind.compute(_frame(closes), timeframe="5m")
    assert result.indicator_name == "RSI"
    assert result.timeframe == "5m"
    assert "rsi" in result.values
    assert len(result.values["rsi"]) == len(closes)


def test_rsi_rejects_period_below_two():
    with pytest.raises(ValueError):
        RSIIndicator(IndicatorConfig(name="RSI", params={"period": 1}))
