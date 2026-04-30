"""Tests for UTBotIndicator — verifies Pine Script parity (RMA-based ATR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.base import IndicatorConfig
from indicators.ut_bot import UTBotIndicator


def _ohlc_from_closes(closes: list) -> pd.DataFrame:
    """Build an OHLC frame from a close series using flat bars (H=L=O=C)."""
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": arr, "high": arr, "low": arr, "close": arr,
    })


def _indicator(atr_period: int = 10, key_value: float = 1.0) -> UTBotIndicator:
    return UTBotIndicator(IndicatorConfig(
        "UTBot", {"atr_period": atr_period, "key_value": key_value},
    ))


# ---------------------------------------------------------------------------
# RMA parity
# ---------------------------------------------------------------------------


def test_rma_seed_matches_sma_at_period_minus_one():
    """First period bars: running SMA. At index period-1 exactly equals SMA of the window."""
    arr = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0], dtype=float)
    out = UTBotIndicator._rma(arr, period=5)
    # indices 0..3: running SMA
    assert out[0] == 10.0
    assert out[1] == pytest.approx((10 + 12) / 2)
    assert out[2] == pytest.approx((10 + 12 + 14) / 3)
    assert out[3] == pytest.approx((10 + 12 + 14 + 16) / 4)
    # index 4 == SMA of first 5 bars (the seed)
    assert out[4] == pytest.approx((10 + 12 + 14 + 16 + 18) / 5)


def test_rma_recursion_matches_wilder_formula():
    """After seed, each bar: rma = 1/p * src + (p-1)/p * rma_prev."""
    arr = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0], dtype=float)
    out = UTBotIndicator._rma(arr, period=5)
    # Seed at index 4: mean of first 5 = 14.0
    seed = 14.0
    # Index 5: (1/5) * 20 + (4/5) * 14 = 4 + 11.2 = 15.2
    assert out[5] == pytest.approx(1 / 5 * 20.0 + 4 / 5 * seed)
    # Index 6: (1/5) * 22 + (4/5) * 15.2 = 4.4 + 12.16 = 16.56
    assert out[6] == pytest.approx(1 / 5 * 22.0 + 4 / 5 * out[5])


def test_rma_slower_than_ema_for_trending_input():
    """Wilder's α=1/p is smaller than EMA's α=2/(p+1) so RMA lags more."""
    arr = np.arange(1.0, 101.0)
    rma = UTBotIndicator._rma(arr, period=10)
    ema = UTBotIndicator._ema(arr, period=10)
    # On a rising series, EMA tracks the current value more closely than RMA.
    # Check at several post-warmup indices.
    for i in (20, 50, 99):
        assert ema[i] > rma[i], f"EMA should lead RMA on uptrend at i={i}"


def test_rma_period_one_is_identity():
    arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(UTBotIndicator._rma(arr, period=1), arr)


def test_rma_short_input_returns_running_sma():
    """n < period: output = running SMA, never NaN."""
    arr = np.array([10.0, 20.0, 30.0])
    out = UTBotIndicator._rma(arr, period=5)
    assert out[0] == 10.0
    assert out[1] == 15.0
    assert out[2] == 20.0


# ---------------------------------------------------------------------------
# Full UT Bot signal flow
# ---------------------------------------------------------------------------


def test_compute_returns_trail_buy_sell_arrays():
    ohlc = _ohlc_from_closes([100.0] * 30)
    ind = _indicator(atr_period=10, key_value=1.0)
    result = ind.compute(ohlc, timeframe="test")

    for key in ("trail", "buy", "sell"):
        assert key in result.values
        assert len(result.values[key]) == 30


def test_flat_price_series_produces_no_signals():
    """Constant price → ATR = 0 → trail tracks price → no crossovers."""
    ohlc = _ohlc_from_closes([100.0] * 50)
    ind = _indicator(atr_period=10, key_value=1.0)
    result = ind.compute(ohlc, timeframe="test")
    assert not result.values["buy"].any()
    assert not result.values["sell"].any()


def test_uptrend_then_drop_fires_sell_signal():
    """Strong uptrend sets an uptrend trailing stop; a sharp drop must cross it."""
    # 40 rising bars, then a cliff down.
    closes = list(np.linspace(100.0, 120.0, 40)) + [118.0, 116.0, 112.0, 108.0, 100.0]
    ohlc = _ohlc_from_closes(closes)
    ind = _indicator(atr_period=10, key_value=1.0)
    result = ind.compute(ohlc, timeframe="test")

    # At least one sell fires on the drop; no buys after the initial uptrend.
    assert result.values["sell"][40:].any()


def test_downtrend_then_rally_fires_buy_signal():
    """Downtrend pins trail above price; rally above trail triggers buy."""
    closes = list(np.linspace(100.0, 80.0, 40)) + [82.0, 85.0, 90.0, 95.0, 100.0]
    ohlc = _ohlc_from_closes(closes)
    ind = _indicator(atr_period=10, key_value=1.0)
    result = ind.compute(ohlc, timeframe="test")

    assert result.values["buy"][40:].any()


def test_buy_and_sell_are_mutually_exclusive_per_bar():
    """No bar can simultaneously be a buy AND a sell."""
    rng = np.random.default_rng(0)
    closes = 100.0 + np.cumsum(rng.standard_normal(200))
    ohlc = _ohlc_from_closes(closes.tolist())
    ind = _indicator(atr_period=10, key_value=1.0)
    result = ind.compute(ohlc, timeframe="test")

    assert not (result.values["buy"] & result.values["sell"]).any()
