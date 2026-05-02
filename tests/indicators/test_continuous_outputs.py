"""Tests for per-bar continuous output arrays on all indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.base import IndicatorConfig
from indicators.ut_bot import UTBotIndicator
from indicators.fvg import FVGIndicator
from indicators.td_sequential import TDSequentialIndicator
from indicators.rsi import RSIIndicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_ohlc(n: int, price: float = 100.0) -> pd.DataFrame:
    arr = np.full(n, price, dtype=float)
    return pd.DataFrame({"open": arr, "high": arr, "low": arr, "close": arr})


def _ohlc_from_closes(closes, high_add: float = 0.5, low_sub: float = 0.5) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open":  c,
        "high":  c + high_add,
        "low":   c - low_sub,
        "close": c,
    })


def _make_ut_bot(atr_period: int = 5, key_value: float = 1.0) -> UTBotIndicator:
    return UTBotIndicator(IndicatorConfig("UTBot", {"atr_period": atr_period, "key_value": key_value}))


def _make_fvg() -> FVGIndicator:
    return FVGIndicator(IndicatorConfig("FVG", {"threshold_percent": 0.0, "auto": False}))


def _make_tds() -> TDSequentialIndicator:
    return TDSequentialIndicator(IndicatorConfig("TDS", {}))


def _make_rsi(period: int = 14) -> RSIIndicator:
    return RSIIndicator(IndicatorConfig("RSI", {"period": period}))


# ---------------------------------------------------------------------------
# UT Bot — continuous distance arrays
# ---------------------------------------------------------------------------

class TestUTBotContinuous:
    def test_ut_distance_is_float_array_length_n(self):
        n = 60
        ohlc = _ohlc_from_closes(np.linspace(100, 120, n))
        result = _make_ut_bot().compute(ohlc)
        arr = result.values["ut_distance"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n
        assert arr.dtype.kind == "f"

    def test_ut_atr_distance_is_float_array_length_n(self):
        n = 60
        ohlc = _ohlc_from_closes(np.linspace(100, 120, n))
        result = _make_ut_bot().compute(ohlc)
        arr = result.values["ut_atr_distance"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n
        assert arr.dtype.kind == "f"

    def test_ut_distance_not_all_nan_after_warmup(self):
        n = 60
        ohlc = _ohlc_from_closes(np.linspace(100, 120, n))
        result = _make_ut_bot(atr_period=5).compute(ohlc)
        arr = result.values["ut_distance"]
        # After warmup period, values should be finite
        assert np.isfinite(arr[10:]).all(), "ut_distance has non-finite values after warmup"

    def test_ut_distance_sign_positive_when_price_above_trail(self):
        """At any bar where close > trail, ut_distance must be positive."""
        n = 80
        ohlc = _ohlc_from_closes(np.linspace(100, 140, n))
        result = _make_ut_bot(atr_period=5).compute(ohlc)
        trail = result.values["trail"]
        close = ohlc["close"].to_numpy()
        ut_dist = result.values["ut_distance"]
        above = close > trail
        assert above.any(), "need at least one bar where close > trail for this test"
        assert (ut_dist[above] > 0).all(), "ut_distance should be positive when close > trail"

    def test_ut_distance_sign_negative_when_price_below_trail(self):
        """At any bar where close < trail, ut_distance must be negative."""
        n = 80
        ohlc = _ohlc_from_closes(np.linspace(140, 100, n))
        result = _make_ut_bot(atr_period=5).compute(ohlc)
        trail = result.values["trail"]
        close = ohlc["close"].to_numpy()
        ut_dist = result.values["ut_distance"]
        below = close < trail
        assert below.any(), "need at least one bar where close < trail for this test"
        assert (ut_dist[below] < 0).all(), "ut_distance should be negative when close < trail"


# ---------------------------------------------------------------------------
# UT Bot — bars_since arrays
# ---------------------------------------------------------------------------

class TestUTBotBarsSince:
    def test_bars_since_buy_is_int_array_length_n(self):
        n = 60
        ohlc = _ohlc_from_closes(np.linspace(100, 120, n))
        result = _make_ut_bot().compute(ohlc)
        arr = result.values["bars_since_buy"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n
        assert np.issubdtype(arr.dtype, np.integer)

    def test_bars_since_sell_is_int_array_length_n(self):
        n = 60
        ohlc = _ohlc_from_closes(np.linspace(100, 120, n))
        result = _make_ut_bot().compute(ohlc)
        arr = result.values["bars_since_sell"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n
        assert np.issubdtype(arr.dtype, np.integer)

    def test_bars_since_buy_starts_at_sentinel_before_first_buy(self):
        """Before any buy signal, bars_since_buy should be 999."""
        n = 60
        ohlc = _flat_ohlc(n)  # flat → no signals
        result = _make_ut_bot().compute(ohlc)
        buy = result.values["buy"]
        bars = result.values["bars_since_buy"]
        # On a flat series no buy fires; all should be sentinel
        if not buy.any():
            assert (bars == 999).all()

    def test_bars_since_buy_zero_on_buy_bar(self):
        """On the bar where a buy fires, bars_since_buy == 0."""
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(100))
        ohlc = _ohlc_from_closes(closes)
        result = _make_ut_bot(atr_period=5).compute(ohlc)
        buy = result.values["buy"]
        bars = result.values["bars_since_buy"]
        if buy.any():
            buy_idx = np.where(buy)[0]
            for idx in buy_idx:
                assert bars[idx] == 0, f"bars_since_buy should be 0 at buy bar {idx}"

    def test_bars_since_buy_increments_after_buy(self):
        """After a buy bar, bars_since_buy should increase by 1 per bar."""
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(100))
        ohlc = _ohlc_from_closes(closes)
        result = _make_ut_bot(atr_period=5).compute(ohlc)
        buy = result.values["buy"]
        bars = result.values["bars_since_buy"]
        if buy.any():
            idx = np.where(buy)[0][0]
            # Check next few bars after the first buy (before any subsequent buy)
            for j in range(1, min(5, len(buy) - idx)):
                if not buy[idx + j]:
                    assert bars[idx + j] == j, (
                        f"bars_since_buy at {idx+j} should be {j}, got {bars[idx+j]}"
                    )
                else:
                    break  # another buy reset the counter


# ---------------------------------------------------------------------------
# FVG — per-bar count and distance arrays
# ---------------------------------------------------------------------------

class TestFVGContinuous:
    def _bullish_gap_ohlc(self, n: int = 80) -> pd.DataFrame:
        """Craft a series that produces at least one bullish FVG.

        Bar pattern for bullish FVG (Pine logic):
          low[i] > high[i-2]  AND  close[i-1] > high[i-2]
        We force this by inserting a gap-up sequence.
        """
        prices = list(np.linspace(100, 102, 40))
        # gap-up: bar i-2 high=102, bar i-1 close=103, bar i low=103.5
        prices += [102.0, 103.0, 103.5]  # gap-up cluster
        prices += list(np.linspace(103.5, 105, max(0, n - 43)))
        prices = prices[:n]
        high = np.array(prices) + 0.3
        low  = np.array(prices) - 0.3
        return pd.DataFrame({
            "open":  np.array(prices),
            "high":  high,
            "low":   low,
            "close": np.array(prices),
        })

    def test_bull_count_active_is_int_array_length_n(self):
        n = 80
        ohlc = self._bullish_gap_ohlc(n)
        result = _make_fvg().compute(ohlc)
        arr = result.values["bull_count_active"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n
        assert np.issubdtype(arr.dtype, np.integer)

    def test_bear_count_active_is_int_array_length_n(self):
        n = 80
        ohlc = self._bullish_gap_ohlc(n)
        result = _make_fvg().compute(ohlc)
        arr = result.values["bear_count_active"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n
        assert np.issubdtype(arr.dtype, np.integer)

    def test_dist_to_nearest_bull_fvg_nan_when_count_zero(self):
        """Wherever bull_count_active == 0, dist_to_nearest_bull_fvg must be NaN."""
        n = 80
        ohlc = self._bullish_gap_ohlc(n)
        result = _make_fvg().compute(ohlc)
        cnt = result.values["bull_count_active"]
        dist = result.values["dist_to_nearest_bull_fvg"]
        zero_mask = cnt == 0
        assert np.isnan(dist[zero_mask]).all(), (
            "dist_to_nearest_bull_fvg should be NaN when no active bull FVGs"
        )

    def test_dist_to_nearest_bear_fvg_nan_when_count_zero(self):
        """Wherever bear_count_active == 0, dist_to_nearest_bear_fvg must be NaN."""
        n = 80
        ohlc = self._bullish_gap_ohlc(n)
        result = _make_fvg().compute(ohlc)
        cnt = result.values["bear_count_active"]
        dist = result.values["dist_to_nearest_bear_fvg"]
        zero_mask = cnt == 0
        assert np.isnan(dist[zero_mask]).all()

    def test_nearest_fvg_age_bars_is_int_array_length_n(self):
        n = 80
        ohlc = self._bullish_gap_ohlc(n)
        result = _make_fvg().compute(ohlc)
        arr = result.values["nearest_fvg_age_bars"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n
        assert np.issubdtype(arr.dtype, np.integer)

    def test_existing_scalar_outputs_intact(self):
        """Existing scalar outputs must still be present and be ints."""
        n = 80
        ohlc = self._bullish_gap_ohlc(n)
        result = _make_fvg().compute(ohlc)
        for key in ("bull_count", "bear_count", "bull_mitigated", "bear_mitigated"):
            assert key in result.values
            assert isinstance(result.values[key], int)


# ---------------------------------------------------------------------------
# TD Sequential — dist_to_tdst_support / dist_to_tdst_resistance
# ---------------------------------------------------------------------------

class TestTDSequentialContinuous:
    def _tds_ohlc(self, n: int = 80) -> pd.DataFrame:
        """Zigzag that generates both buy-9 and sell-9 setups."""
        rng = np.random.default_rng(7)
        closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        return _ohlc_from_closes(closes)

    def test_dist_to_tdst_support_length_n(self):
        n = 80
        ohlc = self._tds_ohlc(n)
        result = _make_tds().compute(ohlc)
        arr = result.values["dist_to_tdst_support"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n

    def test_dist_to_tdst_resistance_length_n(self):
        n = 80
        ohlc = self._tds_ohlc(n)
        result = _make_tds().compute(ohlc)
        arr = result.values["dist_to_tdst_resistance"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n

    def test_dist_to_tdst_support_nan_when_no_level(self):
        """Before any bullish setup completes, support level is NaN → dist is NaN."""
        n = 80
        ohlc = self._tds_ohlc(n)
        result = _make_tds().compute(ohlc)
        support = result.values["tdst_support"]
        dist = result.values["dist_to_tdst_support"]
        no_level = np.isnan(support)
        assert np.isnan(dist[no_level]).all(), (
            "dist_to_tdst_support should be NaN where no support level exists"
        )

    def test_dist_to_tdst_support_positive_when_price_above_support(self):
        """When a support level exists and close > support, dist must be positive."""
        n = 80
        ohlc = self._tds_ohlc(n)
        result = _make_tds().compute(ohlc)
        close_arr = ohlc["close"].to_numpy()
        support = result.values["tdst_support"]
        dist = result.values["dist_to_tdst_support"]
        has_level = ~np.isnan(support)
        above = has_level & (close_arr > support)
        if above.any():
            assert (dist[above] > 0).all(), (
                "dist_to_tdst_support should be positive when price > support"
            )

    def test_dist_capped_at_20_percent(self):
        """Distances must be capped at ±0.20."""
        n = 80
        ohlc = self._tds_ohlc(n)
        result = _make_tds().compute(ohlc)
        for key in ("dist_to_tdst_support", "dist_to_tdst_resistance"):
            arr = result.values[key]
            finite = arr[np.isfinite(arr)]
            if len(finite):
                assert (np.abs(finite) <= 0.20 + 1e-9).all(), (
                    f"{key} exceeds ±0.20 cap"
                )

    def test_no_crash_when_no_setup_completes(self):
        """Flat series → no setups complete → support/resistance remain NaN → no crash."""
        ohlc = _flat_ohlc(30)
        result = _make_tds().compute(ohlc)
        assert "dist_to_tdst_support" in result.values
        assert "dist_to_tdst_resistance" in result.values
        # All NaN since no level exists
        assert np.isnan(result.values["dist_to_tdst_support"]).all()
        assert np.isnan(result.values["dist_to_tdst_resistance"]).all()


# ---------------------------------------------------------------------------
# RSI — rsi_centered
# ---------------------------------------------------------------------------

class TestRSIContinuous:
    def test_rsi_centered_equals_rsi_minus_50(self):
        """rsi_centered must equal rsi - 50 for every bar."""
        n = 80
        rng = np.random.default_rng(1)
        closes = 100.0 + np.cumsum(rng.standard_normal(n))
        ohlc = _ohlc_from_closes(closes)
        result = _make_rsi(period=14).compute(ohlc)
        rsi = result.values["rsi"]
        rsi_c = result.values["rsi_centered"]
        np.testing.assert_allclose(rsi_c, rsi - 50.0)

    def test_rsi_still_present(self):
        n = 50
        ohlc = _ohlc_from_closes(np.linspace(100, 110, n))
        result = _make_rsi().compute(ohlc)
        assert "rsi" in result.values
        assert len(result.values["rsi"]) == n

    def test_rsi_centered_length_n(self):
        n = 60
        ohlc = _ohlc_from_closes(np.linspace(100, 110, n))
        result = _make_rsi(period=14).compute(ohlc)
        arr = result.values["rsi_centered"]
        assert isinstance(arr, np.ndarray)
        assert len(arr) == n

    def test_rsi_centered_nan_before_warmup(self):
        """rsi_centered should be NaN for the first `period` bars (matching rsi)."""
        period = 14
        n = 50
        ohlc = _ohlc_from_closes(np.linspace(100, 110, n))
        result = _make_rsi(period=period).compute(ohlc)
        rsi_c = result.values["rsi_centered"]
        # bars 0..period-1 should be NaN
        assert np.isnan(rsi_c[:period]).all()

    def test_rsi_centered_in_range_after_warmup(self):
        """rsi in [0,100] → rsi_centered in [-50, 50]."""
        n = 80
        rng = np.random.default_rng(2)
        closes = 100.0 + np.cumsum(rng.standard_normal(n))
        ohlc = _ohlc_from_closes(closes)
        result = _make_rsi(period=14).compute(ohlc)
        rsi_c = result.values["rsi_centered"]
        valid = rsi_c[np.isfinite(rsi_c)]
        assert (valid >= -50.0 - 1e-9).all()
        assert (valid <= 50.0 + 1e-9).all()
