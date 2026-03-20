"""Tests for estimate_realized_vol()."""

import math
import pytest

from src.utils.volatility import estimate_realized_vol


class TestStdMethod:
    def test_std_known_values(self):
        # prices: 100, 102, 101, 103
        # returns: 0.02, -0.009803..., 0.019801...
        prices = [100.0, 102.0, 101.0, 103.0]
        vol = estimate_realized_vol(prices, window_sec=10, method="std")

        returns = [
            (102 - 100) / 100,
            (101 - 102) / 102,
            (103 - 101) / 101,
        ]
        mean = sum(returns) / len(returns)
        expected = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
        assert abs(vol - expected) < 1e-12

    def test_flat_prices_returns_zero(self):
        prices = [50.0] * 10
        assert estimate_realized_vol(prices, window_sec=20, method="std") == 0.0


class TestEmaMethod:
    def test_ema_known_values(self):
        prices = [100.0, 102.0, 101.0, 103.0]
        vol = estimate_realized_vol(prices, window_sec=10, method="ema")
        assert vol > 0.0

    def test_ema_differs_from_std(self):
        prices = [100.0, 102.0, 99.0, 104.0, 101.0]
        std_vol = estimate_realized_vol(prices, window_sec=10, method="std")
        ema_vol = estimate_realized_vol(prices, window_sec=10, method="ema")
        assert std_vol != ema_vol


class TestEdgeCases:
    def test_insufficient_data_single_price(self):
        assert estimate_realized_vol([100.0], window_sec=10) == 0.0

    def test_insufficient_data_empty(self):
        assert estimate_realized_vol([], window_sec=10) == 0.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            estimate_realized_vol([100.0, 101.0], window_sec=10, method="bad")

    def test_zero_price_skipped(self):
        # Zero price in denominator should not crash
        prices = [0.0, 0.0, 100.0, 101.0]
        vol = estimate_realized_vol(prices, window_sec=10, method="std")
        # Only one valid return (100->101), std needs >=2 returns
        assert vol == 0.0


class TestWindowBehavior:
    def test_window_shorter_than_history(self):
        prices = [100.0, 200.0, 102.0, 101.0, 103.0]
        # window_sec=3, interval=1 -> last 3 prices: [101, 103] returns from [102,101,103]
        vol_short = estimate_realized_vol(prices, window_sec=3, method="std")
        vol_full = estimate_realized_vol(prices, window_sec=10, method="std")
        # The huge 100->200 jump is excluded from the short window
        assert vol_short < vol_full

    def test_window_larger_than_history(self):
        prices = [100.0, 102.0, 101.0]
        # Should use all prices without error
        vol = estimate_realized_vol(prices, window_sec=1000, method="std")
        assert vol > 0.0

    def test_sample_interval_affects_window(self):
        prices = [100.0, 200.0, 102.0, 101.0, 103.0]
        # window=3s, interval=1s -> 3 prices
        vol_1s = estimate_realized_vol(prices, window_sec=3, sample_interval_sec=1.0)
        # window=3s, interval=0.5s -> 6 prices -> clamped to all 5
        vol_half = estimate_realized_vol(prices, window_sec=3, sample_interval_sec=0.5)
        # With interval=0.5, window covers more prices including the 100->200 jump
        assert vol_half > vol_1s
