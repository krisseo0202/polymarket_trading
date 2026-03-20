"""Tests for simulate_up_prob and estimate_realized_vol."""

import sys
import os
import pytest

# quant_desk.py is at repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from quant_desk import simulate_up_prob
from src.utils.volatility import estimate_realized_vol


class TestSimulateUpProb:
    def test_at_the_money_near_fifty_percent(self):
        # At-the-money, 60s remaining → should be ~50%
        # Use high n_paths to reduce Monte Carlo variance
        result = simulate_up_prob(
            start_price=50000.0,
            current_price=50000.0,
            time_left_sec=60.0,
            vol=0.5,
            n_paths=50_000,
        )
        assert 0.45 <= result <= 0.55

    def test_deep_in_the_money_near_one(self):
        # Price well above start, almost no time left → probability near 1
        result = simulate_up_prob(
            start_price=100.0,
            current_price=110.0,
            time_left_sec=1.0,
            vol=0.001,
            n_paths=10_000,
        )
        assert result > 0.95

    def test_deep_out_of_the_money_near_zero(self):
        # Price well below start, almost no time left → probability near 0
        result = simulate_up_prob(
            start_price=100.0,
            current_price=90.0,
            time_left_sec=1.0,
            vol=0.001,
            n_paths=10_000,
        )
        assert result < 0.05

    def test_near_zero_time_does_not_raise(self):
        result = simulate_up_prob(
            start_price=50000.0,
            current_price=50000.0,
            time_left_sec=0.001,
            vol=0.5,
            n_paths=1_000,
        )
        assert 0.0 <= result <= 1.0

    def test_returns_python_float_not_numpy(self):
        result = simulate_up_prob(
            start_price=50000.0,
            current_price=50000.0,
            time_left_sec=60.0,
            vol=0.5,
            n_paths=100,
        )
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_result_always_in_unit_interval(self):
        for current in [40000.0, 50000.0, 60000.0]:
            result = simulate_up_prob(
                start_price=50000.0,
                current_price=current,
                time_left_sec=30.0,
                vol=0.8,
                n_paths=500,
            )
            assert 0.0 <= result <= 1.0


class TestEstimateRealizedVol:
    def test_flat_prices_returns_zero(self):
        prices = [100.0] * 60
        vol = estimate_realized_vol(prices, window_sec=60)
        assert vol == 0.0

    def test_increasing_prices_returns_positive_vol(self):
        prices = [100.0 + i * 0.5 for i in range(60)]
        vol = estimate_realized_vol(prices, window_sec=60)
        assert vol > 0.0

    def test_volatile_prices_higher_vol_than_stable(self):
        stable = [100.0 + (i % 2) * 0.1 for i in range(60)]
        volatile = [100.0 + (i % 2) * 5.0 for i in range(60)]
        vol_stable = estimate_realized_vol(stable, window_sec=60)
        vol_volatile = estimate_realized_vol(volatile, window_sec=60)
        assert vol_volatile > vol_stable

    def test_fewer_than_two_prices_returns_zero(self):
        assert estimate_realized_vol([], window_sec=60) == 0.0
        assert estimate_realized_vol([100.0], window_sec=60) == 0.0
