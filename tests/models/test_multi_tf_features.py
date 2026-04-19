"""Tests for the multi-timeframe feature computer."""

import numpy as np
import pytest

from src.models.multi_tf_features import (
    MIN_BARS_PER_TF,
    TIMEFRAMES,
    compute_multi_tf_features,
    multi_tf_feature_names,
)


def _synth_ticks(n_hours: int, seed: int = 0):
    """Generate 1s ticks across `n_hours` hours with a gentle random walk."""
    rng = np.random.default_rng(seed)
    ts_base = 1776556800  # 2026-04-19T00:00:00Z
    n = n_hours * 3600
    ts = np.arange(ts_base, ts_base + n, 1, dtype=np.int64)
    price = 100_000.0 + rng.normal(0, 5.0, size=n).cumsum()
    return list(zip(ts.tolist(), price.tolist())), ts_base, n


def test_feature_names_has_91_unique_columns():
    names = multi_tf_feature_names()
    assert len(names) == 91
    assert len(set(names)) == 91


def test_feature_names_follow_convention():
    names = set(multi_tf_feature_names())
    # Spot-check a handful from each family across the TF range.
    for col in ("rsi_1m", "rsi_240m", "ut_5m_trend", "ut_60m_distance_pct",
                "ut_30m_buy_signal", "td_15m_bull_setup", "td_240m_buy_13",
                "td_3m_sell_9"):
        assert col in names, f"missing {col}"


def test_all_tfs_ready_with_three_days_of_ticks():
    # 3 days × 24 hours = 72 hours. 4h TF gets 18 bars (< MIN_BARS_PER_TF=25),
    # so 4h should still be insufficient. Use 5 days to clear the 4h bar.
    ticks, ts_base, n = _synth_ticks(n_hours=24 * 5, seed=1)
    now_ts = ts_base + n - 1
    features, status = compute_multi_tf_features(ticks, now_ts=now_ts)

    assert status == {tf: "ready" for _, tf in TIMEFRAMES}
    # All 91 features should be populated (either non-zero or legitimately 0/1).
    assert set(features.keys()) == set(multi_tf_feature_names())


def test_insufficient_history_leaves_defaults_and_flags_status():
    # Only 10 minutes of ticks — 1m TF gets 10 bars (< MIN), all others fewer.
    ticks, ts_base, _ = _synth_ticks(n_hours=0, seed=2)
    ticks = [(ts_base + i, 100_000.0 + i * 0.1) for i in range(600)]
    features, status = compute_multi_tf_features(ticks, now_ts=ts_base + 599)

    for _, tf in TIMEFRAMES:
        assert status[tf].startswith("insufficient_bars_"), tf
    # All features should stay at 0.0.
    assert all(v == 0.0 for v in features.values())


def test_empty_ticks_returns_default_features_and_no_history_status():
    features, status = compute_multi_tf_features([], now_ts=1776556800)
    assert all(v == 0.0 for v in features.values())
    assert status == {tf: "no_history" for _, tf in TIMEFRAMES}


def test_rsi_above_50_on_strictly_rising_series():
    # 10 hours of strictly rising prices → RSI should be 100 once warmed.
    ts_base = 1776556800
    ticks = [(ts_base + i, 100_000.0 + i * 0.5) for i in range(10 * 3600)]
    features, status = compute_multi_tf_features(ticks, now_ts=ts_base + 10 * 3600 - 1)
    assert status["1m"] == "ready"
    assert status["5m"] == "ready"
    # Strictly up → RSI == 100.
    assert features["rsi_1m"] == 100.0
    assert features["rsi_5m"] == 100.0


def test_ut_bot_trend_is_1_on_strictly_rising_series():
    ts_base = 1776556800
    ticks = [(ts_base + i, 100_000.0 + i * 0.5) for i in range(10 * 3600)]
    features, _ = compute_multi_tf_features(ticks, now_ts=ts_base + 10 * 3600 - 1)
    assert features["ut_1m_trend"] == 1.0
    assert features["ut_5m_trend"] == 1.0


def test_future_bars_are_trimmed_when_now_ts_in_middle():
    # 3 days of ticks, but ask for features at t = 1 day in.
    ticks, ts_base, _ = _synth_ticks(n_hours=72, seed=3)
    mid_ts = ts_base + 24 * 3600
    features, status = compute_multi_tf_features(ticks, now_ts=mid_ts)
    # 1-minute TF at 1 day in: 1440 bars, plenty. 240m TF: 6 bars, insufficient.
    assert status["1m"] == "ready"
    assert status["240m"].startswith("insufficient_bars_")


def test_features_contain_only_finite_floats():
    ticks, ts_base, n = _synth_ticks(n_hours=24 * 5, seed=4)
    features, _ = compute_multi_tf_features(ticks, now_ts=ts_base + n - 1)
    for name, val in features.items():
        assert isinstance(val, float), f"{name} is {type(val)}"
        assert not np.isnan(val), f"{name} is NaN"
        assert np.isfinite(val), f"{name} is not finite"
