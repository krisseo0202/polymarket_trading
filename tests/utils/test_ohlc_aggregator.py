"""Tests for the OHLC aggregator used by multi-TF feature computation."""

import numpy as np
import pandas as pd
import pytest

from src.utils.ohlc_aggregator import aggregate_ohlc_from_1s_ohlc, aggregate_to_ohlc


def test_aggregate_single_bar_from_ticks():
    # Four ticks inside a single 60s bucket.
    ticks = [
        (1776549600, 100.0),  # bar_start
        (1776549615, 110.0),
        (1776549630, 95.0),
        (1776549650, 105.0),
    ]
    out = aggregate_to_ohlc(ticks, interval_s=60)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 110.0
    assert row["low"] == 95.0
    assert row["close"] == 105.0


def test_aggregate_respects_bar_edges():
    # Two bars, split on the 60s boundary.
    ticks = [
        (1776549600, 100.0),  # bar 0
        (1776549655, 110.0),
        (1776549660, 120.0),  # bar 1
        (1776549715, 115.0),
    ]
    out = aggregate_to_ohlc(ticks, interval_s=60)
    assert len(out) == 2
    assert out.iloc[0]["close"] == 110.0
    assert out.iloc[1]["open"] == 120.0
    assert out.iloc[1]["close"] == 115.0


def test_aggregate_drops_empty_buckets_when_gapped():
    # Gap from bar 0 to bar 3 — middle bars should be absent (not filled).
    ticks = [
        (1776549600, 100.0),
        (1776549780, 120.0),  # 3 bars later
    ]
    out = aggregate_to_ohlc(ticks, interval_s=60)
    assert len(out) == 2


def test_aggregate_handles_out_of_order_input():
    ticks = [
        (1776549650, 105.0),
        (1776549600, 100.0),
        (1776549630, 95.0),
        (1776549615, 110.0),
    ]
    out = aggregate_to_ohlc(ticks, interval_s=60)
    row = out.iloc[0]
    # Open must be the earliest-timestamp tick even though it came last.
    assert row["open"] == 100.0
    assert row["close"] == 105.0
    assert row["high"] == 110.0
    assert row["low"] == 95.0


def test_aggregate_sums_tick_volume_when_present():
    ticks = [
        (1776549600, 100.0, 2.0),
        (1776549615, 110.0, 3.0),
        (1776549630, 95.0, 1.5),
    ]
    out = aggregate_to_ohlc(ticks, interval_s=60)
    assert out.iloc[0]["volume"] == 6.5


def test_aggregate_fills_unit_volume_when_absent():
    ticks = [(1776549600, 100.0), (1776549615, 110.0)]
    out = aggregate_to_ohlc(ticks, interval_s=60)
    # Without tick volume, we count ticks (2 here).
    assert out.iloc[0]["volume"] == 2.0


def test_aggregate_empty_input_returns_empty_frame():
    out = aggregate_to_ohlc([], interval_s=60)
    assert len(out) == 0
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_aggregate_rejects_zero_interval():
    with pytest.raises(ValueError):
        aggregate_to_ohlc([(1776549600, 100.0)], interval_s=0)


def test_aggregate_4h_bar_from_many_ticks():
    # 4h = 14400s. Generate ticks from 00:00 to 03:59 UTC — all in one 4h bar.
    ts_base = 1776556800  # 2026-04-19T00:00:00Z
    rng = np.random.default_rng(1)
    ticks = [(ts_base + i, 100.0 + float(rng.normal(0, 0.5))) for i in range(0, 4 * 3600, 10)]
    out = aggregate_to_ohlc(ticks, interval_s=4 * 3600)
    assert len(out) == 1
    # Second 4h bar starts at ts_base + 14400 — we should NOT produce one.
    assert out.index[0].timestamp() == ts_base


def test_aggregate_from_1s_ohlc_matches_tick_aggregator():
    # Given 1s OHLCV bars, aggregating to 5m should give the same candle as
    # computing directly from the underlying ticks (using close for both paths).
    ts = np.arange(1776549600, 1776549600 + 300, 1)  # 5 minutes
    rng = np.random.default_rng(2)
    closes = 100.0 + rng.normal(0, 0.3, size=len(ts)).cumsum()
    df_1s = pd.DataFrame({
        "timestamp": ts, "open": closes, "high": closes,
        "low": closes, "close": closes, "volume": np.ones_like(ts),
    })
    agg = aggregate_ohlc_from_1s_ohlc(df_1s, interval_s=300)
    tick_agg = aggregate_to_ohlc(list(zip(ts, closes)), interval_s=300)
    assert len(agg) == len(tick_agg) == 1
    assert agg.iloc[0]["open"] == pytest.approx(tick_agg.iloc[0]["open"])
    assert agg.iloc[0]["close"] == pytest.approx(tick_agg.iloc[0]["close"])
    assert agg.iloc[0]["high"] == pytest.approx(tick_agg.iloc[0]["high"])
    assert agg.iloc[0]["low"] == pytest.approx(tick_agg.iloc[0]["low"])
