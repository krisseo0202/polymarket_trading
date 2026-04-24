"""Tests for SlotPathState — Family C within-slot path aggregates."""

import math

import pytest

from src.models.slot_path_state import SlotPathState


SLOT_TS = 1_000_000
STRIKE = 100.0


def test_initial_state_produces_default_features():
    state = SlotPathState()
    state.reset(SLOT_TS)

    feats = state.to_features(now_ts=SLOT_TS + 5, btc_now=100.0, strike=STRIKE)

    assert feats["slot_high_excursion_bps"] == 0.0
    assert feats["slot_low_excursion_bps"] == 0.0
    # btc_now==strike → drift 0
    assert feats["slot_drift_bps"] == 0.0
    assert feats["slot_time_above_strike_pct"] == 0.0
    assert feats["slot_strike_crosses"] == 0.0


def test_update_tracks_max_min_and_drift_bps():
    state = SlotPathState()
    state.reset(SLOT_TS)
    state.update(SLOT_TS + 1, 101.0, STRIKE)
    state.update(SLOT_TS + 2, 99.0, STRIKE)
    state.update(SLOT_TS + 3, 100.5, STRIKE)

    feats = state.to_features(SLOT_TS + 3, btc_now=100.5, strike=STRIKE)

    # max excursion: 101 vs 100 → +100bps
    assert feats["slot_high_excursion_bps"] == pytest.approx(100.0)
    # min excursion: 99 vs 100 → -100bps
    assert feats["slot_low_excursion_bps"] == pytest.approx(-100.0)
    # drift at now: 100.5 vs 100 → +50bps
    assert feats["slot_drift_bps"] == pytest.approx(50.0)


def test_cross_count_counts_sign_flips_only():
    state = SlotPathState()
    state.reset(SLOT_TS)
    state.update(SLOT_TS + 1, 101.0, STRIKE)  # sign=+1 (first tick, no cross)
    state.update(SLOT_TS + 2, 102.0, STRIKE)  # sign=+1 (no cross)
    state.update(SLOT_TS + 3, 99.0, STRIKE)   # sign=-1 → cross 1
    state.update(SLOT_TS + 4, 99.5, STRIKE)   # sign=-1 (no cross)
    state.update(SLOT_TS + 5, 100.5, STRIKE)  # sign=+1 → cross 2

    feats = state.to_features(SLOT_TS + 5, btc_now=100.5, strike=STRIKE)

    assert feats["slot_strike_crosses"] == 2.0


def test_time_above_uses_last_known_state_for_intervals():
    state = SlotPathState()
    state.reset(SLOT_TS)
    # tick at t+0 sets last_btc=101 (above); tick at t+10 keeps above;
    # tick at t+20 sets last_btc=99 (below). Interval [t+0, t+10] attributed to
    # above using last_btc=101; interval [t+10, t+20] attributed to above
    # using last_btc=101 at that time → both 10s above.
    state.update(SLOT_TS + 0, 101.0, STRIKE)
    state.update(SLOT_TS + 10, 101.0, STRIKE)
    state.update(SLOT_TS + 20, 99.0, STRIKE)

    feats = state.to_features(SLOT_TS + 30, btc_now=99.0, strike=STRIKE)

    # 20s above + 10s below (trailing [t+20, t+30] attributed below).
    # Elapsed = 30s. time_above_pct = 20/30.
    assert feats["slot_time_above_strike_pct"] == pytest.approx(20.0 / 30.0)


def test_from_ticks_equivalent_to_incremental_updates():
    ticks = [(SLOT_TS + i, 100.0 + (i % 3 - 1) * 0.5) for i in range(20)]
    streamed = SlotPathState()
    streamed.reset(SLOT_TS)
    for ts, btc in ticks:
        streamed.update(ts, btc, STRIKE)
    bulk = SlotPathState.from_ticks(SLOT_TS, STRIKE, ticks)

    a = streamed.to_features(SLOT_TS + 20, btc_now=ticks[-1][1], strike=STRIKE)
    b = bulk.to_features(SLOT_TS + 20, btc_now=ticks[-1][1], strike=STRIKE)
    assert a == b


def test_reset_clears_all_running_state():
    state = SlotPathState()
    state.reset(SLOT_TS)
    state.update(SLOT_TS + 1, 110.0, STRIKE)
    state.update(SLOT_TS + 2, 90.0, STRIKE)

    state.reset(SLOT_TS + 300)

    assert state.slot_max == 0.0
    assert state.slot_min == math.inf
    assert state.cross_count == 0
    assert state.time_above == 0.0
    assert state.time_below == 0.0


def test_update_ignores_pre_slot_ticks_and_bad_inputs():
    state = SlotPathState()
    state.reset(SLOT_TS)
    state.update(SLOT_TS - 5, 110.0, STRIKE)   # pre-slot: ignored
    state.update(SLOT_TS + 1, 0.0, STRIKE)     # zero price: ignored
    state.update(SLOT_TS + 2, 101.0, 0.0)      # zero strike: ignored
    state.update(SLOT_TS + 3, 101.0, STRIKE)   # accepted

    feats = state.to_features(SLOT_TS + 3, btc_now=101.0, strike=STRIKE)
    assert feats["slot_high_excursion_bps"] == pytest.approx(100.0)
    # Only one accepted tick → no crosses possible.
    assert feats["slot_strike_crosses"] == 0.0


def test_to_features_clamps_time_above_pct_to_unit_interval():
    state = SlotPathState()
    state.reset(SLOT_TS)
    state.update(SLOT_TS + 1, 101.0, STRIKE)
    # Query at a time AFTER the slot window closes — time_above could exceed
    # elapsed if the trailing attribution was buggy. Clamp proves the guard.
    feats = state.to_features(SLOT_TS + 1_000, btc_now=101.0, strike=STRIKE)
    assert 0.0 <= feats["slot_time_above_strike_pct"] <= 1.0
