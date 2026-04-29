"""Tests for the EdgeStabilityTracker used by logreg_edge and prob_edge."""

import pytest

from src.strategies._edge_stability import EdgeStabilityTracker


def test_rejects_zero_or_negative_n_ticks():
    for bad in (0, -1, -100):
        with pytest.raises(ValueError):
            EdgeStabilityTracker(n_ticks=bad)


def test_n_ticks_1_fires_on_first_above_tick():
    """n_ticks=1 must reproduce the pre-stability behavior exactly."""
    t = EdgeStabilityTracker(n_ticks=1)
    d = t.update(yes_above=True, no_above=False, slot_ts=1000)
    assert d.yes_ready is True
    assert d.no_ready is False
    assert d.yes_count == 1
    assert d.no_count == 0


def test_n_ticks_3_requires_three_consecutive_hits():
    t = EdgeStabilityTracker(n_ticks=3)
    for i in (1, 2):
        d = t.update(yes_above=True, no_above=False, slot_ts=1000)
        assert d.yes_ready is False
        assert d.yes_count == i
    d = t.update(yes_above=True, no_above=False, slot_ts=1000)
    assert d.yes_ready is True
    assert d.yes_count == 3


def test_failing_tick_resets_that_sides_counter():
    t = EdgeStabilityTracker(n_ticks=3)
    t.update(yes_above=True, no_above=False, slot_ts=1000)
    t.update(yes_above=True, no_above=False, slot_ts=1000)
    # Fail on the 3rd tick — counter must zero.
    d = t.update(yes_above=False, no_above=False, slot_ts=1000)
    assert d.yes_ready is False
    assert d.yes_count == 0
    # Start over: still needs 3 more.
    t.update(yes_above=True, no_above=False, slot_ts=1000)
    t.update(yes_above=True, no_above=False, slot_ts=1000)
    d = t.update(yes_above=True, no_above=False, slot_ts=1000)
    assert d.yes_ready is True


def test_sides_track_independently():
    t = EdgeStabilityTracker(n_ticks=2)
    # YES up, NO down — only YES should accumulate.
    t.update(yes_above=True, no_above=False, slot_ts=1000)
    d = t.update(yes_above=True, no_above=False, slot_ts=1000)
    assert d.yes_ready is True and d.no_ready is False
    # Now flip: YES drops, NO rises — YES resets, NO builds.
    t.update(yes_above=False, no_above=True, slot_ts=1000)
    d = t.update(yes_above=False, no_above=True, slot_ts=1000)
    assert d.yes_ready is False and d.no_ready is True
    assert d.yes_count == 0 and d.no_count == 2


def test_slot_rollover_resets_both_counters():
    t = EdgeStabilityTracker(n_ticks=3)
    t.update(yes_above=True, no_above=True, slot_ts=1000)
    t.update(yes_above=True, no_above=True, slot_ts=1000)
    assert t.yes_count == 2 and t.no_count == 2
    # New slot arrives — counters must zero before this tick is applied.
    d = t.update(yes_above=True, no_above=True, slot_ts=1300)
    assert d.yes_count == 1 and d.no_count == 1
    assert d.yes_ready is False and d.no_ready is False


def test_none_slot_ts_skips_slot_tracking():
    """Callers that don't track slots can pass None — no reset happens."""
    t = EdgeStabilityTracker(n_ticks=2)
    t.update(yes_above=True, no_above=False, slot_ts=None)
    d = t.update(yes_above=True, no_above=False, slot_ts=None)
    assert d.yes_ready is True


def test_manual_reset_zeros_everything():
    t = EdgeStabilityTracker(n_ticks=3)
    t.update(yes_above=True, no_above=True, slot_ts=1000)
    t.update(yes_above=True, no_above=True, slot_ts=1000)
    t.reset()
    assert t.yes_count == 0 and t.no_count == 0
    # After reset, next slot_ts doesn't trigger a second reset.
    d = t.update(yes_above=True, no_above=True, slot_ts=1000)
    assert d.yes_count == 1


def test_counter_saturates_at_ready_does_not_overflow():
    """Past n_ticks the gate stays ready and the count keeps rising — used
    so in-slot re-entry doesn't need to re-warm after a successful signal."""
    t = EdgeStabilityTracker(n_ticks=2)
    t.update(yes_above=True, no_above=False, slot_ts=1000)
    d = t.update(yes_above=True, no_above=False, slot_ts=1000)
    assert d.yes_ready is True and d.yes_count == 2
    d = t.update(yes_above=True, no_above=False, slot_ts=1000)
    assert d.yes_ready is True and d.yes_count == 3
