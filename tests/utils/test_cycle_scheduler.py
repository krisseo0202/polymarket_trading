"""Tests for cycle_scheduler: detect_current_cycle, cycle_index,
aligned_cycle_anchor, and run_last_second_strategy."""

import threading
import time

import pytest

from src.utils.cycle_scheduler import (
    aligned_cycle_anchor,
    cycle_index,
    detect_current_cycle,
    run_last_second_strategy,
)


class TestDetectCurrentCycle:
    def test_seconds_remaining_mid_cycle(self):
        # 3s elapsed in a 10s cycle → 7s remaining
        anchor = time.time() - 3.0
        remaining = detect_current_cycle(anchor, cycle_len=10)
        assert 6.5 <= remaining <= 7.5  # allow ±0.5s for test execution time

    def test_seconds_remaining_near_end(self):
        anchor = time.time() - 9.0
        remaining = detect_current_cycle(anchor, cycle_len=10)
        assert 0.5 <= remaining <= 1.5

    def test_returns_cycle_len_at_boundary(self):
        # Exactly on a boundary: elapsed = 0
        anchor = time.time()
        remaining = detect_current_cycle(anchor, cycle_len=10)
        assert remaining > 9.0  # near full cycle

    def test_second_cycle_wraps_correctly(self):
        # 12s elapsed in 10s cycles → 2s into second cycle → 8s remaining
        anchor = time.time() - 12.0
        remaining = detect_current_cycle(anchor, cycle_len=10)
        assert 7.5 <= remaining <= 8.5

    def test_raises_on_invalid_cycle_len(self):
        with pytest.raises(ValueError):
            detect_current_cycle(time.time(), cycle_len=0)

        with pytest.raises(ValueError):
            detect_current_cycle(time.time(), cycle_len=-5)

    def test_future_anchor_treated_as_zero_elapsed(self):
        # anchor slightly in the future → remaining should be full cycle
        anchor = time.time() + 1.0
        remaining = detect_current_cycle(anchor, cycle_len=10)
        assert remaining > 9.0


class TestCycleIndex:
    def test_zero_at_start(self):
        anchor = time.time()
        assert cycle_index(anchor, cycle_len=10) == 0

    def test_increments_after_one_cycle(self):
        anchor = time.time() - 11.0  # 11s elapsed, 10s cycle → idx=1
        assert cycle_index(anchor, cycle_len=10) == 1

    def test_increments_after_two_cycles(self):
        anchor = time.time() - 21.0
        assert cycle_index(anchor, cycle_len=10) == 2


class TestAlignedCycleAnchor:
    def test_returns_timestamp_divisible_by_300(self):
        anchor = aligned_cycle_anchor(300)
        assert anchor % 300 == pytest.approx(0.0, abs=1e-3)

    def test_anchor_in_past(self):
        anchor = aligned_cycle_anchor(300)
        assert anchor <= time.time()

    def test_anchor_within_one_cycle(self):
        anchor = aligned_cycle_anchor(300)
        assert time.time() - anchor <= 300


class TestRunLastSecondStrategy:
    def test_fires_once_per_cycle(self):
        """Callback fires exactly once per cycle in a short test cycle."""
        counter = [0]
        stop = threading.Event()

        def callback():
            counter[0] += 1

        anchor = time.time()
        t = threading.Thread(
            target=run_last_second_strategy,
            args=(anchor, callback),
            kwargs=dict(
                cycle_len=1,
                trigger_window_s=0.3,
                poll_interval_s=0.05,
                stop_event=stop,
            ),
            daemon=True,
        )
        t.start()
        time.sleep(2.2)
        stop.set()
        t.join(timeout=1.0)

        # Should have fired exactly twice (once per 1s cycle over ~2.2s)
        assert counter[0] == 2, f"Expected 2 firings, got {counter[0]}"

    def test_callback_exception_does_not_crash_loop(self):
        """An exception in the callback must not stop the scheduler."""
        counter = [0]
        stop = threading.Event()

        def flaky_callback():
            counter[0] += 1
            if counter[0] == 1:
                raise RuntimeError("intentional error")

        anchor = time.time()
        t = threading.Thread(
            target=run_last_second_strategy,
            args=(anchor, flaky_callback),
            kwargs=dict(
                cycle_len=1,
                trigger_window_s=0.3,
                poll_interval_s=0.05,
                stop_event=stop,
            ),
            daemon=True,
        )
        t.start()
        time.sleep(2.2)
        stop.set()
        t.join(timeout=1.0)

        # Both cycles must have fired despite the exception on cycle 1
        assert counter[0] == 2, f"Expected 2 firings, got {counter[0]}"

    def test_stop_event_exits_quickly(self):
        """Setting stop_event terminates the loop promptly."""
        stop = threading.Event()

        anchor = time.time()
        t = threading.Thread(
            target=run_last_second_strategy,
            args=(anchor, lambda: None),
            kwargs=dict(
                cycle_len=60,          # long cycle so trigger never fires
                trigger_window_s=5.0,
                poll_interval_s=0.05,
                stop_event=stop,
            ),
            daemon=True,
        )
        t.start()
        time.sleep(0.1)
        stop.set()
        t.join(timeout=0.5)
        assert not t.is_alive(), "Scheduler thread should have exited"
