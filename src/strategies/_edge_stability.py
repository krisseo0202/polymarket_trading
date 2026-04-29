"""Edge-stability gate — block trades that rest on a single-tick edge blip.

Entry decisions in ``logreg_edge`` and ``prob_edge`` compare the model's
probability against the current orderbook. A single orderbook flicker or a
momentary feature miscomputation can push either side briefly over the
threshold and trigger an entry that reverts on the next tick.

This module wraps a small per-side counter that only signals "ready" after
an edge has stayed above threshold for ``n_ticks`` consecutive analyze()
calls within the same slot. Counters reset automatically on slot rollover
(when ``slot_ts`` changes) and on any failing tick.

Setting ``n_ticks=1`` reduces to the pre-stability behavior (fire on any
above-threshold tick), so this is opt-in and never regresses existing runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StabilityDecision:
    """Result of an ``EdgeStabilityTracker.update()`` call.

    ``yes_ready`` / ``no_ready`` are the gate flags callers use to allow a
    signal. ``yes_count`` / ``no_count`` expose the current run length for
    dashboards and skip-reason strings.
    """

    yes_ready: bool
    no_ready: bool
    yes_count: int
    no_count: int


class EdgeStabilityTracker:
    """Per-side consecutive-tick counter with slot-scoped reset.

    Each call to ``update()`` bumps the counter on a side whose edge is
    above threshold this tick, and zeros it on any side that falls below.
    A side is ``ready`` when its counter has reached ``n_ticks``.

    Rationale for slot-scoped reset: edges from different 5-minute slots
    aren't comparable (different strike, different BTC spot, different
    expiry). A counter that survived the rollover would fire on stale
    signal from the prior slot.
    """

    def __init__(self, n_ticks: int = 1):
        if n_ticks < 1:
            raise ValueError(f"n_ticks must be >= 1, got {n_ticks}")
        self.n_ticks = int(n_ticks)
        self._yes_count = 0
        self._no_count = 0
        self._slot_ts: Optional[float] = None

    def update(
        self,
        yes_above: bool,
        no_above: bool,
        slot_ts: Optional[float],
    ) -> StabilityDecision:
        """Advance both counters and return the current gate state.

        Args:
            yes_above: whether the YES-side edge cleared its threshold this tick.
            no_above: same for the NO side.
            slot_ts: slot identifier (e.g. ``slot_expiry_ts``). When it
                changes vs. the previously-seen value, counters reset before
                the new tick is applied. Pass ``None`` to skip slot tracking.
        """
        if slot_ts is not None and slot_ts != self._slot_ts:
            self._yes_count = 0
            self._no_count = 0
            self._slot_ts = slot_ts

        self._yes_count = self._yes_count + 1 if yes_above else 0
        self._no_count = self._no_count + 1 if no_above else 0

        return StabilityDecision(
            yes_ready=self._yes_count >= self.n_ticks,
            no_ready=self._no_count >= self.n_ticks,
            yes_count=self._yes_count,
            no_count=self._no_count,
        )

    @property
    def yes_count(self) -> int:
        return self._yes_count

    @property
    def no_count(self) -> int:
        return self._no_count

    def reset(self) -> None:
        """Force-zero both counters and clear slot tracking."""
        self._yes_count = 0
        self._no_count = 0
        self._slot_ts = None
