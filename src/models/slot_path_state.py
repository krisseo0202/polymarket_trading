"""Stateful within-slot BTC path tracking for Family C features.

Holds per-slot running aggregates over BTC ticks since slot_open:
  max, min, time-above-strike, time-below-strike, cross count.

Two entry points:
  - live:     ``state.update(ts, btc, strike)`` each tick,
              ``state.reset(new_slot_ts)`` on slot rollover.
  - backtest: ``SlotPathState.from_ticks(ticks, strike, slot_ts)`` bulk-builds.

``to_features(now_ts, btc_now, strike)`` produces the Family C feature dict,
attributing the trailing interval from the last update to now_ts using the
last known BTC state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Dict, Iterable, Mapping, Optional, Tuple


_SLOT_SECONDS = 300


FAMILY_C_FEATURES = (
    "slot_high_excursion_bps",
    "slot_low_excursion_bps",
    "slot_drift_bps",
    "slot_time_above_strike_pct",
    "slot_strike_crosses",
)


@dataclass
class SlotPathState:
    """Running within-slot path aggregates for one 5-min window."""

    slot_ts: int = 0
    slot_max: float = 0.0
    slot_min: float = inf
    last_sign: int = 0  # sign(btc - strike) at last update; 0 = unseen
    cross_count: int = 0
    time_above: float = 0.0
    time_below: float = 0.0
    last_ts: Optional[float] = None
    last_btc: Optional[float] = None

    def reset(self, new_slot_ts: int) -> None:
        """Reset all running state for a new slot boundary."""
        self.slot_ts = int(new_slot_ts)
        self.slot_max = 0.0
        self.slot_min = inf
        self.last_sign = 0
        self.cross_count = 0
        self.time_above = 0.0
        self.time_below = 0.0
        self.last_ts = None
        self.last_btc = None

    def update(self, ts: float, btc: float, strike: float) -> None:
        """Fold one (ts, btc) tick into the running aggregates.

        Ignores ticks before slot_ts, non-positive prices, or non-positive
        strike. Safe to call with the same timestamp twice — dt==0 produces
        no attribution.
        """
        if btc <= 0 or strike <= 0:
            return
        if ts < self.slot_ts:
            return

        if btc > self.slot_max:
            self.slot_max = btc
        if btc < self.slot_min:
            self.slot_min = btc

        if self.last_ts is not None and self.last_btc is not None:
            dt = ts - self.last_ts
            if dt > 0:
                if self.last_btc > strike:
                    self.time_above += dt
                elif self.last_btc < strike:
                    self.time_below += dt

        new_sign = _sign(btc - strike)
        if self.last_sign != 0 and new_sign != 0 and new_sign != self.last_sign:
            self.cross_count += 1
        self.last_sign = new_sign
        self.last_ts = float(ts)
        self.last_btc = float(btc)

    @classmethod
    def from_ticks(
        cls,
        slot_ts: int,
        strike: float,
        ticks: Iterable[Tuple[float, float]],
    ) -> "SlotPathState":
        """Build state from an iterable of (ts, btc) ticks — backtest helper."""
        state = cls()
        state.reset(slot_ts)
        for ts, btc in ticks:
            state.update(float(ts), float(btc), float(strike))
        return state

    def to_features(
        self,
        now_ts: float,
        btc_now: float,
        strike: float,
    ) -> Dict[str, float]:
        """Produce the Family C feature dict at time ``now_ts``.

        Attributes the trailing interval [last_ts, now_ts] using the last known
        BTC state so short slots don't leave seconds unassigned.
        """
        features: Dict[str, float] = {name: 0.0 for name in FAMILY_C_FEATURES}
        if strike <= 0:
            return features

        time_above = self.time_above
        time_below = self.time_below
        if self.last_ts is not None and self.last_btc is not None and now_ts > self.last_ts:
            dt = now_ts - self.last_ts
            if self.last_btc > strike:
                time_above += dt
            elif self.last_btc < strike:
                time_below += dt

        if self.slot_max > 0:
            features["slot_high_excursion_bps"] = (self.slot_max - strike) / strike * 10_000.0
        if self.slot_min < inf:
            features["slot_low_excursion_bps"] = (self.slot_min - strike) / strike * 10_000.0
        if btc_now > 0:
            features["slot_drift_bps"] = (btc_now - strike) / strike * 10_000.0

        elapsed = now_ts - self.slot_ts
        if elapsed > 0:
            features["slot_time_above_strike_pct"] = max(0.0, min(1.0, time_above / elapsed))

        features["slot_strike_crosses"] = float(self.cross_count)
        return features


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def advance_from_snapshot(
    state: SlotPathState, state_ts: int, snapshot: Mapping[str, object],
) -> int:
    """Fold all ticks in ``snapshot`` that fall within the current slot
    into ``state``, resetting on slot boundary. Returns the new
    ``state_ts`` so the caller can cache it.

    Re-scanning ticks every cycle is idempotent: max/min are monotone and
    the sign-cross logic guards against backwards timestamps.
    """
    slot_expiry_ts = snapshot.get("slot_expiry_ts")
    strike_price = snapshot.get("strike_price")
    btc_prices = list(snapshot.get("btc_prices") or [])
    if slot_expiry_ts is None or strike_price is None or not btc_prices:
        return state_ts

    try:
        slot_ts = int(float(slot_expiry_ts)) - _SLOT_SECONDS
        strike = float(strike_price)
    except (TypeError, ValueError):
        return state_ts
    if strike <= 0:
        return state_ts

    if slot_ts != state_ts:
        state.reset(slot_ts)
        state_ts = slot_ts

    last_ts = state.last_ts or float(slot_ts)
    for entry in btc_prices:
        try:
            ts = float(entry[0])
            price = float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if ts <= last_ts or ts < slot_ts:
            continue
        state.update(ts, price, strike)
    return state_ts


def features_from_snapshot(
    state: SlotPathState, snapshot: Mapping[str, object],
) -> Dict[str, float]:
    """Compute Family C features from ``state`` at the snapshot's ``now_ts``."""
    slot_expiry_ts = snapshot.get("slot_expiry_ts")
    strike_price = snapshot.get("strike_price")
    now_ts = snapshot.get("now_ts")
    btc_prices = list(snapshot.get("btc_prices") or [])
    if not btc_prices or slot_expiry_ts is None or strike_price is None:
        return {}
    try:
        now_ts_f = float(now_ts) if now_ts is not None else float(btc_prices[-1][0])
        btc_now = float(btc_prices[-1][1])
        strike = float(strike_price)
    except (TypeError, ValueError, IndexError):
        return {}
    return state.to_features(now_ts_f, btc_now, strike)


# Default values used when no slot state is supplied to the feature builder.
DEFAULT_SLOT_PATH_FEATURES: Dict[str, float] = {name: 0.0 for name in FAMILY_C_FEATURES}
