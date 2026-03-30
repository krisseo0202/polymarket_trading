"""Single source of truth for the current 5-minute Polymarket slot."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

SLOT_INTERVAL_S: int = 300


@dataclass
class SlotContext:
    """Immutable snapshot of one 5-minute slot's resolved state."""

    slot_start_ts: int            # floor(unix / 300) * 300
    slot_end_ts: int              # slot_start_ts + 300
    strike_price: Optional[float]     # Chainlink slot-open, or regex fallback
    strike_source: str            # "chainlink" | "regex" | "unknown"
    btc_ref_price: Optional[float]    # latest Chainlink tick price
    captured_at: float            # time.time() when this snapshot was built

    def seconds_remaining(self, now: Optional[float] = None) -> float:
        t = now if now is not None else time.time()
        return max(0.0, self.slot_end_ts - t)

    def is_same_slot(self, ts: float) -> bool:
        """True when ts falls within [slot_start_ts, slot_end_ts)."""
        return self.slot_start_ts <= ts < self.slot_end_ts

    @staticmethod
    def slot_for(ts: float) -> int:
        """Return the slot-start timestamp for a given unix timestamp."""
        return int(math.floor(ts / SLOT_INTERVAL_S) * SLOT_INTERVAL_S)


class SlotStateManager:
    """
    Single source of truth for the current 5-minute slot.

    Thread-safe.  Updated by bot.py each cycle and optionally between
    cycles via update_from_chainlink().  The clock_fn parameter makes
    the class trivially testable with a fixed time substitute.
    """

    def __init__(self, clock_fn=time.time):
        self._clock = clock_fn
        self._lock = threading.Lock()
        self._ctx: Optional[SlotContext] = None

    # ── Read ─────────────────────────────────────────────────────────────────

    def get(self) -> Optional[SlotContext]:
        """Return the most recently built SlotContext, or None before first update."""
        with self._lock:
            return self._ctx

    def current_slot_ts(self) -> int:
        """Return floor(now/300)*300 using the injected clock."""
        return SlotContext.slot_for(self._clock())

    def seconds_remaining(self) -> float:
        """Seconds left in the current slot per the injected clock."""
        now = self._clock()
        return max(0.0, self.current_slot_ts() + SLOT_INTERVAL_S - now)

    # ── Write ─────────────────────────────────────────────────────────────────

    def update(
        self,
        strike_price: Optional[float],
        strike_source: str,
        btc_ref_price: Optional[float],
        now: Optional[float] = None,
    ) -> SlotContext:
        """
        Build and store a new SlotContext for the current clock moment.
        Returns the new context.
        """
        t = now if now is not None else self._clock()
        start_ts = SlotContext.slot_for(t)
        ctx = SlotContext(
            slot_start_ts=start_ts,
            slot_end_ts=start_ts + SLOT_INTERVAL_S,
            strike_price=strike_price,
            strike_source=strike_source,
            btc_ref_price=btc_ref_price,
            captured_at=t,
        )
        with self._lock:
            self._ctx = ctx
        return ctx

    def update_from_chainlink(
        self,
        chainlink_feed,
        fallback_question: str = "",
        parse_strike_fn=None,
        now: Optional[float] = None,
    ) -> SlotContext:
        """
        Pull strike and ref price from ChainlinkFeed; fall back to regex parse.
        Replaces the bot.py:1157-1171 strike-resolution block.
        """
        if parse_strike_fn is None:
            from src.models import parse_strike_price as parse_strike_fn  # type: ignore[assignment]

        strike: Optional[float] = None
        source = "unknown"
        btc_ref: Optional[float] = None

        if chainlink_feed is not None:
            latest = chainlink_feed.get_latest()
            if latest is not None:
                btc_ref = latest.price

            slot_open = chainlink_feed.get_slot_open_price()
            if slot_open is not None:
                strike = slot_open.price
                source = "chainlink"

        if strike is None and fallback_question:
            parsed = parse_strike_fn(fallback_question)
            if parsed is not None:
                strike = parsed
                source = "regex"

        return self.update(
            strike_price=strike,
            strike_source=source,
            btc_ref_price=btc_ref,
            now=now,
        )

    def sync_to_bot_state(self, state) -> None:
        """
        Write current slot's Chainlink fields into BotState for persistence.
        Replaces _snapshot_chainlink_state in bot.py (except chainlink_healthy,
        which the caller sets directly from chainlink_feed.is_healthy()).
        """
        ctx = self.get()
        if ctx is None:
            state.chainlink_ref_price = None
            state.chainlink_ref_slot_ts = None
            return
        if ctx.strike_source == "chainlink":
            state.chainlink_ref_price = ctx.strike_price
            state.chainlink_ref_slot_ts = ctx.slot_start_ts
        else:
            state.chainlink_ref_price = None
            state.chainlink_ref_slot_ts = None
