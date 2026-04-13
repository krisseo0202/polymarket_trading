"""Single source of truth for the current 5-minute Polymarket slot."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from .inventory import InventoryState
    from .state_store import BotState
    from .risk_manager import RiskManager

_GAMMA_API = "https://gamma-api.polymarket.com"

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

    def __init__(
        self,
        clock_fn=time.time,
        logger: Optional[logging.Logger] = None,
    ):
        self._clock = clock_fn
        self._logger = logger or logging.getLogger("slot_state")
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
        btc_feed=None,
        now: Optional[float] = None,
    ) -> SlotContext:
        """
        Pull strike and ref price from ChainlinkFeed; fall back to regex parse,
        then to BTC feed slot-boundary price.

        Validates that the slot-open price belongs to the *current* slot before
        using it. A slot mismatch (e.g. after a feed reconnect mid-slot) triggers
        a fallback to regex parsing and a warning log.
        """
        if parse_strike_fn is None:
            from src.models import parse_strike_price as parse_strike_fn  # type: ignore[assignment]

        t = now if now is not None else self._clock()
        current_slot_ts = SlotContext.slot_for(t)

        strike: Optional[float] = None
        source = "unknown"
        btc_ref: Optional[float] = None

        if chainlink_feed is not None:
            latest = chainlink_feed.get_latest()
            if latest is not None:
                btc_ref = latest.price

            slot_open = chainlink_feed.get_slot_open_price()
            if slot_open is not None:
                if slot_open.slot_ts == current_slot_ts:
                    strike = slot_open.price
                    source = "chainlink"
                else:
                    # Feed has a price but it's from a different slot — stale or reconnected.
                    age_slots = (current_slot_ts - slot_open.slot_ts) // SLOT_INTERVAL_S
                    self._logger.warning(
                        f"Chainlink slot_open ({slot_open.slot_ts}) is {age_slots} slot(s) "
                        f"behind current ({current_slot_ts}); falling back"
                    )

        if strike is None and fallback_question:
            parsed = parse_strike_fn(fallback_question)
            if parsed is not None:
                strike = parsed
                source = "regex"
                self._logger.debug(
                    f"Strike resolved via regex: ${strike:,.2f} "
                    f"(Chainlink {'stale' if chainlink_feed else 'disabled'})"
                )

        # Fallback: use the BTC feed's price closest to the slot boundary.
        # Not perfect (Coinbase != Chainlink), but close enough to keep the
        # model running rather than skipping every cycle.
        if strike is None and btc_feed is not None:
            strike = _btc_feed_slot_open(btc_feed, current_slot_ts)
            if strike is not None:
                source = "btc_feed"
                self._logger.info(
                    f"Strike resolved via BTC feed: ${strike:,.2f} "
                    f"(Chainlink unavailable)"
                )

        if strike is None:
            self._logger.debug("Strike price unavailable (no Chainlink, no regex, no BTC feed)")

        return self.update(
            strike_price=strike,
            strike_source=source,
            btc_ref_price=btc_ref,
            now=t,
        )

    def sync_to_bot_state(self, state) -> None:  # type: ignore[override]
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


def _btc_feed_slot_open(btc_feed, slot_ts: int) -> Optional[float]:
    """Find the BTC feed price closest to a slot boundary.

    Returns None if no buffered price is within 30s of the slot start.
    Prices are sorted by timestamp so we use bisect for O(log n) lookup.
    """
    import bisect

    prices = btc_feed.get_recent_prices(window_s=600)
    if not prices:
        return None

    timestamps = [entry[0] for entry in prices]
    idx = bisect.bisect_left(timestamps, slot_ts)

    best: Optional[float] = None
    best_dist = float("inf")
    # Check the entry at and around the insertion point
    for i in (idx - 1, idx):
        if 0 <= i < len(prices):
            dist = abs(timestamps[i] - slot_ts)
            if dist < best_dist:
                best_dist = dist
                best = float(prices[i][1])

    if best is not None and best_dist <= 30.0:
        return best
    return None


def fetch_slot_market(slot_ts: int, logger) -> Optional[dict]:
    """Fetch the full market info for a slot from Gamma API.

    Returns dict with keys: outcome ("Up"/"Down"), yes_token_id, no_token_id.
    Returns None if the market is not yet resolved or on API error.
    """
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        resp = requests.get(
            f"{_GAMMA_API}/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
        if not events:
            return None
        market = (events[0].get("markets") or [{}])[0]
        closed = market.get("closed", False)
        raw = market.get("outcomePrices", "")
        outcome_prices: list = json.loads(raw) if isinstance(raw, str) else (raw or [])
        if not closed or len(outcome_prices) < 2:
            return None

        outcome: Optional[str] = None
        if float(outcome_prices[0]) > 0.9:
            outcome = "Up"
        elif float(outcome_prices[1]) > 0.9:
            outcome = "Down"
        if outcome is None:
            return None

        raw_tokens = market.get("clobTokenIds", "")
        if isinstance(raw_tokens, str) and raw_tokens:
            token_ids: list = json.loads(raw_tokens)
        elif isinstance(raw_tokens, list):
            token_ids = raw_tokens
        else:
            token_ids = []

        return {
            "outcome": outcome,
            "yes_token_id": token_ids[0] if len(token_ids) > 0 else "",
            "no_token_id": token_ids[1] if len(token_ids) > 1 else "",
        }
    except Exception as exc:
        logger.warning(f"Gamma API error fetching market for slot {slot_ts}: {exc}")
        return None


def fetch_slot_outcome(slot_ts: int, logger) -> Optional[str]:
    """
    Query Gamma API for the resolution outcome of a specific 5-min slot.
    Returns "Up", "Down", or None (unresolved / API error).
    """
    info = fetch_slot_market(slot_ts, logger)
    return info["outcome"] if info else None


def apply_slot_settlement(
    yes_token_id: str,
    no_token_id: str,
    outcome: str,
    inventories: "Dict[str, InventoryState]",
    state: "BotState",
    risk_manager: "RiskManager",
    paper_trading: bool,
    logger,
    label_prefix: str = "Settlement",
) -> int:
    """Apply synthetic settlement SELLs for any open positions, given a known outcome.

    Returns the number of positions settled.
    """
    from .inventory import apply_fill_to_state  # local import avoids circular at module level

    settlement = {
        "YES": 0.99 if outcome == "Up" else 0.01,
        "NO":  0.99 if outcome == "Down" else 0.01,
    }
    settled = 0
    for token_id, side in ((yes_token_id, "YES"), (no_token_id, "NO")):
        inv = inventories.get(token_id)
        if inv is None or inv.position <= 0:
            continue
        price = settlement[side]
        size = inv.position
        entry_cost = inv.avg_cost
        realized = apply_fill_to_state(inv, "SELL", price, size, state, risk_manager)
        if realized > 0:
            state.session_wins += 1
        elif realized < 0:
            state.session_losses += 1

        # Record settlement in the trade log so the dashboard and post-hoc
        # analysis can show the full lifecycle: BUY entry → settlement result.
        # Before this fix, only BUY entries were logged and settlements were
        # invisible (realized_pnl_delta always showed 0.0000).
        state.trade_log.append({
            "ts": time.time(),
            "action": "SETTLE",
            "outcome": side,
            "price": price,
            "size": size,
            "entry_price": entry_cost,
            "resolved_outcome": outcome,
            "realized_pnl_delta": realized,
            "token_id": token_id,
            "strategy_name": state.strategy_name,
        })
        state.trade_log = state.trade_log[-20:]

        label = "paper" if paper_trading else "live"
        logger.info(
            f"[{label}] {label_prefix}: {side} {token_id[:8]} "
            f"{size:.2f}sh @ {price:.2f} (entry {entry_cost:.4f}) "
            f"→ realized {realized:+.4f} (outcome: {outcome})"
        )
        settled += 1
    return settled


def settle_expiring_positions(
    yes_token_id: str,
    no_token_id: str,
    slot_ts: int,
    inventories: "Dict[str, InventoryState]",
    state: "BotState",
    risk_manager: "RiskManager",
    paper_trading: bool,
    logger,
) -> Optional[str]:
    """
    Apply synthetic settlement SELLs for open positions in an expired slot.

    Called at market rollover BEFORE resetting slot_realized_pnl.
    """
    has_open = any(
        inventories.get(tid) is not None and inventories[tid].position > 0
        for tid in (yes_token_id, no_token_id)
    )
    if not has_open:
        return None

    outcome = fetch_slot_outcome(slot_ts, logger)
    if outcome is None:
        logger.warning(
            f"Cannot settle slot {slot_ts}: outcome not yet available "
            "(market may still be resolving — positions left open)"
        )
        return None

    apply_slot_settlement(
        yes_token_id, no_token_id, outcome,
        inventories, state, risk_manager, paper_trading, logger,
        label_prefix="Settlement SELL",
    )
    return outcome
