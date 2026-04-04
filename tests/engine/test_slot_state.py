"""Tests for SlotContext and SlotStateManager."""

import pytest
from unittest.mock import MagicMock

from src.engine.slot_state import SLOT_INTERVAL_S, SlotContext, SlotStateManager


# ── SlotContext static / instance behaviour ───────────────────────────────────

_SLOT_TS = 1_699_999_800  # floor(1_699_999_800 / 300) * 300 — a true 300-aligned boundary


class TestSlotContext:
    def test_slot_for_aligns_to_boundary(self):
        # _SLOT_TS + 150 is mid-slot; should map back to _SLOT_TS
        assert SlotContext.slot_for(_SLOT_TS + 150) == _SLOT_TS

    def test_slot_end_is_start_plus_300(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert ctx.slot_end_ts == ctx.slot_start_ts + SLOT_INTERVAL_S

    def test_seconds_remaining_at_start(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert ctx.seconds_remaining(now=float(_SLOT_TS)) == 300.0

    def test_seconds_remaining_at_end(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert ctx.seconds_remaining(now=float(_SLOT_TS + 300)) == 0.0

    def test_seconds_remaining_clamps_to_zero(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert ctx.seconds_remaining(now=float(_SLOT_TS + 400)) == 0.0

    def test_is_same_slot_true_inside(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert ctx.is_same_slot(float(_SLOT_TS + 150))

    def test_is_same_slot_false_at_end(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        # slot_end_ts is exclusive
        assert not ctx.is_same_slot(float(_SLOT_TS + 300))

    def test_is_same_slot_false_before(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert not ctx.is_same_slot(float(_SLOT_TS - 1))


# ── SlotStateManager.update ───────────────────────────────────────────────────

class TestSlotStateManagerUpdate:
    def test_get_returns_none_before_first_update(self):
        mgr = SlotStateManager(clock_fn=lambda: 1_700_000_150.0)
        assert mgr.get() is None

    def test_update_returns_correct_fields(self):
        mgr = SlotStateManager(clock_fn=lambda: float(_SLOT_TS + 50))
        ctx = mgr.update(strike_price=84500.0, strike_source="chainlink", btc_ref_price=84510.0)
        assert ctx.slot_start_ts == _SLOT_TS
        assert ctx.slot_end_ts == _SLOT_TS + 300
        assert ctx.strike_price == 84500.0
        assert ctx.strike_source == "chainlink"
        assert ctx.btc_ref_price == 84510.0

    def test_get_returns_latest_context(self):
        ts = 1_700_000_000.0
        mgr = SlotStateManager(clock_fn=lambda: ts)
        mgr.update(strike_price=100.0, strike_source="regex", btc_ref_price=None)
        ctx = mgr.update(strike_price=200.0, strike_source="chainlink", btc_ref_price=201.0)
        assert mgr.get() is ctx
        assert mgr.get().strike_price == 200.0

    def test_update_uses_injected_clock(self):
        # mid-slot: _SLOT_TS + 150
        mgr = SlotStateManager(clock_fn=lambda: float(_SLOT_TS + 150))
        ctx = mgr.update(strike_price=None, strike_source="unknown", btc_ref_price=None)
        assert ctx.slot_start_ts == _SLOT_TS
        assert ctx.slot_end_ts == _SLOT_TS + 300

    def test_current_slot_ts_uses_injected_clock(self):
        mgr = SlotStateManager(clock_fn=lambda: float(_SLOT_TS + 150))
        assert mgr.current_slot_ts() == _SLOT_TS

    def test_seconds_remaining_uses_injected_clock(self):
        # 100s into slot
        mgr = SlotStateManager(clock_fn=lambda: float(_SLOT_TS + 100))
        assert mgr.seconds_remaining() == pytest.approx(200.0, abs=0.01)


# ── SlotStateManager.update_from_chainlink ────────────────────────────────────

_TEST_NOW = 1_700_000_100.0
# slot_for(1_700_000_100) = 1_700_000_100 (it's exactly divisible by 300)
_TEST_SLOT_TS = int(_TEST_NOW // 300 * 300)  # == 1_700_000_100


class TestUpdateFromChainlink:
    def _mock_feed(self, slot_open_price=None, latest_price=None, slot_ts=_TEST_SLOT_TS):
        feed = MagicMock()
        # SlotOpenPrice mock — slot_ts must match the test clock's current slot
        if slot_open_price is not None:
            slot_open = MagicMock()
            slot_open.price = slot_open_price
            slot_open.slot_ts = slot_ts
            feed.get_slot_open_price.return_value = slot_open
        else:
            feed.get_slot_open_price.return_value = None
        # Latest tick mock
        if latest_price is not None:
            latest = MagicMock()
            latest.price = latest_price
            feed.get_latest.return_value = latest
        else:
            feed.get_latest.return_value = None
        return feed

    def test_chainlink_slot_open_used_as_strike(self):
        feed = self._mock_feed(slot_open_price=84500.0, latest_price=84510.0)
        mgr = SlotStateManager(clock_fn=lambda: _TEST_NOW)
        ctx = mgr.update_from_chainlink(feed, fallback_question="BTC above $84,000?")
        assert ctx.strike_price == 84500.0
        assert ctx.strike_source == "chainlink"

    def test_stale_slot_open_falls_back_to_regex(self):
        """slot_open from previous slot should not be used; regex fallback instead."""
        stale_slot_ts = _TEST_SLOT_TS - 300  # one slot behind
        feed = self._mock_feed(slot_open_price=84500.0, latest_price=84510.0, slot_ts=stale_slot_ts)
        mgr = SlotStateManager(clock_fn=lambda: _TEST_NOW)
        parse_fn = lambda q: 84100.0  # noqa: E731
        ctx = mgr.update_from_chainlink(
            feed, fallback_question="BTC above $84,100?", parse_strike_fn=parse_fn
        )
        assert ctx.strike_price == 84100.0
        assert ctx.strike_source == "regex"

    def test_regex_fallback_when_no_slot_open(self):
        feed = self._mock_feed(slot_open_price=None, latest_price=84510.0)
        mgr = SlotStateManager(clock_fn=lambda: _TEST_NOW)
        parse_fn = lambda q: 84000.0  # noqa: E731
        ctx = mgr.update_from_chainlink(
            feed, fallback_question="BTC above $84,000?", parse_strike_fn=parse_fn
        )
        assert ctx.strike_price == 84000.0
        assert ctx.strike_source == "regex"

    def test_unknown_source_when_both_missing(self):
        feed = self._mock_feed(slot_open_price=None, latest_price=None)
        mgr = SlotStateManager(clock_fn=lambda: _TEST_NOW)
        parse_fn = lambda q: None  # noqa: E731
        ctx = mgr.update_from_chainlink(feed, fallback_question="", parse_strike_fn=parse_fn)
        assert ctx.strike_price is None
        assert ctx.strike_source == "unknown"

    def test_none_feed_falls_back_to_regex(self):
        mgr = SlotStateManager(clock_fn=lambda: _TEST_NOW)
        parse_fn = lambda q: 83000.0  # noqa: E731
        ctx = mgr.update_from_chainlink(
            None, fallback_question="BTC above $83,000?", parse_strike_fn=parse_fn
        )
        assert ctx.strike_price == 83000.0
        assert ctx.strike_source == "regex"

    def test_btc_ref_price_from_latest_tick(self):
        feed = self._mock_feed(slot_open_price=84500.0, latest_price=84510.0)
        mgr = SlotStateManager(clock_fn=lambda: _TEST_NOW)
        ctx = mgr.update_from_chainlink(feed)
        assert ctx.btc_ref_price == 84510.0


# ── SlotStateManager.sync_to_bot_state ───────────────────────────────────────

class TestSyncToBotState:
    def _make_state(self):
        state = MagicMock()
        state.chainlink_ref_price = None
        state.chainlink_ref_slot_ts = None
        return state

    def test_chainlink_source_writes_fields(self):
        mgr = SlotStateManager(clock_fn=lambda: float(_SLOT_TS))
        mgr.update(strike_price=84500.0, strike_source="chainlink", btc_ref_price=84510.0)
        state = self._make_state()
        mgr.sync_to_bot_state(state)
        assert state.chainlink_ref_price == 84500.0
        assert state.chainlink_ref_slot_ts == _SLOT_TS

    def test_regex_source_clears_chainlink_fields(self):
        ts = 1_700_000_000.0
        mgr = SlotStateManager(clock_fn=lambda: ts)
        mgr.update(strike_price=84000.0, strike_source="regex", btc_ref_price=None)
        state = self._make_state()
        state.chainlink_ref_price = 99999.0  # pre-existing value
        mgr.sync_to_bot_state(state)
        assert state.chainlink_ref_price is None
        assert state.chainlink_ref_slot_ts is None

    def test_no_context_clears_fields(self):
        mgr = SlotStateManager(clock_fn=lambda: 1_700_000_000.0)
        # no update called
        state = self._make_state()
        state.chainlink_ref_price = 99999.0
        mgr.sync_to_bot_state(state)
        assert state.chainlink_ref_price is None
        assert state.chainlink_ref_slot_ts is None


# ── Rollover detection ────────────────────────────────────────────────────────

class TestRolloverDetection:
    def test_rollover_detected_via_is_same_slot(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert not ctx.is_same_slot(float(_SLOT_TS + 300))

    def test_no_rollover_mid_slot(self):
        ctx = SlotContext(
            slot_start_ts=_SLOT_TS, slot_end_ts=_SLOT_TS + 300,
            strike_price=None, strike_source="unknown",
            btc_ref_price=None, captured_at=float(_SLOT_TS),
        )
        assert ctx.is_same_slot(float(_SLOT_TS + 150))
