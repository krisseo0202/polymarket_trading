from unittest.mock import patch

from rich.console import Console

import dashboard
from src.utils.chainlink_feed import SlotOpenPrice


class _FakeBtcFeed:
    class _Book:
        def __init__(self, mid: float):
            self.mid = mid

    def __init__(self, mid: float = 84_550.0):
        self._book = self._Book(mid)

    def get_latest_book(self):
        return self._book


class _FakeChainlinkFeed:
    def __init__(self, slot_open=None, healthy=True):
        self._slot_open = slot_open
        self._healthy = healthy

    def get_slot_open_price(self):
        return self._slot_open

    def is_healthy(self):
        return self._healthy


def test_live_chainlink_current_slot_is_preferred():
    current_slot = 1_700_000_100
    chainlink_feed = _FakeChainlinkFeed(
        SlotOpenPrice(slot_ts=current_slot, price=84_500.0, captured_at=0.0)
    )
    bot_state = {
        "chainlink_ref_price": 84_490.0,
        "chainlink_ref_slot_ts": current_slot,
    }

    price, source = dashboard._resolve_price_to_beat(
        chainlink_feed,
        bot_state=bot_state,
        current_slot=current_slot,
    )

    assert price == 84_500.0
    assert source == "Chainlink (live)"


def test_bot_snapshot_used_when_live_feed_missed_slot_open():
    current_slot = 1_700_000_100
    chainlink_feed = _FakeChainlinkFeed(slot_open=None)
    bot_state = {
        "chainlink_ref_price": 84_490.0,
        "chainlink_ref_slot_ts": current_slot,
    }

    price, source = dashboard._resolve_price_to_beat(
        chainlink_feed,
        bot_state=bot_state,
        current_slot=current_slot,
    )

    assert price == 84_490.0
    assert source == "Chainlink (bot snapshot)"


@patch("dashboard._fetch_slot_open_from_binance", return_value=None)
def test_previous_slot_bot_snapshot_is_rejected(mock_binance):
    current_slot = 1_700_000_100
    bot_state = {
        "chainlink_ref_price": 84_490.0,
        "chainlink_ref_slot_ts": current_slot - 300,
    }

    price, source = dashboard._resolve_price_to_beat(
        None,
        bot_state=bot_state,
        current_slot=current_slot,
    )

    assert price is None
    assert "Waiting" in source


@patch("dashboard._fetch_slot_open_from_binance", return_value=None)
def test_waiting_state_does_not_fallback_to_regex_price(mock_binance):
    current_slot = 1_700_000_100
    feed = _FakeBtcFeed(mid=84_550.0)
    market = {"title": "Will BTC be above $999,999 at 12:05 PM ET?", "end_ts": current_slot + 300}

    with patch("dashboard._server_now", return_value=float(current_slot + 120)):
        panel = dashboard._build_market_cycle_panel(
            feed,
            None,
            bot_state=None,
            market=market,
        )

    console = Console(record=True, width=120)
    console.print(panel)
    rendered = console.export_text()

    assert "Waiting" in rendered
    assert "$999,999" not in rendered


def test_market_cycle_panel_renders_price_to_beat_label():
    current_slot = 1_700_000_100
    market = {"title": "Will BTC be above $999,999 at 12:05 PM ET?", "end_ts": current_slot + 300}
    feed = _FakeBtcFeed(mid=84_550.0)
    chainlink_feed = _FakeChainlinkFeed(
        SlotOpenPrice(slot_ts=current_slot, price=84_500.0, captured_at=0.0)
    )

    with patch("dashboard._server_now", return_value=float(current_slot + 120)):
        panel = dashboard._build_market_cycle_panel(
            feed,
            chainlink_feed,
            bot_state=None,
            market=market,
        )

    console = Console(record=True, width=120)
    console.print(panel)
    rendered = console.export_text()

    assert "Price to beat" in rendered
    assert "$84,500.00" in rendered
    assert "Chainlink (live)" in rendered
