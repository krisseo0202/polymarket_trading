"""Tests for BtcPriceFeed using direct message injection (no real WebSocket)."""

import json
import time

import pytest

from src.utils.btc_feed import BtcPriceFeed, BookSnapshot


def _make_feed(**kwargs) -> BtcPriceFeed:
    """Create a feed with fast stale thresholds for testing."""
    defaults = dict(stale_warn_s=0.2, reconnect_s=1.0, buffer_s=10.0)
    defaults.update(kwargs)
    return BtcPriceFeed(**defaults)


def _tick(bid: float, ask: float) -> str:
    return json.dumps({"b": str(bid), "a": str(ask)})


class TestInitialState:
    def test_no_data_before_first_message(self):
        feed = _make_feed()
        assert feed.get_latest_mid() is None
        assert feed.get_latest_book() is None
        assert feed.get_feed_age_ms() is None

    def test_is_healthy_false_before_first_message(self):
        feed = _make_feed()
        assert feed.is_healthy() is False


class TestMessageInjection:
    def test_mid_price_computed_correctly(self):
        feed = _make_feed()
        feed._handle_message(_tick(bid=50000.0, ask=50020.0))
        assert feed.get_latest_mid() == pytest.approx(50010.0)

    def test_book_snapshot_fields(self):
        feed = _make_feed()
        feed._handle_message(_tick(bid=49990.0, ask=50010.0))
        book = feed.get_latest_book()
        assert book is not None
        assert book.bid == 49990.0
        assert book.ask == 50010.0
        assert book.mid == pytest.approx(50000.0)

    def test_is_healthy_true_after_fresh_message(self):
        feed = _make_feed(stale_warn_s=5.0)
        feed._handle_message(_tick(50000, 50010))
        assert feed.is_healthy() is True

    def test_multiple_messages_update_latest(self):
        feed = _make_feed()
        feed._handle_message(_tick(50000, 50010))
        feed._handle_message(_tick(51000, 51010))
        assert feed.get_latest_mid() == pytest.approx(51005.0)


class TestStaleness:
    def test_is_healthy_false_when_stale(self):
        feed = _make_feed(stale_warn_s=0.1)
        feed._handle_message(_tick(50000, 50010))
        # Backdate the timestamp to simulate staleness
        with feed._lock:
            old = feed._latest
            feed._latest = BookSnapshot(
                bid=old.bid,
                ask=old.ask,
                mid=old.mid,
                exchange_ts=None,
                local_ts=time.time() - 1.0,  # 1 second ago
            )
        assert feed.is_healthy() is False

    def test_feed_age_ms_increases_over_time(self):
        feed = _make_feed()
        feed._handle_message(_tick(50000, 50010))
        with feed._lock:
            old = feed._latest
            feed._latest = BookSnapshot(
                bid=old.bid, ask=old.ask, mid=old.mid,
                exchange_ts=None, local_ts=time.time() - 0.5,
            )
        age = feed.get_feed_age_ms()
        assert age is not None
        assert age >= 400  # at least 400ms


class TestPriceBuffer:
    def test_get_recent_prices_empty_initially(self):
        feed = _make_feed()
        assert feed.get_recent_prices(60) == []

    def test_get_recent_prices_returns_injected_ticks(self):
        feed = _make_feed(buffer_s=60.0)
        feed._handle_message(_tick(50000, 50010))
        feed._handle_message(_tick(50100, 50110))
        prices = feed.get_recent_prices(60)
        assert len(prices) == 2
        mids = [m for _, m in prices]
        assert pytest.approx(50005.0) in mids
        assert pytest.approx(50105.0) in mids

    def test_get_recent_prices_respects_time_window(self):
        feed = _make_feed(buffer_s=300.0)
        # Inject an old tick by backdating buffer entry
        now = time.time()
        with feed._lock:
            feed._buffer.append((now - 120.0, 49000.0))  # 2 minutes ago
            feed._buffer.append((now - 10.0, 50000.0))   # 10 seconds ago

        recent = feed.get_recent_prices(60)   # last 60s only
        assert len(recent) == 1
        assert recent[0][1] == 50000.0

    def test_get_recent_prices_ordered_oldest_first(self):
        feed = _make_feed(buffer_s=300.0)
        now = time.time()
        with feed._lock:
            feed._buffer.append((now - 5.0, 50000.0))
            feed._buffer.append((now - 2.0, 50100.0))
            feed._buffer.append((now - 0.5, 50200.0))
        prices = feed.get_recent_prices(60)
        ts_list = [ts for ts, _ in prices]
        assert ts_list == sorted(ts_list)


class TestMalformedMessages:
    def test_malformed_json_silently_skipped(self):
        feed = _make_feed()
        feed._handle_message("not valid json {{{")
        assert feed.get_latest_mid() is None  # no state change

    def test_missing_bid_key_silently_skipped(self):
        feed = _make_feed()
        feed._handle_message(json.dumps({"a": "50010.0"}))  # no "b"
        assert feed.get_latest_mid() is None

    def test_missing_ask_key_silently_skipped(self):
        feed = _make_feed()
        feed._handle_message(json.dumps({"b": "50000.0"}))  # no "a"
        assert feed.get_latest_mid() is None

    def test_subscription_ack_silently_skipped(self):
        feed = _make_feed()
        feed._handle_message(json.dumps({"result": None, "id": 1}))
        assert feed.get_latest_mid() is None

    def test_non_numeric_prices_silently_skipped(self):
        feed = _make_feed()
        feed._handle_message(json.dumps({"b": "not_a_number", "a": "50010.0"}))
        assert feed.get_latest_mid() is None
