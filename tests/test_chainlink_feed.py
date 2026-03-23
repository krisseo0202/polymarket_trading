"""Unit tests for ChainlinkFeed — slot-open capture, message parsing, health."""

import json
import time
from unittest.mock import patch

import pytest

from src.utils.chainlink_feed import ChainlinkFeed, _RTDS_TOPIC


@pytest.fixture
def feed():
    """Create a ChainlinkFeed without starting WebSocket threads."""
    return ChainlinkFeed(
        symbol="btc/usd",
        slot_interval_s=300,
        stale_warn_s=10.0,
        buffer_s=600.0,
    )


def _make_msg(symbol: str, value: float, ts: float) -> str:
    """Build a JSON RTDS message matching the expected format."""
    return json.dumps([
        _RTDS_TOPIC,
        {"symbol": symbol, "value": value, "timestamp": ts},
    ])


def _make_snapshot_msg(symbol: str, points) -> str:
    """Build a current RTDS snapshot message with payload.data."""
    return json.dumps({
        "topic": "crypto_prices",
        "type": "subscribe",
        "timestamp": int(time.time() * 1000),
        "payload": {
            "symbol": symbol,
            "data": points,
        },
    })


class TestHandleMessageValid:
    def test_valid_payload_sets_latest(self, feed):
        now = time.time()
        raw = _make_msg("btc/usd", 84500.0, now)
        feed._handle_message(raw)

        latest = feed.get_latest()
        assert latest is not None
        assert latest.price == 84500.0
        assert latest.symbol == "btc/usd"

    def test_valid_payload_sets_reference_price(self, feed):
        now = time.time()
        raw = _make_msg("btc/usd", 84500.0, now)
        feed._handle_message(raw)

        ref = feed.get_reference_price()
        assert ref == 84500.0

    def test_snapshot_payload_sets_latest_and_reference_price(self, feed):
        slot_ts = (1700000100 // 300) * 300
        raw = _make_snapshot_msg(
            "btc/usd",
            [
                {"timestamp": (slot_ts - 5) * 1000, "value": 84490.0},
                {"timestamp": slot_ts * 1000, "value": 84500.0},
                {"timestamp": (slot_ts + 12) * 1000, "value": 84510.0},
            ],
        )
        feed._handle_message(raw)

        latest = feed.get_latest()
        assert latest is not None
        assert latest.price == 84510.0
        assert latest.exchange_ts == float(slot_ts + 12)

        slot_open = feed.get_slot_open_price()
        assert slot_open is not None
        assert slot_open.slot_ts == (slot_ts // 300) * 300
        assert slot_open.price == 84500.0


class TestHandleMessageInvalid:
    def test_malformed_json_no_crash(self, feed):
        feed._handle_message("not json {{{")
        assert feed.get_latest() is None

    def test_wrong_topic_ignored(self, feed):
        raw = json.dumps(["wrong_topic", {"symbol": "btc/usd", "value": 100, "timestamp": 0}])
        feed._handle_message(raw)
        assert feed.get_latest() is None

    def test_wrong_symbol_ignored(self, feed):
        raw = _make_msg("eth/usd", 3000.0, time.time())
        feed._handle_message(raw)
        assert feed.get_latest() is None

    def test_missing_value_no_crash(self, feed):
        raw = json.dumps([_RTDS_TOPIC, {"symbol": "btc/usd", "timestamp": 0}])
        feed._handle_message(raw)
        assert feed.get_latest() is None


class TestSlotOpenCapture:
    def test_first_price_in_slot_captured(self, feed):
        # Use a deterministic time aligned to a slot boundary
        slot_ts = 1700000100  # arbitrary, within some slot
        slot_floor = (slot_ts // 300) * 300

        with patch("src.utils.chainlink_feed.time") as mock_time:
            mock_time.time.return_value = float(slot_ts)
            raw = _make_msg("btc/usd", 84500.0, float(slot_ts))
            feed._handle_message(raw)

        slot_open = feed.get_slot_open_price()
        assert slot_open is not None
        assert slot_open.slot_ts == slot_floor
        assert slot_open.price == 84500.0

    def test_subsequent_price_same_slot_no_overwrite(self, feed):
        slot_ts = 1700000100
        slot_floor = (slot_ts // 300) * 300

        with patch("src.utils.chainlink_feed.time") as mock_time:
            # First price
            mock_time.time.return_value = float(slot_ts)
            feed._handle_message(_make_msg("btc/usd", 84500.0, float(slot_ts)))

            # Second price, same slot
            mock_time.time.return_value = float(slot_ts + 30)
            feed._handle_message(_make_msg("btc/usd", 84600.0, float(slot_ts + 30)))

        slot_open = feed.get_slot_open_price()
        assert slot_open.price == 84500.0  # first price preserved

    def test_new_slot_captures_new_price(self, feed):
        slot1_ts = 1700000100
        slot2_ts = slot1_ts + 300  # next slot

        with patch("src.utils.chainlink_feed.time") as mock_time:
            mock_time.time.return_value = float(slot1_ts)
            feed._handle_message(_make_msg("btc/usd", 84500.0, float(slot1_ts)))

            mock_time.time.return_value = float(slot2_ts)
            feed._handle_message(_make_msg("btc/usd", 85000.0, float(slot2_ts)))

        slot_open = feed.get_slot_open_price()
        assert slot_open.price == 85000.0  # new slot's price

    def test_snapshot_ms_timestamps_backfill_current_slot_open(self, feed):
        slot_ts = (1700000100 // 300) * 300
        raw = _make_snapshot_msg(
            "btc/usd",
            [
                {"timestamp": (slot_ts - 1) * 1000, "value": 84499.0},
                {"timestamp": slot_ts * 1000, "value": 84500.0},
                {"timestamp": (slot_ts + 1) * 1000, "value": 84501.0},
            ],
        )
        feed._handle_message(raw)

        slot_open = feed.get_slot_open_price()
        assert slot_open is not None
        assert slot_open.slot_ts == (slot_ts // 300) * 300
        assert slot_open.price == 84500.0


class TestBufferTrimming:
    def test_old_entries_trimmed(self, feed):
        now = time.time()
        # First entry: 700s ago
        with patch("src.utils.chainlink_feed.time") as mock_time:
            mock_time.time.return_value = now - 700
            feed._handle_message(_make_msg("btc/usd", 84000.0, now - 700))

            # Second entry: now (trims the first since buffer_s=600)
            mock_time.time.return_value = now
            feed._handle_message(_make_msg("btc/usd", 85000.0, now))

        # Only the recent entry should remain (first was >600s ago, trimmed)
        prices = feed.get_recent_prices(window_s=800)
        assert len(prices) == 1
        assert prices[0][1] == 85000.0


class TestIsHealthy:
    def test_no_data_unhealthy(self, feed):
        assert feed.is_healthy() is False

    def test_fresh_data_healthy(self, feed):
        feed._handle_message(_make_msg("btc/usd", 84500.0, time.time()))
        assert feed.is_healthy() is True

    def test_stale_data_unhealthy(self, feed):
        old_ts = time.time() - 20  # well beyond 10s stale threshold
        with patch("src.utils.chainlink_feed.time") as mock_time:
            mock_time.time.return_value = old_ts
            feed._handle_message(_make_msg("btc/usd", 84500.0, old_ts))

        # Now check health at current time — feed age > stale threshold
        assert feed.is_healthy() is False


class TestGetRecentPrices:
    def test_returns_within_window(self, feed):
        now = time.time()
        with patch("src.utils.chainlink_feed.time") as mock_time:
            mock_time.time.return_value = now - 100
            feed._handle_message(_make_msg("btc/usd", 84000.0, now - 100))
            mock_time.time.return_value = now - 10
            feed._handle_message(_make_msg("btc/usd", 84500.0, now - 10))

        prices = feed.get_recent_prices(window_s=60)
        assert len(prices) == 1  # only the recent one within 60s
        assert prices[0][1] == 84500.0
