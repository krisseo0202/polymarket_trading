"""Integration tests for run_pipeline().

All external I/O is mocked — no live WebSocket, no HTTP calls.
Uses a 2-second cycle so tests complete quickly.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import PipelineConfig, run_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(**overrides) -> PipelineConfig:
    defaults = dict(
        market_id="test-id",
        cycle_len=2,
        trigger_window_s=0.5,
        n_paths=100,
        dry_run=True,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _healthy_feed_mock() -> MagicMock:
    """BtcPriceFeed mock that is healthy and returns plausible prices."""
    feed = MagicMock()
    feed.is_healthy.return_value = True
    feed.get_latest_mid.return_value = 50000.0
    feed.get_recent_prices.return_value = [
        (time.time() - i, 50000.0 + i * 0.1) for i in range(60)
    ]
    return feed


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPipelineRuns:
    @patch("src.pipeline.fetch_market_odds", return_value=(0.60, 0.40))
    @patch("src.pipeline.BtcPriceFeed")
    def test_stop_event_terminates_thread_promptly(self, MockFeed, mock_odds):
        MockFeed.return_value = _healthy_feed_mock()
        config = _make_config()
        stop = threading.Event()

        t = threading.Thread(
            target=run_pipeline,
            args=(config,),
            kwargs=dict(stop_event=stop),
            daemon=True,
        )
        t.start()
        time.sleep(0.2)
        stop.set()
        t.join(timeout=1.5)

        assert not t.is_alive(), "Pipeline thread should have exited after stop_event"

    @patch("src.pipeline.fetch_market_odds", return_value=(0.60, 0.40))
    @patch("src.pipeline.BtcPriceFeed")
    def test_fetch_market_odds_called_at_least_once(self, MockFeed, mock_odds):
        MockFeed.return_value = _healthy_feed_mock()
        config = _make_config()
        stop = threading.Event()

        t = threading.Thread(
            target=run_pipeline,
            args=(config,),
            kwargs=dict(stop_event=stop),
            daemon=True,
        )
        t.start()
        time.sleep(4.5)
        stop.set()
        t.join(timeout=1.5)

        assert mock_odds.call_count >= 1

    @patch("src.pipeline.fetch_market_odds", return_value=(0.60, 0.40))
    @patch("src.pipeline.BtcPriceFeed")
    def test_dry_run_no_place_order(self, MockFeed, mock_odds):
        """With dry_run=True and client=None, place_order must never be called."""
        feed_mock = _healthy_feed_mock()
        MockFeed.return_value = feed_mock

        mock_client = MagicMock()
        config = _make_config(dry_run=True)
        stop = threading.Event()

        t = threading.Thread(
            target=run_pipeline,
            args=(config,),
            kwargs=dict(client=mock_client, stop_event=stop),
            daemon=True,
        )
        t.start()
        time.sleep(4.5)
        stop.set()
        t.join(timeout=1.5)

        mock_client.place_order.assert_not_called()

    @patch("src.pipeline.fetch_market_odds", return_value=(0.60, 0.40))
    @patch("src.pipeline.BtcPriceFeed")
    def test_no_exceptions_raised(self, MockFeed, mock_odds):
        """Pipeline must not propagate exceptions from mocked components."""
        MockFeed.return_value = _healthy_feed_mock()
        config = _make_config()
        stop = threading.Event()
        exc_holder = []

        def _target():
            try:
                run_pipeline(config, stop_event=stop)
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        time.sleep(4.5)
        stop.set()
        t.join(timeout=1.5)

        assert exc_holder == [], f"Pipeline raised exception: {exc_holder}"


class TestUnhealthyFeedSkipsCycle:
    @patch("src.pipeline.fetch_market_odds", return_value=(0.60, 0.40))
    @patch("src.pipeline.BtcPriceFeed")
    def test_unhealthy_feed_skips_odds_call(self, MockFeed, mock_odds):
        """When feed.is_healthy() returns False, fetch_market_odds must NOT be called."""
        feed_mock = _healthy_feed_mock()
        feed_mock.is_healthy.return_value = False
        MockFeed.return_value = feed_mock

        config = _make_config()
        stop = threading.Event()

        t = threading.Thread(
            target=run_pipeline,
            args=(config,),
            kwargs=dict(stop_event=stop),
            daemon=True,
        )
        t.start()
        time.sleep(4.5)
        stop.set()
        t.join(timeout=1.5)

        mock_odds.assert_not_called()


class TestApiFailureHandling:
    @patch("src.pipeline.fetch_market_odds", side_effect=RuntimeError("503 timeout"))
    @patch("src.pipeline.BtcPriceFeed")
    def test_api_failure_skips_cycle_without_crash(self, MockFeed, mock_odds):
        """RuntimeError from fetch_market_odds must be caught; pipeline keeps running."""
        MockFeed.return_value = _healthy_feed_mock()
        config = _make_config()
        stop = threading.Event()
        exc_holder = []

        def _target():
            try:
                run_pipeline(config, stop_event=stop)
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        time.sleep(4.5)
        stop.set()
        t.join(timeout=1.5)

        assert exc_holder == [], "API failure should be caught, not propagated"
        assert not t.is_alive(), "Thread should have exited cleanly"
