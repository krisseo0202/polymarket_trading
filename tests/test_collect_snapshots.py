"""Tests for scripts/collect_snapshots.py."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.collect_snapshots import (
    BookFetcher,
    MarketDiscovery,
    SnapshotCollector,
    _determine_outcome,
    _parse_json_field,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_levels(n: int, base_price: float = 0.50, base_size: float = 10.0):
    """Return n book levels as CLOB-style dicts."""
    return [{"price": str(base_price - i * 0.001), "size": str(base_size)} for i in range(n)]


# ── BookFetcher imbalance ─────────────────────────────────────────────────────

class TestBookFetcherImbalance:
    def _mock_response(self, bids, asks):
        resp = MagicMock()
        resp.json.return_value = {"bids": bids, "asks": asks}
        resp.raise_for_status = MagicMock()
        return resp

    def test_imbalance_top5_of_7_bids(self):
        """Only the top 5 bid levels should contribute to imbalance."""
        bids = _make_levels(7, base_size=10.0)   # 7 levels × 10 each → top-5 = 50
        asks = _make_levels(3, base_size=10.0)   # 3 levels × 10 each → 30
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response(bids, asks)):
            result = fetcher.fetch("dummy_token")
        assert result is not None
        # top-5 bids = 50, top-3 asks = 30 (only 3 exist)
        expected = (50 - 30) / (50 + 30)
        assert result.imbalance == pytest.approx(expected, abs=1e-9)

    def test_imbalance_empty_asks(self):
        """All bids, no asks → imbalance = 1.0."""
        bids = _make_levels(3, base_size=10.0)
        asks = []
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response(bids, asks)):
            result = fetcher.fetch("dummy_token")
        assert result is not None
        assert result.imbalance == pytest.approx(1.0)

    def test_imbalance_empty_bids(self):
        """No bids, all asks → imbalance = -1.0."""
        bids = []
        asks = _make_levels(3, base_size=10.0)
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response(bids, asks)):
            result = fetcher.fetch("dummy_token")
        assert result is not None
        assert result.imbalance == pytest.approx(-1.0)

    def test_empty_book_returns_none(self):
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response([], [])):
            result = fetcher.fetch("dummy_token")
        assert result is None

    def test_mid_and_spread_computed(self):
        bids = [{"price": "0.500", "size": "10"}]
        asks = [{"price": "0.510", "size": "10"}]
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response(bids, asks)):
            result = fetcher.fetch("dummy_token")
        assert result is not None
        assert result.mid == pytest.approx(0.505)
        assert result.spread == pytest.approx(0.010)

    def test_full_depth_fields_populated(self):
        """Full-depth fields: bids/asks dicts, total depth, level counts."""
        bids = _make_levels(7, base_price=0.50, base_size=10.0)
        asks = _make_levels(3, base_price=0.51, base_size=20.0)
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response(bids, asks)):
            result = fetcher.fetch("dummy_token")
        assert result is not None
        # Full book dicts contain all levels
        assert len(result.bids) == 7
        assert len(result.asks) == 3
        # Total depth (7 bids × 10 = 70, 3 asks × 20 = 60)
        assert result.bid_depth == pytest.approx(70.0)
        assert result.ask_depth == pytest.approx(60.0)
        # Level counts
        assert result.n_bid_levels == 7
        assert result.n_ask_levels == 3

    def test_full_bids_asks_are_price_size_dicts(self):
        """full_bids/full_asks are {price_str: size_str} dicts from the raw CLOB response."""
        bids = [{"price": "0.500", "size": "25"}, {"price": "0.499", "size": "15"}]
        asks = [{"price": "0.510", "size": "30"}]
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response(bids, asks)):
            result = fetcher.fetch("dummy_token")
        assert result.bids == {"0.500": "25", "0.499": "15"}
        assert result.asks == {"0.510": "30"}

    def test_depth_one_sided(self):
        """All bids, no asks → bid_depth populated, ask_depth = 0."""
        bids = _make_levels(4, base_size=10.0)
        asks = []
        fetcher = BookFetcher(top_n=5)
        with patch("requests.get", return_value=self._mock_response(bids, asks)):
            result = fetcher.fetch("dummy_token")
        assert result is not None
        assert result.bid_depth == pytest.approx(40.0)
        assert result.ask_depth == pytest.approx(0.0)
        assert result.n_bid_levels == 4
        assert result.n_ask_levels == 0


# ── MarketDiscovery caching ───────────────────────────────────────────────────

class TestMarketDiscovery:
    def _mock_gamma_response(self, slot_ts, closed=False, outcome_prices=None, token_ids=None):
        outcome_prices = outcome_prices or ["0.5", "0.5"]
        token_ids = token_ids or ["up_tok", "dn_tok"]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = [{
            "markets": [{
                "question": f"BTC slot {slot_ts}",
                "closed": closed,
                "outcomePrices": json.dumps(outcome_prices),
                "clobTokenIds": json.dumps(token_ids),
            }]
        }]
        return resp

    def test_caches_same_slot(self):
        slot_ts = 1_699_999_800
        disc = MarketDiscovery()
        mock_resp = self._mock_gamma_response(slot_ts)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            disc.get_current_market(slot_ts)
            disc.get_current_market(slot_ts)
        assert mock_get.call_count == 1

    def test_invalidates_on_new_slot(self):
        disc = MarketDiscovery()
        slot_a = 1_699_999_800
        slot_b = 1_700_000_100
        mock_resp = self._mock_gamma_response(slot_a)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            disc.get_current_market(slot_a)
            mock_get.return_value = self._mock_gamma_response(slot_b)
            disc.get_current_market(slot_b)
        assert mock_get.call_count == 2

    def test_returns_outcome_when_closed(self):
        slot_ts = 1_699_999_800
        disc = MarketDiscovery()
        mock_resp = self._mock_gamma_response(
            slot_ts, closed=True, outcome_prices=["0.99", "0.01"]
        )
        with patch("requests.get", return_value=mock_resp):
            outcome = disc.get_outcome_for_slot(slot_ts)
        assert outcome == "Up"

    def test_returns_none_when_not_closed(self):
        slot_ts = 1_699_999_800
        disc = MarketDiscovery()
        mock_resp = self._mock_gamma_response(
            slot_ts, closed=False, outcome_prices=["0.50", "0.50"]
        )
        with patch("requests.get", return_value=mock_resp):
            outcome = disc.get_outcome_for_slot(slot_ts)
        assert outcome is None

    def test_invalidate_clears_cache(self):
        slot_ts = 1_699_999_800
        disc = MarketDiscovery()
        mock_resp = self._mock_gamma_response(slot_ts)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            disc.get_current_market(slot_ts)
            disc.invalidate()
            disc.get_current_market(slot_ts)
        assert mock_get.call_count == 2


# ── Outcome sentinel logic ────────────────────────────────────────────────────

class TestOutcomeSentinel:
    def _make_collector(self, output_path):
        config = {"output": output_path, "snapshot_interval_s": 5}
        collector = SnapshotCollector(config)
        return collector

    def test_sentinel_written_on_rollover(self, tmp_path):
        output = str(tmp_path / "snaps.jsonl")
        collector = self._make_collector(output)
        # Mock discovery to return "Up"
        collector._discovery.get_outcome_for_slot = MagicMock(return_value="Up")
        collector._discovery.invalidate = MagicMock()
        with patch("time.sleep"):  # skip the 3s wait
            collector._on_slot_rollover(ended_slot_ts=1_699_999_800)
        with open(output) as f:
            line = json.loads(f.read().strip())
        assert line["type"] == "outcome"
        assert line["slot_ts"] == 1_699_999_800
        assert line["outcome"] == "Up"

    def test_no_sentinel_when_outcome_unavailable(self, tmp_path):
        output = str(tmp_path / "snaps.jsonl")
        collector = self._make_collector(output)
        collector._discovery.get_outcome_for_slot = MagicMock(return_value=None)
        collector._discovery.invalidate = MagicMock()
        with patch("time.sleep"):
            collector._on_slot_rollover(ended_slot_ts=1_699_999_800)
        assert not os.path.exists(output)


# ── Snapshot schema ───────────────────────────────────────────────────────────

class TestSnapshotSchema:
    REQUIRED_KEYS = {
        "slot_ts", "snapshot_ts",
        "up_token", "down_token",
        "strike", "strike_source",
        "btc_now", "btc_source",
        "yes_bid", "yes_ask", "yes_mid", "yes_spread", "yes_imbalance",
        "yes_bids", "yes_asks", "yes_bid_depth", "yes_ask_depth", "yes_n_levels",
        "no_bid", "no_ask", "no_mid", "no_spread", "no_imbalance",
        "no_bids", "no_asks", "no_bid_depth", "no_ask_depth", "no_n_levels",
        "realized_vol_30s", "realized_vol_60s",
    }

    def _make_collector(self, output_path):
        config = {"output": output_path, "snapshot_interval_s": 5}
        return SnapshotCollector(config)

    def test_all_keys_present_with_null_feeds(self, tmp_path):
        output = str(tmp_path / "snaps.jsonl")
        collector = self._make_collector(output)
        # Mock feeds to return None for everything
        collector._btc_feed = MagicMock()
        collector._btc_feed.get_latest_mid.return_value = None
        collector._btc_feed.get_recent_prices.return_value = []
        collector._chainlink_feed = MagicMock()
        collector._chainlink_feed.get_latest.return_value = None
        collector._chainlink_feed.get_slot_open_price.return_value = None
        collector._book_fetcher = MagicMock()
        collector._book_fetcher.fetch_pair.return_value = (None, None)

        market = {"up_token": "yes_tok", "down_token": "no_tok"}
        collector._collect_snapshot(slot_ts=1_699_999_800, market=market)

        with open(output) as f:
            record = json.loads(f.read().strip())

        missing = self.REQUIRED_KEYS - set(record.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_null_feeds_produce_null_values(self, tmp_path):
        output = str(tmp_path / "snaps.jsonl")
        collector = self._make_collector(output)
        collector._btc_feed = MagicMock()
        collector._btc_feed.get_latest_mid.return_value = None
        collector._btc_feed.get_recent_prices.return_value = []
        collector._chainlink_feed = MagicMock()
        collector._chainlink_feed.get_latest.return_value = None
        collector._chainlink_feed.get_slot_open_price.return_value = None
        collector._book_fetcher = MagicMock()
        collector._book_fetcher.fetch_pair.return_value = (None, None)

        market = {"up_token": "yes_tok", "down_token": "no_tok"}
        collector._collect_snapshot(slot_ts=1_699_999_800, market=market)

        with open(output) as f:
            record = json.loads(f.read().strip())

        assert record["btc_now"] is None
        assert record["strike"] is None
        assert record["yes_bid"] is None
        assert record["realized_vol_30s"] is None


# ── _parse_json_field and _determine_outcome ──────────────────────────────────

class TestHelpers:
    def test_parse_json_field_string(self):
        assert _parse_json_field('["a", "b"]') == ["a", "b"]

    def test_parse_json_field_list(self):
        assert _parse_json_field(["a", "b"]) == ["a", "b"]

    def test_parse_json_field_bad_string(self):
        assert _parse_json_field("not json") == []

    def test_determine_outcome_up(self):
        assert _determine_outcome(["0.99", "0.01"], closed=True) == "Up"

    def test_determine_outcome_down(self):
        assert _determine_outcome(["0.01", "0.99"], closed=True) == "Down"

    def test_determine_outcome_not_closed(self):
        assert _determine_outcome(["0.99", "0.01"], closed=False) is None

    def test_determine_outcome_ambiguous(self):
        assert _determine_outcome(["0.50", "0.50"], closed=True) is None
