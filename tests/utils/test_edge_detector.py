"""Tests for detect_edge() — pure function, no mocking needed."""

import pytest

from src.utils.edge_detector import detect_edge, EdgeDecision


class TestBuyUpDetected:
    def test_up_edge_above_threshold(self):
        d = detect_edge(bot_up_prob=0.65, market_up_odds=0.55, market_down_odds=0.45)
        assert d.side == "BUY_UP"
        assert d.edge == pytest.approx(0.10, abs=1e-9)

    def test_up_edge_stored_in_up_edge_field(self):
        d = detect_edge(bot_up_prob=0.65, market_up_odds=0.55, market_down_odds=0.45)
        assert d.up_edge == pytest.approx(0.10, abs=1e-9)

    def test_kelly_positive_for_buy_up(self):
        d = detect_edge(bot_up_prob=0.65, market_up_odds=0.55, market_down_odds=0.45)
        assert d.kelly_fraction > 0.0


class TestBuyDownDetected:
    def test_down_edge_above_threshold(self):
        # bot says 35% up → 65% down; market DOWN at 0.45 → down_edge = 0.65-0.45=0.20
        d = detect_edge(bot_up_prob=0.35, market_up_odds=0.55, market_down_odds=0.45)
        assert d.side == "BUY_DOWN"
        assert d.edge == pytest.approx(0.20, abs=1e-9)

    def test_kelly_positive_for_buy_down(self):
        d = detect_edge(bot_up_prob=0.35, market_up_odds=0.55, market_down_odds=0.45)
        assert d.kelly_fraction > 0.0


class TestNoTrade:
    def test_both_edges_below_threshold(self):
        d = detect_edge(bot_up_prob=0.52, market_up_odds=0.55, market_down_odds=0.45)
        assert d.side == "NO_TRADE"

    def test_kelly_zero_for_no_trade(self):
        d = detect_edge(bot_up_prob=0.52, market_up_odds=0.55, market_down_odds=0.45)
        assert d.kelly_fraction == 0.0

    def test_edge_zero_for_no_trade(self):
        d = detect_edge(bot_up_prob=0.52, market_up_odds=0.55, market_down_odds=0.45)
        assert d.edge == 0.0

    def test_raw_edges_still_reported_for_no_trade(self):
        d = detect_edge(bot_up_prob=0.52, market_up_odds=0.55, market_down_odds=0.45)
        # up_edge = 0.52-0.55 = -0.03, down_edge = 0.48-0.45 = 0.03 (exactly at threshold, not above)
        assert d.up_edge == pytest.approx(-0.03, abs=1e-9)


class TestDualEdgePicksLarger:
    def test_both_positive_up_larger(self):
        # up_edge=0.10, down_edge=0.05 → pick UP
        d = detect_edge(bot_up_prob=0.65, market_up_odds=0.55, market_down_odds=0.60)
        assert d.side == "BUY_UP"

    def test_both_positive_down_larger(self):
        # down_edge > up_edge
        # bot=0.55, up_market=0.52 → up_edge=0.03; down market=0.38 → down_edge=0.45-0.38=0.07
        d = detect_edge(bot_up_prob=0.55, market_up_odds=0.52, market_down_odds=0.38)
        assert d.side == "BUY_DOWN"
        assert d.edge > 0.03  # down edge larger


class TestKellyCap:
    def test_kelly_capped_at_max_kelly(self):
        # Extreme edge — Kelly would exceed cap
        d = detect_edge(
            bot_up_prob=0.99,
            market_up_odds=0.50,
            market_down_odds=0.50,
            max_kelly=0.05,
        )
        assert d.side == "BUY_UP"
        assert d.kelly_fraction == pytest.approx(0.05)


class TestLiquidityGuard:
    def test_thin_book_returns_no_trade(self):
        # sum = 0.80 < 0.85 threshold
        d = detect_edge(bot_up_prob=0.70, market_up_odds=0.50, market_down_odds=0.30)
        assert d.side == "NO_TRADE"
        assert d.kelly_fraction == 0.0
        assert d.skip_reason is not None
        assert "thin" in d.skip_reason.lower()

    def test_exactly_at_threshold_proceeds(self):
        # sum = 0.85 exactly → should NOT be blocked
        d = detect_edge(bot_up_prob=0.70, market_up_odds=0.50, market_down_odds=0.35)
        # 0.50 + 0.35 = 0.85 → not thin
        assert d.side == "BUY_UP"  # edge = 0.70-0.50 = 0.20

    def test_skip_reason_none_for_normal_no_trade(self):
        # Below edge threshold but NOT thin book
        d = detect_edge(bot_up_prob=0.52, market_up_odds=0.55, market_down_odds=0.45)
        assert d.skip_reason is None


class TestFieldCompleteness:
    def test_all_fields_populated(self):
        d = detect_edge(bot_up_prob=0.65, market_up_odds=0.55, market_down_odds=0.45)
        assert isinstance(d, EdgeDecision)
        assert d.bot_up_prob == 0.65
        assert d.market_up_odds == 0.55
        assert d.market_down_odds == 0.45
        assert d.min_edge_threshold == 0.03

    def test_custom_min_edge(self):
        # With min_edge=0.15, edge of 0.10 should NOT signal
        d = detect_edge(
            bot_up_prob=0.65,
            market_up_odds=0.55,
            market_down_odds=0.45,
            min_edge=0.15,
        )
        assert d.side == "NO_TRADE"
