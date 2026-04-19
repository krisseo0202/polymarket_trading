"""Unit tests for ProbEdgeStrategy.

Tests verify the core decision rule:
    BUY YES if edge_yes >= min_edge
    BUY NO  if edge_no  >= min_edge
    otherwise: NO TRADE

and the exit rules:
    stop_loss, time_limit, edge_reprice
"""

import time
from typing import Any, Mapping

import pytest

from src.api.types import OrderBook, OrderBookEntry, Position
from src.models.prediction import PredictionResult
from src.strategies.prob_edge import ProbEdgeStrategy
from src.utils.market_utils import spread_pct as _spread_pct


# ── Helpers ───────────────────────────────────────────────────────────────────

def _book(bid: float, ask: float, tick: float = 0.001) -> OrderBook:
    """Minimal OrderBook with one bid/ask level."""
    return OrderBook(
        market_id="mkt",
        token_id="tok",
        bids=[OrderBookEntry(price=bid, size=100.0)],
        asks=[OrderBookEntry(price=ask, size=100.0)],
        tick_size=tick,
    )


def _prediction(prob_yes: float, yes_ask: float, no_ask: float) -> PredictionResult:
    edge_yes = prob_yes - yes_ask
    prob_no  = 1.0 - prob_yes
    edge_no  = prob_no - no_ask
    return PredictionResult(
        prob_yes=prob_yes,
        model_version="test_v1",
        feature_status="ready",
        edge_yes=edge_yes,
        edge_no=edge_no,
    )


class _MockModel:
    """Stub model that returns a fixed prediction regardless of snapshot."""

    def __init__(self, prob_yes: float, yes_ask: float, no_ask: float):
        self._result = _prediction(prob_yes, yes_ask, no_ask)
        self.model_version = "mock_v1"

    def predict(self, snapshot: Mapping[str, Any]) -> PredictionResult:
        return self._result


class _NullProbModel:
    """Stub model that simulates unavailable probability output."""

    model_version = "null_prob_v1"

    def predict(self, snapshot: Mapping[str, Any]) -> PredictionResult:
        return PredictionResult(
            prob_yes=None,
            model_version=self.model_version,
            feature_status="insufficient_btc_history",
            edge_yes=None,
            edge_no=None,
        )


class _MockBtcFeed:
    """Healthy BTC feed stub with deterministic recent prices."""

    def __init__(self, prices):
        self._prices = prices

    def is_healthy(self) -> bool:
        return True

    def get_recent_prices(self, window: int = 300):
        return self._prices


CFG_BASE = {
    "min_edge": 0.04,
    "max_spread_pct": 0.20,
    "min_seconds_to_expiry": 10,
    "max_seconds_to_expiry": 290,
    "exit_edge": -0.02,
    "stop_loss_pct": 0.15,
    "max_hold_seconds": 270,
    "kelly_fraction": 0.20,
    "position_size_usdc": 50.0,
    "min_confidence": 0.51,
    "min_entry_price": 0.05,
    "max_entry_price": 0.95,
}

YES_TID = "yes_token"
NO_TID  = "no_token"
MKT_ID  = "market_001"


def _make_strategy(prob_yes: float, yes_ask: float, no_ask: float, **cfg_overrides) -> ProbEdgeStrategy:
    cfg = {**CFG_BASE, **cfg_overrides}
    strat = ProbEdgeStrategy(
        config=cfg,
        btc_feed=None,
        model_service=_MockModel(prob_yes, yes_ask, no_ask),
    )
    strat.set_tokens(MKT_ID, YES_TID, NO_TID)
    return strat


def _market_data(
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    positions=None,
    balance: float = 1000.0,
) -> dict:
    yes_b = _book(yes_bid, yes_ask)
    yes_b.token_id = YES_TID
    no_b  = _book(no_bid,  no_ask)
    no_b.token_id  = NO_TID
    return {
        "order_books": {YES_TID: yes_b, NO_TID: no_b},
        "positions": positions or [],
        "balance": balance,
        "question": "BTC up or down?",
        "strike_price": 50000.0,
        "slot_expiry_ts": time.time() + 120,
    }


# ── Entry tests ───────────────────────────────────────────────────────────────

class TestEntry:
    def test_buy_yes_when_edge_yes_clears_threshold(self):
        """p_up=0.62, yes_ask=0.55 → edge_yes=0.07 >= 0.04 → BUY YES."""
        strat = _make_strategy(prob_yes=0.62, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        sigs = strat.analyze(data)
        assert len(sigs) == 1
        assert sigs[0].action == "BUY"
        assert sigs[0].outcome == "YES"
        assert sigs[0].price == pytest.approx(0.55, abs=0.001)

    def test_buy_no_when_edge_no_clears_threshold(self):
        """p_up=0.38, no_ask=0.55 → edge_no=(0.62-0.55)=0.07 → BUY NO."""
        strat = _make_strategy(prob_yes=0.38, yes_ask=0.48, no_ask=0.55)
        data = _market_data(yes_bid=0.47, yes_ask=0.48, no_bid=0.54, no_ask=0.55)
        sigs = strat.analyze(data)
        assert len(sigs) == 1
        assert sigs[0].action == "BUY"
        assert sigs[0].outcome == "NO"

    def test_no_trade_when_neither_side_clears_threshold(self):
        """p_up=0.52, yes_ask=0.50 → edge=0.02 < 0.04 → NO TRADE."""
        strat = _make_strategy(prob_yes=0.52, yes_ask=0.50, no_ask=0.50)
        data = _market_data(yes_bid=0.49, yes_ask=0.50, no_bid=0.49, no_ask=0.50)
        sigs = strat.analyze(data)
        assert sigs == []

    def test_no_trade_when_fee_erases_positive_edge(self):
        """Raw edge can be positive while net edge falls below threshold after fees."""
        strat = _make_strategy(
            prob_yes=0.62,
            yes_ask=0.58,
            no_ask=0.42,
            fee_enabled=True,
            taker_fee_rate=0.072,
            taker_fee_exponent=1.0,
            min_edge=0.03,
        )
        data = _market_data(yes_bid=0.57, yes_ask=0.58, no_bid=0.41, no_ask=0.42)
        sigs = strat.analyze(data)
        assert sigs == []
        assert strat.last_edge_yes == pytest.approx(0.04, abs=0.001)
        assert strat.last_net_edge_yes < 0.03

    def test_picks_higher_edge_when_both_sides_qualify(self):
        """Both sides clear min_edge but YES has higher edge — should pick YES."""
        # edge_yes = 0.62 - 0.55 = 0.07
        # edge_no  = 0.38 - 0.30 = 0.08  ← higher
        strat = _make_strategy(prob_yes=0.62, yes_ask=0.55, no_ask=0.30)
        # This is unusual (prob_yes > 0.5 but no_ask is very low) but tests the rule
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.29, no_ask=0.30)
        sigs = strat.analyze(data)
        assert len(sigs) == 1
        # edge_no = (1-0.62) - 0.30 = 0.38 - 0.30 = 0.08 > edge_yes = 0.07
        assert sigs[0].outcome == "NO"

    def test_no_trade_when_spread_too_wide(self):
        """Wide spread should filter out the candidate."""
        strat = _make_strategy(
            prob_yes=0.70, yes_ask=0.55, no_ask=0.48,
            max_spread_pct=0.05,  # tight spread filter
        )
        # yes spread = (0.70 - 0.30) / 0.50 = 80% → filtered
        data = _market_data(yes_bid=0.30, yes_ask=0.70, no_bid=0.47, no_ask=0.48)
        sigs = strat.analyze(data)
        # YES filtered (wide), NO: edge_no = (1-0.70) - 0.48 = -0.18 < 0 → no trade
        assert sigs == []

    def test_no_trade_when_already_in_position(self):
        """Should not enter again when already holding a position."""
        strat = _make_strategy(prob_yes=0.70, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        sigs1 = strat.analyze(data)
        assert len(sigs1) == 1  # first entry
        sigs2 = strat.analyze(data)
        assert sigs2 == []  # already in position

    def test_no_trade_past_expiry_window(self):
        """Suppress entry when time-to-expiry is outside the allowed window."""
        strat = _make_strategy(prob_yes=0.70, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        # TTE < min_seconds_to_expiry
        data["slot_expiry_ts"] = time.time() + 5  # 5s left < 10s min
        sigs = strat.analyze(data)
        assert sigs == []


# ── Kelly sizing ──────────────────────────────────────────────────────────────

class TestKellySizing:
    def test_kelly_size_scales_with_edge(self):
        """Larger edge → larger Kelly fraction → larger size."""
        strat = _make_strategy(prob_yes=0.62, yes_ask=0.55, no_ask=0.48)
        # f_full = (0.62 - 0.55) / (1 - 0.55) = 0.07 / 0.45 ≈ 0.1556
        # f_used = 0.20 * 0.1556 = 0.0311
        # usdc = 0.0311 * 1000 = 31.1 → capped at position_size_usdc=50
        size_usdc = strat._kelly_size(0.62, 0.55, 1000.0)
        assert 0 < size_usdc <= 50.0
        assert size_usdc == pytest.approx(0.20 * (0.07 / 0.45) * 1000.0, rel=0.01)

    def test_kelly_size_capped_at_position_size_usdc(self):
        """Very large balance should be capped at position_size_usdc."""
        strat = _make_strategy(prob_yes=0.90, yes_ask=0.55, no_ask=0.48)
        size = strat._kelly_size(0.90, 0.55, 1_000_000.0)
        assert size == pytest.approx(50.0)

    def test_kelly_size_zero_when_no_edge(self):
        """prob <= price → Kelly is 0 → no trade."""
        strat = _make_strategy(prob_yes=0.50, yes_ask=0.55, no_ask=0.48)
        assert strat._kelly_size(0.50, 0.55, 1000.0) == 0.0

    def test_kelly_size_uses_position_size_when_no_balance(self):
        """Balance=0 falls back to position_size_usdc as the cap anchor."""
        strat = _make_strategy(prob_yes=0.62, yes_ask=0.55, no_ask=0.48)
        size = strat._kelly_size(0.62, 0.55, 0.0)
        assert 0 < size <= 50.0


# ── Exit tests ────────────────────────────────────────────────────────────────

class TestExit:
    def _enter(self, strat: ProbEdgeStrategy, yes_bid: float = 0.54, yes_ask: float = 0.55):
        """Helper: enter a YES position via analyze()."""
        data = _market_data(yes_bid=yes_bid, yes_ask=yes_ask, no_bid=0.44, no_ask=0.45)
        sigs = strat.analyze(data)
        assert sigs[0].action == "BUY"
        return sigs

    def test_stop_loss_exit(self):
        """Position drops 20% → stop-loss triggered (threshold=15%)."""
        strat = _make_strategy(prob_yes=0.70, yes_ask=0.55, no_ask=0.45)
        self._enter(strat, yes_ask=0.55)
        entry_price = strat.entry_price  # 0.55
        bid_now = entry_price * 0.80    # 20% loss
        data = _market_data(yes_bid=bid_now, yes_ask=bid_now + 0.01, no_bid=0.44, no_ask=0.45)
        sigs = strat.analyze(data)
        assert len(sigs) == 1
        assert sigs[0].action == "SELL"
        assert "stop_loss" in sigs[0].reason

    def test_time_limit_exit(self):
        """Holding too long triggers time_limit exit."""
        strat = _make_strategy(prob_yes=0.70, yes_ask=0.55, no_ask=0.45, max_hold_seconds=1)
        self._enter(strat, yes_ask=0.55)
        # Back-date entry timestamp
        strat.entry_timestamp = time.monotonic() - 5  # held for 5s > 1s limit
        data = _market_data(yes_bid=0.55, yes_ask=0.56, no_bid=0.44, no_ask=0.45)
        sigs = strat.analyze(data)
        assert len(sigs) == 1
        assert sigs[0].action == "SELL"
        assert "time_limit" in sigs[0].reason

    def test_edge_reprice_exit(self):
        """Edge of held position drops below exit_edge → edge_reprice exit."""
        # Enter YES at 0.55 with prob_yes=0.70
        strat = _make_strategy(
            prob_yes=0.70, yes_ask=0.55, no_ask=0.45, exit_edge=-0.02
        )
        self._enter(strat, yes_ask=0.55)
        # Now model reprices: prob_yes=0.50, bid=0.55 → held_edge = 0.50 - 0.55 = -0.05 < -0.02
        # Use a new model that returns prob_yes=0.50
        strat.model_service = _MockModel(prob_yes=0.50, yes_ask=0.56, no_ask=0.45)
        data = _market_data(yes_bid=0.55, yes_ask=0.56, no_bid=0.44, no_ask=0.45)
        sigs = strat.analyze(data)
        assert len(sigs) == 1
        assert sigs[0].action == "SELL"
        assert "edge_reprice" in sigs[0].reason

    def test_no_exit_when_position_profitable_and_edge_intact(self):
        """No exit when stop/time/edge-reprice conditions are all met (hold)."""
        strat = _make_strategy(prob_yes=0.70, yes_ask=0.55, no_ask=0.45, exit_edge=-0.05)
        self._enter(strat, yes_ask=0.55)
        # Current bid = 0.60 (profit), held_edge = 0.70 - 0.60 = +0.10 > -0.05
        data = _market_data(yes_bid=0.60, yes_ask=0.61, no_bid=0.39, no_ask=0.40)
        sigs = strat.analyze(data)
        assert sigs == []  # no exit

    def test_position_auto_recovered_after_restart(self):
        """If bot restarts with open position in positions list, state recovers."""
        strat = _make_strategy(prob_yes=0.52, yes_ask=0.50, no_ask=0.50)
        # No position state in strategy (fresh restart), but positions list has YES
        pos = Position(
            market_id=MKT_ID,
            token_id=YES_TID,
            outcome="YES",
            size=50.0,
            average_price=0.50,
        )
        data = _market_data(
            yes_bid=0.49, yes_ask=0.50, no_bid=0.49, no_ask=0.50,
            positions=[pos],
        )
        sigs = strat.analyze(data)
        assert strat.active_token_id == YES_TID
        assert strat.entry_price == pytest.approx(0.50)
        assert sigs == []  # no new entry or exit (edge < min_edge, no exit trigger)


# ── Observable state ──────────────────────────────────────────────────────────

class TestObservableState:
    def test_last_prob_yes_updated_after_analyze(self):
        strat = _make_strategy(prob_yes=0.65, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        strat.analyze(data)
        assert strat.last_prob_yes == pytest.approx(0.65)
        assert strat.last_prob_no == pytest.approx(0.35)

    def test_last_edge_yes_and_no_updated(self):
        strat = _make_strategy(prob_yes=0.65, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        strat.analyze(data)
        assert strat.last_edge_yes == pytest.approx(0.65 - 0.55, abs=0.001)
        assert strat.last_edge_no  == pytest.approx(0.35 - 0.48, abs=0.001)

    def test_last_net_edge_matches_raw_edge_when_fees_disabled(self):
        strat = _make_strategy(prob_yes=0.65, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        strat.analyze(data)
        assert strat.last_net_edge_yes == pytest.approx(strat.last_edge_yes, abs=0.001)

    def test_probability_context_observables_updated(self):
        strat = _make_strategy(prob_yes=0.65, yes_ask=0.55, no_ask=0.48)
        now = time.time()
        strat.btc_feed = _MockBtcFeed([
            (now - 60, 49_900.0, 1.0),
            (now, 50_100.0, 1.0),
        ])
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        strat.analyze(data)
        assert strat.last_expected_fill_yes == pytest.approx(0.55, abs=0.001)
        assert strat.last_expected_fill_no is None
        assert strat.last_tte_seconds is not None
        assert strat.last_tte_seconds > 0
        assert strat.last_distance_to_break_pct == pytest.approx((50_100.0 - 50_000.0) / 50_000.0, abs=1e-6)
        assert strat.last_distance_to_strike_bps == pytest.approx(20.0, abs=0.1)

    def test_last_feature_status_propagated(self):
        strat = _make_strategy(prob_yes=0.65, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        strat.analyze(data)
        assert strat.last_feature_status == "ready"

    def test_missing_prob_clears_probability_observables(self):
        strat = _make_strategy(prob_yes=0.65, yes_ask=0.55, no_ask=0.48)
        data = _market_data(yes_bid=0.54, yes_ask=0.55, no_bid=0.47, no_ask=0.48)
        strat.analyze(data)
        strat.model_service = _NullProbModel()
        sigs = strat.analyze(data)
        assert sigs == []
        assert strat.last_prob_yes is None
        assert strat.last_prob_no is None
        assert strat.last_edge_yes is None
        assert strat.last_edge_no is None
        assert strat.last_net_edge_yes is None
        assert strat.last_net_edge_no is None


# ── Dynamic edge threshold ────────────────────────────────────────────────────

class TestRequiredEdge:
    def test_required_edge_higher_near_expiry(self):
        """Near expiry the required edge must be HIGHER, not lower.

        Books thin and gamma is high near expiry — the strategy should demand more
        confirmation, not less.  This is a regression guard for the direction of the
        dynamic threshold formula.
        """
        strat = _make_strategy(prob_yes=0.60, yes_ask=0.55, no_ask=0.44)
        max_tte = strat.max_seconds_to_expiry
        edge_at_zero = strat._required_edge(0.0)
        edge_at_max = strat._required_edge(max_tte)
        assert edge_at_zero > edge_at_max, (
            f"required_edge at tte=0 ({edge_at_zero:.4f}) should exceed "
            f"required_edge at tte={max_tte}s ({edge_at_max:.4f})"
        )
        # Sanity: both should be at least min_edge
        assert edge_at_zero >= strat.min_edge
        assert edge_at_max >= strat.min_edge


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_spread_pct_symmetric(self):
        assert _spread_pct(0.48, 0.52) == pytest.approx(0.04 / 0.50)

    def test_spread_pct_zero_mid(self):
        assert _spread_pct(0.0, 0.0) == float("inf")
