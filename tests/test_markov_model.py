"""Tests for the Markov transition matrix + Monte Carlo model.

Covers: discretize, build_transition_matrix, validate_transition_matrix,
simulate_paths, estimate_p_yes, MarkovModel.predict, compare_model_vs_market.
"""

import numpy as np
import pytest

from src.models.markov_model import (
    build_transition_matrix,
    compare_model_vs_market,
    discretize,
    estimate_p_yes,
    MarkovModel,
    simulate_paths,
    validate_transition_matrix,
)


# ---------------------------------------------------------------------------
# discretize
# ---------------------------------------------------------------------------

class TestDiscretize:
    def test_basic_binning(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        states, edges = discretize(prices, n_states=5)
        assert states.shape == (5,)
        assert len(edges) == 6  # n_states + 1
        assert states.min() >= 0
        assert states.max() < 5

    def test_monotonic_prices_produce_ordered_states(self):
        prices = np.linspace(10, 20, 100)
        states, _ = discretize(prices, n_states=10)
        # First state should be lower than last.
        assert states[0] <= states[-1]

    def test_all_identical_prices(self):
        prices = np.full(50, 100.0)
        states, edges = discretize(prices, n_states=10)
        # Should not crash; all land in the same bin.
        assert len(np.unique(states)) == 1

    def test_custom_edges(self):
        prices = np.array([1.5, 2.5, 3.5])
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        states, returned_edges = discretize(prices, n_states=3, edges=edges)
        assert np.array_equal(returned_edges, edges)
        assert states[0] == 0  # 1.5 in [1, 2)
        assert states[1] == 1  # 2.5 in [2, 3)
        assert states[2] == 2  # 3.5 in [3, 4)

    def test_empty_prices_raises(self):
        with pytest.raises(ValueError, match="empty"):
            discretize(np.array([]), n_states=10)

    def test_single_price(self):
        states, edges = discretize(np.array([42.0]), n_states=5)
        assert states.shape == (1,)
        assert 0 <= states[0] < 5


# ---------------------------------------------------------------------------
# build_transition_matrix
# ---------------------------------------------------------------------------

class TestBuildTransitionMatrix:
    def test_rows_sum_to_one(self):
        states = np.array([0, 1, 2, 1, 0, 2, 2, 1])
        T = build_transition_matrix(states, n_states=3, smoothing=1e-6)
        row_sums = T.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_shape(self):
        T = build_transition_matrix(np.array([0, 1, 0]), n_states=5, smoothing=1e-6)
        assert T.shape == (5, 5)

    def test_smoothing_prevents_zeros(self):
        # Only transitions 0->1 and 1->0 observed.
        T = build_transition_matrix(np.array([0, 1, 0, 1]), n_states=3, smoothing=1e-6)
        # All entries should be > 0 thanks to smoothing.
        assert (T > 0).all()

    def test_no_smoothing_has_zeros(self):
        T = build_transition_matrix(np.array([0, 1, 0, 1]), n_states=3, smoothing=0.0)
        # State 2 was never visited, row 2 should be uniform (guard against
        # all-zero rows), but cells like T[0, 0] should be 0.
        assert T[0, 0] == 0.0  # never saw 0->0

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_transition_matrix(np.array([0]), n_states=3)

    def test_observed_transitions_dominate(self):
        # 100 transitions of 0->1, smoothing tiny.
        states = np.array([0, 1] * 50)
        T = build_transition_matrix(states, n_states=3, smoothing=1e-9)
        # T[0, 1] should be ~1.0 (heavily observed).
        assert T[0, 1] > 0.99


# ---------------------------------------------------------------------------
# validate_transition_matrix
# ---------------------------------------------------------------------------

class TestValidateTransitionMatrix:
    def test_valid_matrix_returns_empty(self):
        T = np.array([[0.5, 0.5], [0.3, 0.7]])
        assert validate_transition_matrix(T) == []

    def test_bad_row_sum_detected(self):
        T = np.array([[0.5, 0.3], [0.3, 0.7]])  # row 0 sums to 0.8
        warnings = validate_transition_matrix(T)
        assert any("rows deviate" in w for w in warnings)

    def test_negative_entries_detected(self):
        T = np.array([[0.5, 0.5], [-0.1, 1.1]])
        warnings = validate_transition_matrix(T)
        assert any("negative" in w for w in warnings)

    def test_non_square_detected(self):
        T = np.ones((2, 3))
        warnings = validate_transition_matrix(T)
        assert any("not square" in w for w in warnings)


# ---------------------------------------------------------------------------
# simulate_paths
# ---------------------------------------------------------------------------

class TestSimulatePaths:
    def test_shape(self):
        T = np.array([[0.5, 0.5], [0.5, 0.5]])
        paths = simulate_paths(T, start_state=0, n_steps=10, n_paths=100)
        assert paths.shape == (100, 11)  # n_paths x (n_steps + 1)

    def test_all_start_at_initial_state(self):
        T = np.eye(3)  # absorbing states
        paths = simulate_paths(T, start_state=2, n_steps=5, n_paths=50)
        assert (paths[:, 0] == 2).all()

    def test_absorbing_state_stays(self):
        T = np.eye(3)
        paths = simulate_paths(T, start_state=1, n_steps=20, n_paths=100)
        # All columns should be 1 (absorbing).
        assert (paths == 1).all()

    def test_deterministic_with_seed(self):
        T = np.array([[0.3, 0.7], [0.6, 0.4]])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = simulate_paths(T, start_state=0, n_steps=50, n_paths=200, rng=rng1)
        p2 = simulate_paths(T, start_state=0, n_steps=50, n_paths=200, rng=rng2)
        np.testing.assert_array_equal(p1, p2)

    def test_states_in_valid_range(self):
        T = np.ones((5, 5)) / 5
        paths = simulate_paths(T, start_state=0, n_steps=30, n_paths=500)
        assert paths.min() >= 0
        assert paths.max() < 5

    def test_start_state_clamped(self):
        T = np.eye(2)
        paths = simulate_paths(T, start_state=99, n_steps=1, n_paths=10)
        # Should clamp to max valid state (1).
        assert (paths[:, 0] == 1).all()


# ---------------------------------------------------------------------------
# estimate_p_yes
# ---------------------------------------------------------------------------

class TestEstimatePYes:
    def test_all_above_strike(self):
        edges = np.array([10.0, 20.0, 30.0])
        # 2 bins: midpoints 15.0, 25.0. Both >= strike 10.
        paths = np.array([[0, 1], [0, 0], [0, 1]])  # terminal: [1, 0, 1]
        p = estimate_p_yes(paths, edges, strike=10.0)
        assert p == 1.0  # all midpoints (15 and 25) >= 10

    def test_none_above_strike(self):
        edges = np.array([10.0, 20.0, 30.0])
        paths = np.array([[0, 0], [0, 0]])  # terminal: [0, 0], midpoint 15.0
        p = estimate_p_yes(paths, edges, strike=20.0)
        assert p == 0.0  # midpoint 15 < 20

    def test_mixed(self):
        edges = np.array([10.0, 20.0, 30.0])
        # terminal states: 0 (mid=15), 1 (mid=25), 0, 1
        paths = np.array([[0, 0], [0, 1], [0, 0], [0, 1]])
        p = estimate_p_yes(paths, edges, strike=20.0)
        assert p == 0.5  # 2/4 paths have terminal mid >= 20


# ---------------------------------------------------------------------------
# MarkovModel.predict (integration)
# ---------------------------------------------------------------------------

class TestMarkovModelPredict:
    def _make_snapshot(self, n=300, strike=100.0, tte=60.0):
        """Generate a synthetic snapshot with trending BTC prices."""
        now = 1700000000.0
        ts_list = [(now - n + i, 100.0 + 0.01 * i) for i in range(n)]
        return {
            "btc_prices": ts_list,
            "strike_price": strike,
            "slot_expiry_ts": now + tte,
            "now_ts": now,
        }

    def test_happy_path_returns_ready(self):
        model = MarkovModel(n_states=20, n_paths=1000, seed=42)
        result = model.predict(self._make_snapshot())
        assert result.feature_status == "ready"
        assert 0.01 <= result.prob_yes <= 0.99

    def test_insufficient_history(self):
        model = MarkovModel()
        snap = {"btc_prices": [(1, 100.0)] * 5, "strike_price": 100.0,
                "slot_expiry_ts": 9999999999.0}
        result = model.predict(snap)
        assert result.feature_status == "insufficient_btc_history"
        assert result.prob_yes is None

    def test_missing_strike(self):
        model = MarkovModel()
        snap = self._make_snapshot()
        del snap["strike_price"]
        snap["question"] = "no strike here"
        result = model.predict(snap)
        assert result.feature_status == "missing_strike"

    def test_missing_expiry(self):
        model = MarkovModel()
        snap = self._make_snapshot()
        del snap["slot_expiry_ts"]
        result = model.predict(snap)
        assert result.feature_status == "missing_expiry"

    def test_last_features_populated(self):
        model = MarkovModel(n_states=20, n_paths=500, seed=7)
        model.predict(self._make_snapshot())
        f = model.last_features
        assert "p_yes" in f
        assert "n_observations" in f
        assert f["n_paths"] == 500
        assert f["strike"] == 100.0

    def test_price_well_above_strike_favors_yes(self):
        """When current price is well above strike, most paths stay above."""
        model = MarkovModel(n_states=30, n_paths=3000, seed=123)
        now = 1700000000.0
        # Random walk centered around 100, strike at 99 (current is above).
        rng = np.random.default_rng(42)
        px = 100.0 + np.cumsum(rng.normal(0, 0.05, 300))
        ts_list = [(now - 300 + i, float(px[i])) for i in range(300)]
        snap = {
            "btc_prices": ts_list,
            "strike_price": float(px[-1]) - 2.0,  # strike well below current
            "slot_expiry_ts": now + 30,
            "now_ts": now,
        }
        result = model.predict(snap)
        assert result.prob_yes > 0.50

    def test_reproducible_with_seed(self):
        snap = self._make_snapshot()
        r1 = MarkovModel(n_states=20, n_paths=500, seed=99).predict(snap)
        r2 = MarkovModel(n_states=20, n_paths=500, seed=99).predict(snap)
        assert r1.prob_yes == r2.prob_yes


# ---------------------------------------------------------------------------
# compare_model_vs_market
# ---------------------------------------------------------------------------

class TestCompareModelVsMarket:
    def test_yes_edge(self):
        result = compare_model_vs_market(p_model=0.60, market_yes_price=0.50)
        assert result["edge_yes"] > 0
        assert result["recommended_side"] == "YES"

    def test_no_edge(self):
        result = compare_model_vs_market(p_model=0.35, market_yes_price=0.50)
        assert result["edge_no"] > 0
        assert result["recommended_side"] == "NO"

    def test_skip_when_no_edge(self):
        result = compare_model_vs_market(p_model=0.50, market_yes_price=0.50)
        assert result["recommended_side"] == "SKIP"

    def test_explicit_no_price(self):
        result = compare_model_vs_market(
            p_model=0.70, market_yes_price=0.55, market_no_price=0.50
        )
        assert result["market_no_price"] == 0.50
        assert result["edge_yes"] == pytest.approx(0.15, abs=0.01)
