"""Markov transition matrix + Monte Carlo simulator for Polymarket prices.

Discretizes BTC prices into N states, builds a transition matrix from
observed historical transitions, then simulates forward paths to estimate
P(YES) = P(BTC at expiry >= strike).

The model implements the standard predict(snapshot) -> PredictionResult
interface so it can plug directly into LogRegEdgeStrategy.

Usage standalone:

    from src.models.markov_model import (
        discretize, build_transition_matrix, simulate_paths,
        estimate_p_yes, MarkovModel,
    )

    # From raw prices
    states, edges = discretize(prices, n_states=50)
    T = build_transition_matrix(states, n_states=50, smoothing=1e-6)
    paths = simulate_paths(T, start_state=states[-1], n_steps=60, n_paths=5000)
    p_up = estimate_p_yes(paths, edges, strike=prices[-1])

    # Or via the model object (live-compatible)
    model = MarkovModel(n_states=50, n_paths=5000, smoothing=1e-6)
    result = model.predict(snapshot)
    print(result.prob_yes, result.feature_status)
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

import numpy as np

from .xgb_model import PredictionResult
from .feature_builder import parse_strike_price


# ---------------------------------------------------------------------------
# Core functions (stateless, testable)
# ---------------------------------------------------------------------------

def discretize(
    prices: np.ndarray,
    n_states: int = 50,
    edges: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map continuous prices into integer state indices in [0, n_states).

    Args:
        prices: 1-D array of price observations.
        n_states: Number of discrete bins.
        edges: Pre-computed bin edges (n_states + 1,). If None, computed
               from the data using equal-width bins spanning
               [min - epsilon, max + epsilon].

    Returns:
        (states, edges) — integer state array and the bin edges used.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        raise ValueError("prices array is empty")

    if edges is None:
        lo, hi = float(prices.min()), float(prices.max())
        if lo == hi:
            # All prices identical — center a tiny range around them so
            # digitize still produces valid indices.
            lo -= 0.5
            hi += 0.5
        eps = (hi - lo) * 1e-9
        edges = np.linspace(lo - eps, hi + eps, n_states + 1)

    # np.digitize returns 1-based bin indices; shift to 0-based and clamp.
    states = np.clip(np.digitize(prices, edges) - 1, 0, n_states - 1)
    return states, edges


def build_transition_matrix(
    states: np.ndarray,
    n_states: int,
    smoothing: float = 1e-6,
) -> np.ndarray:
    """Build a row-stochastic Markov transition matrix from observed state transitions.

    Args:
        states: 1-D integer array of state indices (from discretize).
        n_states: Total number of states (matrix is n_states x n_states).
        smoothing: Laplace-style additive smoothing applied to every cell
                   before normalization. Prevents zero-probability
                   transitions and stabilizes rows with few observations.

    Returns:
        T — (n_states, n_states) row-stochastic matrix where T[i, j] is
        the probability of transitioning from state i to state j.

    Raises:
        ValueError: If states has fewer than 2 observations (need at least
        one transition).
    """
    states = np.asarray(states, dtype=int)
    if states.size < 2:
        raise ValueError("Need at least 2 observations to build transitions")

    T = np.full((n_states, n_states), smoothing, dtype=float)

    # Count observed transitions.
    for i in range(len(states) - 1):
        T[states[i], states[i + 1]] += 1.0

    # Normalize each row to sum to 1.
    row_sums = T.sum(axis=1, keepdims=True)
    # Guard against rows that are all-zero (shouldn't happen with smoothing > 0,
    # but defend anyway: replace with uniform).
    zero_rows = (row_sums < 1e-15).ravel()
    if zero_rows.any():
        T[zero_rows] = 1.0 / n_states
        row_sums[zero_rows] = 1.0
    T /= row_sums

    return T


def validate_transition_matrix(T: np.ndarray, tol: float = 1e-6) -> list[str]:
    """Run sanity checks on a transition matrix. Returns a list of warnings (empty = OK).

    Checks:
        1. Square matrix
        2. All entries non-negative
        3. Row sums within tol of 1.0
        4. No all-zero rows (degenerate states)
    """
    warnings = []
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        warnings.append(f"Matrix is not square: shape={T.shape}")
        return warnings

    if (T < -tol).any():
        n_neg = int((T < -tol).sum())
        warnings.append(f"{n_neg} negative entries (min={T.min():.2e})")

    row_sums = T.sum(axis=1)
    bad_rows = np.abs(row_sums - 1.0) > tol
    if bad_rows.any():
        worst = float(np.max(np.abs(row_sums - 1.0)))
        warnings.append(
            f"{int(bad_rows.sum())} rows deviate from sum=1 "
            f"(worst={worst:.2e})"
        )

    zero_rows = (T.sum(axis=1) < 1e-15)
    if zero_rows.any():
        warnings.append(f"{int(zero_rows.sum())} all-zero rows (empty states)")

    return warnings


def simulate_paths(
    T: np.ndarray,
    start_state: int,
    n_steps: int,
    n_paths: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Forward-simulate Markov chain paths from a given starting state.

    Args:
        T: (n_states, n_states) row-stochastic transition matrix.
        start_state: Integer state index to start each path from.
        n_steps: Number of transitions to simulate per path.
        n_paths: Number of independent paths.
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        (n_paths, n_steps + 1) integer array of state indices.
        Column 0 is always start_state.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_states = T.shape[0]
    start_state = int(np.clip(start_state, 0, n_states - 1))

    paths = np.empty((n_paths, n_steps + 1), dtype=int)
    paths[:, 0] = start_state

    # Vectorized: at each step, all paths advance simultaneously.
    # Pre-compute cumulative probabilities for efficient sampling.
    cumprobs = np.cumsum(T, axis=1)

    for t in range(n_steps):
        current = paths[:, t]                        # (n_paths,)
        u = rng.random(n_paths)                      # (n_paths,) uniform [0, 1)
        # For each path, find the next state via the CDF of its current row.
        row_cdf = cumprobs[current]                  # (n_paths, n_states)
        paths[:, t + 1] = (u[:, None] < row_cdf).argmax(axis=1)

    return paths


def estimate_p_yes(
    paths: np.ndarray,
    edges: np.ndarray,
    strike: float,
) -> float:
    """Estimate P(YES) = P(final price >= strike) from simulated paths.

    Maps terminal state indices back to price midpoints, then counts what
    fraction land at or above the strike.
    """
    # Bin midpoints: midpoint of each [edges[i], edges[i+1]) interval.
    midpoints = 0.5 * (edges[:-1] + edges[1:])

    terminal_states = paths[:, -1]
    # Clamp to valid midpoint indices.
    terminal_states = np.clip(terminal_states, 0, len(midpoints) - 1)
    terminal_prices = midpoints[terminal_states]

    return float((terminal_prices >= strike).mean())


# ---------------------------------------------------------------------------
# Model class — predict(snapshot) -> PredictionResult interface
# ---------------------------------------------------------------------------

class MarkovModel:
    """Markov chain Monte Carlo model for BTC 5-minute Up/Down markets.

    Fits a transition matrix on the trailing BTC price history available in
    the snapshot, then simulates forward paths to the slot expiry.

    Parameters:
        n_states: Number of discrete price bins (higher = finer resolution
                  but sparser transitions; 50 is a good default for ~300
                  1s observations in a 5-minute window).
        n_paths: Number of Monte Carlo forward paths.
        smoothing: Laplace smoothing constant for the transition matrix.
        lookback_s: How many seconds of BTC history to use for fitting
                    (default 600 = 10 minutes).
        seed: Random seed for reproducibility (None = non-deterministic).
    """

    MODEL_VERSION = "markov_mc_v1"

    def __init__(
        self,
        n_states: int = 50,
        n_paths: int = 5000,
        smoothing: float = 1e-6,
        lookback_s: float = 600.0,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.n_states = n_states
        self.n_paths = n_paths
        self.smoothing = smoothing
        self.lookback_s = lookback_s
        self._rng = np.random.default_rng(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.ready = True

        # Populated after each predict() for observability / logging.
        self.last_features: dict = {}

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        """Estimate P(YES) using Markov chain Monte Carlo simulation.

        Snapshot keys consumed:
            btc_prices: list of (ts, price[, volume]) tuples
            strike_price: float (or extracted from 'question')
            slot_expiry_ts: float — Unix timestamp of slot close
            now_ts: float (optional)
        """
        btc_prices: list = list(snapshot.get("btc_prices") or [])
        if len(btc_prices) < 15:
            return PredictionResult(None, self.MODEL_VERSION, "insufficient_btc_history")

        now_ts = float(snapshot.get("now_ts") or btc_prices[-1][0])
        btc_mid = float(btc_prices[-1][1])
        if btc_mid <= 0:
            return PredictionResult(None, self.MODEL_VERSION, "invalid_btc_price")

        # Strike
        strike_price = snapshot.get("strike_price")
        if strike_price is None:
            question = str(snapshot.get("question") or "")
            strike_price = parse_strike_price(question)
        if strike_price is None or float(strike_price) <= 0:
            return PredictionResult(None, self.MODEL_VERSION, "missing_strike")
        strike_price = float(strike_price)

        # Time to expiry
        slot_expiry_ts = snapshot.get("slot_expiry_ts")
        if slot_expiry_ts is None:
            return PredictionResult(None, self.MODEL_VERSION, "missing_expiry")
        tte = max(0.0, float(slot_expiry_ts) - now_ts)

        # Extract price series within lookback window.
        ts_arr = np.array([float(p[0]) for p in btc_prices])
        px_arr = np.array([float(p[1]) for p in btc_prices])
        mask = ts_arr >= (now_ts - self.lookback_s)
        px_window = px_arr[mask]
        if len(px_window) < 10:
            return PredictionResult(None, self.MODEL_VERSION, "insufficient_window")

        try:
            # Discretize
            states, edges = discretize(px_window, n_states=self.n_states)

            # Build transition matrix
            T = build_transition_matrix(states, self.n_states, self.smoothing)

            # Validate
            warnings = validate_transition_matrix(T)
            if warnings:
                for w in warnings:
                    self.logger.warning("Markov T matrix: %s", w)

            # Number of forward steps ≈ seconds remaining (1s per BTC tick).
            n_steps = max(1, int(round(tte)))

            # Simulate
            paths = simulate_paths(
                T, start_state=states[-1],
                n_steps=n_steps, n_paths=self.n_paths,
                rng=self._rng,
            )

            # Estimate P(YES)
            p_yes = estimate_p_yes(paths, edges, strike_price)

            # Populate observability features.
            midpoints = 0.5 * (edges[:-1] + edges[1:])
            terminal_prices = midpoints[np.clip(paths[:, -1], 0, len(midpoints) - 1)]
            self.last_features = {
                "n_observations": int(len(px_window)),
                "n_states_occupied": int(len(np.unique(states))),
                "n_steps": n_steps,
                "n_paths": self.n_paths,
                "smoothing": self.smoothing,
                "strike": strike_price,
                "current_price": btc_mid,
                "p_yes": p_yes,
                "terminal_mean": float(terminal_prices.mean()),
                "terminal_std": float(terminal_prices.std()),
                "terminal_median": float(np.median(terminal_prices)),
            }

            p_yes = float(np.clip(p_yes, 0.01, 0.99))
            return PredictionResult(p_yes, self.MODEL_VERSION, "ready")

        except Exception as exc:
            self.logger.warning("Markov predict failed: %s", exc)
            return PredictionResult(None, self.MODEL_VERSION, "predict_failed")


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_model_vs_market(
    p_model: float,
    market_yes_price: float,
    market_no_price: Optional[float] = None,
) -> dict:
    """Compare model probability against market-implied probability.

    Args:
        p_model: Model's P(YES) estimate.
        market_yes_price: Best ask (or mid) on the YES token.
        market_no_price: Best ask on the NO token (optional; derived from
                         YES if absent).

    Returns:
        Dict with edge analysis:
            market_implied_yes — mid-market P(YES)
            edge_yes — p_model - market_yes_price
            edge_no — (1 - p_model) - market_no_price
            recommended_side — "YES", "NO", or "SKIP"
    """
    if market_no_price is None:
        market_no_price = max(0.01, 1.0 - market_yes_price)

    market_implied = 0.5 * (market_yes_price + (1.0 - market_no_price))
    edge_yes = p_model - market_yes_price
    edge_no = (1.0 - p_model) - market_no_price

    if edge_yes > 0.02 and edge_yes >= edge_no:
        side = "YES"
    elif edge_no > 0.02:
        side = "NO"
    else:
        side = "SKIP"

    return {
        "p_model": round(p_model, 4),
        "market_implied_yes": round(market_implied, 4),
        "market_yes_price": round(market_yes_price, 4),
        "market_no_price": round(market_no_price, 4),
        "edge_yes": round(edge_yes, 4),
        "edge_no": round(edge_no, 4),
        "recommended_side": side,
    }
