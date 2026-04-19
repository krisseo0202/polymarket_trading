"""Markov transition matrix + Monte Carlo simulator for Polymarket prices.

Composite state S_t = (price_bin, tte_bin, return_bin, micro_bin) captures
price level, time-to-expiry, short-term momentum, and realized volatility.
The transition matrix is built from observed state transitions, then forward
paths are simulated to estimate P(YES) = P(BTC at expiry >= strike).

During simulation, time-to-expiry advances deterministically (decrements
each tick) while the other dimensions evolve stochastically via the
learned transition matrix.

The model implements the standard predict(snapshot) -> PredictionResult
interface so it can plug directly into LogRegEdgeStrategy.

Usage standalone:

    from src.models.markov_model import (
        discretize, compute_features, build_composite_states,
        build_transition_matrix, simulate_composite_paths,
        estimate_p_yes_composite, MarkovModel,
    )

    # Compute features and composite states
    features = compute_features(prices, timestamps, expiry_ts)
    states, edges, dims = build_composite_states(features)
    T = build_transition_matrix(states, n_states=np.prod(dims), smoothing=1e-4)

    # Simulate with TTE correction
    tte_schedule = ...  # TTE bin at each simulation step
    paths = simulate_composite_paths(T, states[-1], n_steps=60,
                                     dims=dims, tte_schedule=tte_schedule)
    p_up = estimate_p_yes_composite(paths, edges["price"], dims, strike)

    # Or via the model object (live-compatible)
    model = MarkovModel()
    result = model.predict(snapshot)
    print(result.prob_yes, result.feature_status)
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .prediction import PredictionResult
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
# Composite state functions (multi-dimensional Markov state)
# ---------------------------------------------------------------------------

def compute_features(
    prices: np.ndarray,
    timestamps: np.ndarray,
    expiry_ts: float,
    return_lookback: int = 5,
    vol_lookback: int = 15,
) -> dict[str, np.ndarray]:
    """Compute per-tick features for composite Markov state.

    Args:
        prices: 1-D array of BTC prices (one per second).
        timestamps: 1-D array of Unix timestamps matching prices.
        expiry_ts: Slot expiry Unix timestamp.
        return_lookback: Number of ticks for rolling return (default 5s).
        vol_lookback: Number of ticks for realized volatility (default 15s).

    Returns:
        Dict with keys: price, tte, returns, micro — each a 1-D array.
    """
    n = len(prices)
    prices = np.asarray(prices, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    # Time to expiry (seconds remaining)
    tte = np.maximum(0.0, expiry_ts - timestamps)

    # Rolling returns: (price[t] - price[t - lookback]) / price[t - lookback]
    returns = np.zeros(n)
    if n > return_lookback:
        past = prices[:-return_lookback]
        returns[return_lookback:] = (
            (prices[return_lookback:] - past) / np.maximum(past, 1e-10)
        )

    # Microstructure: rolling realized volatility (std of 1s log returns)
    log_ret = np.zeros(n)
    if n > 1:
        log_ret[1:] = np.diff(np.log(np.maximum(prices, 1e-10)))
    micro = pd.Series(log_ret).rolling(vol_lookback, min_periods=2).std().to_numpy()
    micro = np.nan_to_num(micro, nan=0.0)

    return {"price": prices, "tte": tte, "returns": returns, "micro": micro}


def build_composite_states(
    features: dict[str, np.ndarray],
    n_price: int = 15,
    n_tte: int = 4,
    n_return: int = 5,
    n_micro: int = 3,
    edges: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int, int]]:
    """Discretize per-tick features into composite state indices.

    Args:
        features: Dict from compute_features() with keys price/tte/returns/micro.
        n_price, n_tte, n_return, n_micro: Bin counts per dimension.
        edges: Optional pre-computed bin edges (dict of arrays). If None,
               edges are derived from the data.

    Returns:
        (states, edges_out, dims) where:
            states: 1-D int array of flat composite state indices
            edges_out: dict of bin edges per dimension
            dims: (n_price, n_tte, n_return, n_micro)
    """
    dims = (n_price, n_tte, n_return, n_micro)
    if edges is None:
        edges = {}

    pb, pe = discretize(features["price"], n_price, edges.get("price"))
    tb, te = discretize(features["tte"], n_tte, edges.get("tte"))
    rb, re = discretize(features["returns"], n_return, edges.get("returns"))
    mb, me = discretize(features["micro"], n_micro, edges.get("micro"))

    edges_out = {"price": pe, "tte": te, "returns": re, "micro": me}
    states = encode_composite(pb, tb, rb, mb, dims)
    return states, edges_out, dims


def encode_composite(
    price_bin: np.ndarray,
    tte_bin: np.ndarray,
    return_bin: np.ndarray,
    micro_bin: np.ndarray,
    dims: tuple[int, int, int, int],
) -> np.ndarray:
    """Encode component bins into a flat composite state index."""
    coords = np.array([
        np.asarray(price_bin), np.asarray(tte_bin),
        np.asarray(return_bin), np.asarray(micro_bin),
    ])
    return np.ravel_multi_index(coords, dims, mode="clip")


def decode_composite(
    states: np.ndarray,
    dims: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode flat composite index to (price_bin, tte_bin, return_bin, micro_bin)."""
    return np.unravel_index(np.asarray(states).clip(0, np.prod(dims) - 1), dims)


def simulate_composite_paths(
    T: np.ndarray,
    start_state: int,
    n_steps: int,
    dims: tuple[int, int, int, int],
    tte_schedule: np.ndarray,
    n_paths: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate composite Markov paths with deterministic TTE progression.

    At each step the chain proposes a full composite next-state, then the
    TTE component is overridden to match the known TTE schedule (TTE
    decreases deterministically toward 0 at expiry).

    Args:
        T: Row-stochastic transition matrix (n_total x n_total).
        start_state: Flat composite state index to start from.
        n_steps: Number of forward steps (≈ seconds to expiry).
        dims: (n_price, n_tte, n_return, n_micro).
        tte_schedule: 1-D int array of TTE bin indices at each step
                      (length >= n_steps + 1).
        n_paths: Number of independent paths.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        (n_paths, n_steps + 1) integer array of composite state indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = int(np.prod(dims))
    n_price, n_tte, n_return, n_micro = dims
    start_state = int(np.clip(start_state, 0, n_total - 1))

    paths = np.empty((n_paths, n_steps + 1), dtype=int)
    paths[:, 0] = start_state

    cumprobs = np.cumsum(T, axis=1)

    for t in range(n_steps):
        current = paths[:, t]
        u = rng.random(n_paths)
        # searchsorted is O(n_paths * log(n_total)) vs boolean matrix O(n_paths * n_total)
        next_state = np.array([
            np.searchsorted(cumprobs[c], u_i) for c, u_i in zip(current, u)
        ]).clip(0, n_total - 1)

        # Decode, override TTE bin, re-encode
        pb, _, rb, mb = decode_composite(next_state, dims)

        tte_idx = min(t + 1, len(tte_schedule) - 1)
        tb_val = int(np.clip(tte_schedule[tte_idx], 0, n_tte - 1))

        paths[:, t + 1] = encode_composite(
            pb, np.full(n_paths, tb_val, dtype=int), rb, mb, dims,
        )

    return paths


def estimate_p_yes_composite(
    paths: np.ndarray,
    price_edges: np.ndarray,
    dims: tuple[int, int, int, int],
    strike: float,
) -> float:
    """Estimate P(YES) from composite terminal states.

    Extracts the price-bin component from each terminal composite state,
    maps it to a price midpoint, and counts what fraction >= strike.
    """
    n_price = dims[0]
    terminal = paths[:, -1]
    price_bins, _, _, _ = decode_composite(terminal, dims)
    price_bins = np.clip(price_bins, 0, n_price - 1)
    midpoints = 0.5 * (price_edges[:-1] + price_edges[1:])
    terminal_prices = midpoints[price_bins]
    return float((terminal_prices >= strike).mean())


# ---------------------------------------------------------------------------
# Model class — predict(snapshot) -> PredictionResult interface
# ---------------------------------------------------------------------------

class MarkovModel:
    """Composite-state Markov chain Monte Carlo model for BTC 5-min markets.

    State S_t = (price_bin, tte_bin, return_bin, micro_bin) captures:
      - price_bin:  BTC price level
      - tte_bin:    time-to-expiry (deterministic during simulation)
      - return_bin: short-term BTC return (momentum)
      - micro_bin:  realized volatility (microstructure)

    Fits a transition matrix on the trailing BTC history, then simulates
    forward composite paths to the slot expiry to estimate P(YES).

    Parameters:
        n_price_bins: Discrete bins for BTC price.
        n_tte_bins:   Discrete bins for time-to-expiry.
        n_return_bins: Discrete bins for rolling return.
        n_micro_bins: Discrete bins for realized volatility.
        n_paths: Number of Monte Carlo forward paths.
        smoothing: Laplace smoothing for the transition matrix (higher
                   values needed because composite state space is larger).
        lookback_s: Seconds of BTC history used for fitting (default 600).
        return_lookback: Ticks for rolling return (default 5s).
        vol_lookback: Ticks for realized volatility (default 15s).
        seed: Random seed for reproducibility (None = non-deterministic).
    """

    MODEL_VERSION = "markov_mc_v2"

    def __init__(
        self,
        n_price_bins: int = 15,
        n_tte_bins: int = 4,
        n_return_bins: int = 5,
        n_micro_bins: int = 3,
        n_paths: int = 5000,
        smoothing: float = 1e-4,
        lookback_s: float = 600.0,
        return_lookback: int = 5,
        vol_lookback: int = 15,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.n_price_bins = n_price_bins
        self.n_tte_bins = n_tte_bins
        self.n_return_bins = n_return_bins
        self.n_micro_bins = n_micro_bins
        self.dims = (n_price_bins, n_tte_bins, n_return_bins, n_micro_bins)
        self.n_paths = n_paths
        self.smoothing = smoothing
        self.lookback_s = lookback_s
        self.return_lookback = return_lookback
        self.vol_lookback = vol_lookback
        self._rng = np.random.default_rng(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.ready = True

        # Populated after each predict() for observability / logging.
        self.last_features: dict = {}

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        """Estimate P(YES) using composite-state Markov chain MC simulation.

        Snapshot keys consumed:
            btc_prices: list of (ts, price[, volume]) tuples
            strike_price: float (or extracted from 'question')
            slot_expiry_ts: float — Unix timestamp of slot close
            now_ts: float (optional)
        """
        btc_prices: list = list(snapshot.get("btc_prices") or [])
        if len(btc_prices) < 30:
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
        slot_expiry_ts = float(slot_expiry_ts)
        tte = max(0.0, slot_expiry_ts - now_ts)

        # Extract price/timestamp series within lookback window.
        ts_arr = np.array([float(p[0]) for p in btc_prices])
        px_arr = np.array([float(p[1]) for p in btc_prices])
        mask = ts_arr >= (now_ts - self.lookback_s)
        ts_window = ts_arr[mask]
        px_window = px_arr[mask]
        if len(px_window) < 30:
            return PredictionResult(None, self.MODEL_VERSION, "insufficient_window")

        try:
            # Compute per-tick features
            features = compute_features(
                px_window, ts_window, slot_expiry_ts,
                self.return_lookback, self.vol_lookback,
            )

            # Build composite states
            states, edges, dims = build_composite_states(
                features,
                self.n_price_bins, self.n_tte_bins,
                self.n_return_bins, self.n_micro_bins,
            )

            # Build transition matrix on composite state space
            T = build_transition_matrix(
                states, int(np.prod(self.dims)), self.smoothing,
            )

            # Validate
            warnings = validate_transition_matrix(T)
            if warnings:
                for w in warnings:
                    self.logger.warning("Markov T matrix: %s", w)

            # TTE schedule: bin the decreasing TTE at each simulation step
            n_steps = max(1, int(round(tte)))
            tte_values = np.maximum(0.0, tte - np.arange(n_steps + 1))
            tte_schedule = np.clip(
                np.digitize(tte_values, edges["tte"]) - 1,
                0, self.n_tte_bins - 1,
            )

            # Simulate composite paths
            paths = simulate_composite_paths(
                T, start_state=int(states[-1]),
                n_steps=n_steps, dims=dims,
                tte_schedule=tte_schedule,
                n_paths=self.n_paths, rng=self._rng,
            )

            # Estimate P(YES)
            p_yes = estimate_p_yes_composite(
                paths, edges["price"], dims, strike_price,
            )

            # Observability: terminal price distribution
            price_bins, _, _, _ = decode_composite(paths[:, -1], dims)
            price_bins = np.clip(price_bins, 0, self.n_price_bins - 1)
            midpoints = 0.5 * (edges["price"][:-1] + edges["price"][1:])
            terminal_prices = midpoints[price_bins]

            self.last_features = {
                "n_observations": int(len(px_window)),
                "dims": self.dims,
                "n_total_states": int(np.prod(self.dims)),
                "n_states_occupied": int(len(np.unique(states))),
                "n_steps": n_steps,
                "n_paths": self.n_paths,
                "smoothing": self.smoothing,
                "strike": strike_price,
                "current_price": btc_mid,
                "current_tte": round(tte, 1),
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
