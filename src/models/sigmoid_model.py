"""Minimal sigmoid probability model for BTC Up/Down 5-minute markets.

Score formula:
    distance    = log(price_now / strike)
    time_weight = 1 + amplifier * (1 - tte / max_tte)   # amplify near expiry
    score       = w_dist  * distance * time_weight
                + w_mom1  * momentum_1m
                + w_mom3  * momentum_3m
                + w_mom5  * momentum_5m
                + w_td    * td_signal          (optional)
    prob_yes    = sigmoid(score)
"""

from __future__ import annotations

import logging
import math
from typing import Mapping, Optional, Sequence, Tuple

from .feature_builder import parse_strike_price
from .prediction import PredictionResult
from .schema import DEFAULT_THRESHOLDS

_EPS = 1e-12


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _lookup_price_at(
    btc_prices: Sequence[Tuple[float, float]],
    now_ts: float,
    lookback_seconds: float,
) -> Optional[float]:
    """Return the BTC price closest to (now_ts - lookback_seconds), or None."""
    target_ts = now_ts - lookback_seconds
    best: Optional[Tuple[float, float]] = None
    for ts, price, *_ in btc_prices:
        ts, price = float(ts), float(price)
        if price <= 0:
            continue
        if best is None or abs(ts - target_ts) < abs(best[0] - target_ts):
            best = (ts, price)
    return best[1] if best is not None else None


class BTCSigmoidModel:
    """Minimal, interpretable probability model using a weighted sigmoid."""

    def __init__(
        self,
        model_version: str = "btc_sigmoid_v1",
        # Feature weights
        w_dist: float = 50.0,
        w_mom1: float = 20.0,
        w_mom3: float = 12.0,
        w_mom5: float = 6.0,
        w_td: float = 0.5,
        # Time-amplification
        amplifier: float = 2.0,
        max_tte: float = 300.0,
        # Data requirements
        min_btc_samples: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self._model_version = model_version
        self.w_dist = w_dist
        self.w_mom1 = w_mom1
        self.w_mom3 = w_mom3
        self.w_mom5 = w_mom5
        self.w_td = w_td
        self.amplifier = amplifier
        self.max_tte = max_tte
        self.min_btc_samples = min_btc_samples
        self.logger = logger or logging.getLogger(__name__)
        self.last_breakdown: dict = {}

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def thresholds(self):
        return DEFAULT_THRESHOLDS

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        btc_prices = list(snapshot.get("btc_prices") or [])
        if len(btc_prices) < self.min_btc_samples:
            return PredictionResult(None, self.model_version, "insufficient_btc_history")

        try:
            price_now = float(btc_prices[-1][1])
        except (IndexError, TypeError, ValueError):
            return PredictionResult(None, self.model_version, "invalid_btc_price")
        if price_now <= 0:
            return PredictionResult(None, self.model_version, "invalid_btc_price")

        strike = snapshot.get("strike_price")
        if strike is None:
            strike = parse_strike_price(str(snapshot.get("question") or ""))
        if strike is None or float(strike) <= 0:
            return PredictionResult(None, self.model_version, "missing_strike")
        strike = float(strike)

        slot_expiry_ts = snapshot.get("slot_expiry_ts")
        if slot_expiry_ts is None:
            return PredictionResult(None, self.model_version, "missing_expiry")
        now_ts = float(snapshot.get("now_ts") or btc_prices[-1][0])
        tte = max(0.0, float(slot_expiry_ts) - now_ts)
        if tte <= 0:
            return PredictionResult(None, self.model_version, "market_expired")

        # ── Core features ──────────────────────────────────────────────────────
        distance = math.log(price_now / strike)
        time_weight = 1.0 + self.amplifier * (1.0 - min(tte, self.max_tte) / self.max_tte)

        def _momentum(lookback: float) -> float:
            ref = _lookup_price_at(btc_prices, now_ts, lookback)
            if ref is None or ref <= 0:
                return 0.0
            return math.log(price_now / ref)

        momentum_1m = _momentum(60.0)
        momentum_3m = _momentum(180.0)
        momentum_5m = _momentum(300.0)
        td_signal = float(snapshot.get("td_signal") or 0)

        # ── Sigmoid score ──────────────────────────────────────────────────────
        score = (
            self.w_dist * distance * time_weight
            + self.w_mom1 * momentum_1m
            + self.w_mom3 * momentum_3m
            + self.w_mom5 * momentum_5m
            + self.w_td * td_signal
        )
        prob_yes = _sigmoid(score)

        # ── Store breakdown for dashboard inspection ───────────────────────────
        self.last_breakdown = {
            "distance":     round(distance, 6),
            "time_weight":  round(time_weight, 3),
            "dist_contrib": round(self.w_dist * distance * time_weight, 4),
            "mom1_contrib": round(self.w_mom1 * momentum_1m, 4),
            "mom3_contrib": round(self.w_mom3 * momentum_3m, 4),
            "mom5_contrib": round(self.w_mom5 * momentum_5m, 4),
            "td_contrib":   round(self.w_td * td_signal, 4),
            "score":        round(score, 4),
            "prob_yes":     round(prob_yes, 4),
        }

        # ── Edge vs order book ─────────────────────────────────────────────────
        prob_no = max(0.0, min(1.0, 1.0 - prob_yes))

        def _best_ask(book) -> Optional[float]:
            if book is None:
                return None
            asks = getattr(book, "asks", None)
            if not asks:
                return None
            try:
                v = float(asks[0].price)
                return v if v > 0 else None
            except (AttributeError, IndexError, TypeError, ValueError):
                return None

        yes_ask = _best_ask(snapshot.get("yes_book"))
        no_ask = _best_ask(snapshot.get("no_book"))
        edge_yes = prob_yes - yes_ask if yes_ask is not None else None
        edge_no = prob_no - no_ask if no_ask is not None else None

        status = "ready" if (yes_ask is not None and no_ask is not None) else "ready_missing_order_books"

        return PredictionResult(
            prob_yes=prob_yes,
            model_version=self.model_version,
            feature_status=status,
            edge_yes=edge_yes,
            edge_no=edge_no,
        )
