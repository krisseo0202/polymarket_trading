"""Simple analytical baseline model for BTC Up/Down probability trading."""

from __future__ import annotations

import logging
import math
from typing import Mapping, Optional, Sequence, Tuple

from src.api.types import OrderBook

from .feature_builder import parse_strike_price
from .prediction import PredictionResult
from .schema import DEFAULT_THRESHOLDS


_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
_EPS = 1e-12


class BTCUpDownBaselineModel:
    """Estimate YES/NO probability with a simple GBM-style baseline."""

    def __init__(
        self,
        model_version: str = "btc_updown_baseline_v1",
        vol_window_seconds: int = 120,
        min_samples: int = 5,
        min_annualized_vol: float = 0.05,
        max_annualized_vol: float = 3.0,
        max_abs_drift: float = 3.0,
        logger: Optional[logging.Logger] = None,
    ):
        self._model_version = model_version
        self.vol_window_seconds = int(vol_window_seconds)
        self.min_samples = int(min_samples)
        self.min_annualized_vol = float(min_annualized_vol)
        self.max_annualized_vol = float(max_annualized_vol)
        self.max_abs_drift = float(max_abs_drift)
        self.logger = logger or logging.getLogger(__name__)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def thresholds(self):
        return DEFAULT_THRESHOLDS

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        btc_prices = list(snapshot.get("btc_prices") or [])
        if len(btc_prices) < 2:
            return PredictionResult(None, self.model_version, "insufficient_btc_history")

        current_price = float(btc_prices[-1][1])
        if current_price <= 0:
            return PredictionResult(None, self.model_version, "invalid_btc_price")

        question = str(snapshot.get("question") or "")
        strike_price = snapshot.get("strike_price")
        if strike_price is None:
            strike_price = parse_strike_price(question)
        if strike_price is None or float(strike_price) <= 0:
            return PredictionResult(None, self.model_version, "missing_strike")
        strike_price = float(strike_price)

        now_ts = float(snapshot.get("now_ts") or btc_prices[-1][0])
        slot_expiry_ts = snapshot.get("slot_expiry_ts")
        if slot_expiry_ts is None:
            return PredictionResult(None, self.model_version, "missing_expiry")
        seconds_to_expiry = max(0.0, float(slot_expiry_ts) - now_ts)
        if seconds_to_expiry <= 0:
            return PredictionResult(None, self.model_version, "market_expired")

        stats = self._estimate_annualized_stats(btc_prices, now_ts)
        if stats is None:
            return PredictionResult(None, self.model_version, "insufficient_vol_window")
        annualized_vol, annualized_drift = stats

        prob_yes = self._probability_above_strike(
            current_price=current_price,
            strike_price=strike_price,
            annualized_vol=annualized_vol,
            annualized_drift=annualized_drift,
            seconds_to_expiry=seconds_to_expiry,
        )

        yes_book = snapshot.get("yes_book")
        no_book = snapshot.get("no_book")
        yes_ask = self._best_ask(yes_book)
        no_ask = self._best_ask(no_book)
        edge_yes = prob_yes - yes_ask if yes_ask is not None else None
        prob_no = max(0.0, min(1.0, 1.0 - prob_yes))
        edge_no = prob_no - no_ask if no_ask is not None else None

        status = "ready"
        if yes_ask is None or no_ask is None:
            status = "ready_missing_order_books"

        return PredictionResult(
            prob_yes=prob_yes,
            model_version=self.model_version,
            feature_status=status,
            edge_yes=edge_yes,
            edge_no=edge_no,
        )

    def _estimate_annualized_stats(
        self,
        btc_prices: Sequence[Tuple[float, float]],
        now_ts: float,
    ) -> Optional[Tuple[float, float]]:
        cutoff = now_ts - self.vol_window_seconds
        window = [(float(ts), float(price)) for ts, price, *_ in btc_prices if float(ts) >= cutoff and float(price) > 0]
        if len(window) < self.min_samples:
            return None

        log_returns = []
        dt_seconds = []
        for (ts0, p0), (ts1, p1) in zip(window, window[1:]):
            dt = ts1 - ts0
            if dt <= 0 or p0 <= 0 or p1 <= 0:
                continue
            log_returns.append(math.log(p1 / p0))
            dt_seconds.append(dt)

        if len(log_returns) < max(2, self.min_samples - 1):
            return None

        # Use total-elapsed / (n-1) for mean_dt: robust against burst ticks from
        # WebSocket reconnects or clustering, which inflate sample count and bias
        # arithmetic mean toward shorter intervals.
        total_elapsed = window[-1][0] - window[0][0]
        if total_elapsed <= 0 or len(log_returns) < 1:
            return None
        mean_dt = total_elapsed / len(log_returns)

        mean_lr = sum(log_returns) / len(log_returns)
        variance = sum((value - mean_lr) ** 2 for value in log_returns) / max(1, len(log_returns) - 1)
        annualized_vol = math.sqrt(max(variance, 0.0)) * math.sqrt(_SECONDS_PER_YEAR / mean_dt)
        annualized_vol = min(max(annualized_vol, self.min_annualized_vol), self.max_annualized_vol)

        # Drift estimated from 120s of tick data is extremely noisy (std error >> mean).
        # At 5-min horizons the vol term dominates d2; zeroing drift reduces noise injection.
        annualized_drift = 0.0
        return annualized_vol, annualized_drift

    def _probability_above_strike(
        self,
        current_price: float,
        strike_price: float,
        annualized_vol: float,
        annualized_drift: float,
        seconds_to_expiry: float,
    ) -> float:
        horizon_years = max(seconds_to_expiry / _SECONDS_PER_YEAR, _EPS)
        sigma_sqrt_t = annualized_vol * math.sqrt(horizon_years)

        if sigma_sqrt_t <= _EPS:
            forward = current_price * math.exp(annualized_drift * horizon_years)
            if forward > strike_price:
                return 1.0
            if forward < strike_price:
                return 0.0
            return 0.5

        d2 = (
            math.log(current_price / strike_price)
            + (annualized_drift - 0.5 * annualized_vol * annualized_vol) * horizon_years
        ) / sigma_sqrt_t
        return max(0.0, min(1.0, _normal_cdf(d2)))

    @staticmethod
    def _best_ask(book: object) -> Optional[float]:
        if not isinstance(book, OrderBook):
            return None
        if not book.asks:
            return None
        ask = float(book.asks[0].price)
        return ask if ask > 0 else None


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
