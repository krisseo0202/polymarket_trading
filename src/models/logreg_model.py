"""Logistic regression model for BTC Up/Down probability prediction.

Computes 8 BTC-only features from the live snapshot and returns a
calibrated P(Up) via a pre-trained sklearn LogisticRegression.

Usage:
    model = LogRegModel.from_snapshot_data(markets_df, btc_df)  # train
    model = LogRegModel.load("models/logreg")                   # load saved
    result = model.predict(snapshot)                             # live
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .feature_builder import parse_strike_price
from .xgb_model import PredictionResult


LR_FEATURES = [
    "ret_15s", "ret_30s", "ret_60s", "rolling_vol_60s",
    "rsi_14", "dist_to_strike", "ma_12_gap", "time_to_expiry_sec",
]


class LogRegModel:
    """Logistic regression model with the same predict() interface as the XGB model."""

    def __init__(
        self,
        model=None,
        scaler=None,
        model_version: str = "logreg_v1",
        logger: Optional[logging.Logger] = None,
    ):
        self._model = model
        self._scaler = scaler
        self._model_version = model_version
        self.logger = logger or logging.getLogger(__name__)
        self.ready = model is not None and scaler is not None

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "min_edge": 0.05,
            "min_prob_yes": 0.54,
            "max_prob_yes_for_no": 0.46,
            "max_spread_pct": 0.06,
            "exit_edge": -0.01,
            "min_seconds_to_expiry": 20.0,
            "max_seconds_to_expiry": 280.0,
        }

    # ------------------------------------------------------------------
    # Live prediction
    # ------------------------------------------------------------------

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        """Predict P(Up) from a live snapshot. Same interface as XGB/baseline models."""
        if not self.ready:
            return PredictionResult(None, self._model_version, "model_not_loaded")

        btc_prices: List[Tuple[float, float]] = list(snapshot.get("btc_prices") or [])
        if len(btc_prices) < 2:
            return PredictionResult(None, self._model_version, "insufficient_btc_history")

        now_ts = float(snapshot.get("now_ts") or btc_prices[-1][0])
        btc_mid = float(btc_prices[-1][1])
        if btc_mid <= 0:
            return PredictionResult(None, self._model_version, "invalid_btc_price")

        strike_price = snapshot.get("strike_price")
        if strike_price is None:
            question = str(snapshot.get("question") or "")
            strike_price = parse_strike_price(question)
        if strike_price is None or float(strike_price) <= 0:
            return PredictionResult(None, self._model_version, "missing_strike")
        strike_price = float(strike_price)

        slot_expiry_ts = snapshot.get("slot_expiry_ts")
        if slot_expiry_ts is None:
            return PredictionResult(None, self._model_version, "missing_expiry")
        tte = max(0.0, float(slot_expiry_ts) - now_ts)

        ts_arr = np.array([float(p[0]) for p in btc_prices])
        px_arr = np.array([float(p[1]) for p in btc_prices])

        features = np.array([[
            _ts_return(ts_arr, px_arr, now_ts, 15),
            _ts_return(ts_arr, px_arr, now_ts, 30),
            _ts_return(ts_arr, px_arr, now_ts, 60),
            _rolling_vol(ts_arr, px_arr, now_ts, 60),
            _rsi(ts_arr, px_arr, now_ts, 14),
            (btc_mid - strike_price) / strike_price,
            _ma_gap(ts_arr, px_arr, now_ts, 12, btc_mid),
            tte,
        ]], dtype=float)

        try:
            scaled = self._scaler.transform(features)
            prob_yes = float(self._model.predict_proba(scaled)[0, 1])
        except Exception as exc:
            self.logger.warning("logreg predict failed: %s", exc)
            return PredictionResult(None, self._model_version, "predict_failed")

        prob_yes = max(0.0, min(1.0, prob_yes))
        return PredictionResult(prob_yes, self._model_version, "ready")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save model and scaler to directory."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "logreg_model.pkl"), "wb") as f:
            pickle.dump(self._model, f)
        with open(os.path.join(directory, "logreg_scaler.pkl"), "wb") as f:
            pickle.dump(self._scaler, f)
        meta = {
            "model_version": self._model_version,
            "features": LR_FEATURES,
            "coef": self._model.coef_[0].tolist() if self._model else [],
            "intercept": float(self._model.intercept_[0]) if self._model else 0.0,
        }
        with open(os.path.join(directory, "logreg_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str, logger: Optional[logging.Logger] = None) -> LogRegModel:
        """Load a saved model from directory."""
        log = logger or logging.getLogger(__name__)
        try:
            with open(os.path.join(directory, "logreg_model.pkl"), "rb") as f:
                model = pickle.load(f)
            with open(os.path.join(directory, "logreg_scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)
            meta_path = os.path.join(directory, "logreg_meta.json")
            version = "logreg_v1"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    version = json.load(f).get("model_version", version)
            return cls(model=model, scaler=scaler, model_version=version, logger=log)
        except Exception as exc:
            log.error("Failed to load logreg model from %s: %s", directory, exc)
            return cls(logger=log)

    # ------------------------------------------------------------------
    # Training convenience
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        model_version: str = "logreg_v1",
        logger: Optional[logging.Logger] = None,
    ) -> LogRegModel:
        """Train a new model from feature matrix and labels."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(X_scaled, y)
        return cls(model=model, scaler=scaler, model_version=model_version, logger=logger)


# -- feature helpers (operate on numpy arrays for live speed) ----------------

def _asof_idx(ts_arr: np.ndarray, target: float) -> int:
    """Index of the last timestamp <= target, or -1."""
    idx = int(np.searchsorted(ts_arr, target, side="right")) - 1
    return idx


def _ts_return(ts_arr: np.ndarray, px_arr: np.ndarray, now: float, lookback: int) -> float:
    cur_idx = _asof_idx(ts_arr, now)
    prev_idx = _asof_idx(ts_arr, now - lookback)
    if cur_idx < 0 or prev_idx < 0:
        return 0.0
    prev = float(px_arr[prev_idx])
    if prev <= 0:
        return 0.0
    return (float(px_arr[cur_idx]) - prev) / prev


def _rolling_vol(ts_arr: np.ndarray, px_arr: np.ndarray, now: float, lookback: int) -> float:
    start = now - lookback
    mask = (ts_arr >= start) & (ts_arr <= now)
    prices = px_arr[mask]
    if len(prices) < 2:
        return 0.0
    rets = np.diff(prices) / prices[:-1]
    return float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0


def _rsi(ts_arr: np.ndarray, px_arr: np.ndarray, now: float, period: int) -> float:
    idx = int(np.searchsorted(ts_arr, now, side="right"))
    need = period + 1
    if idx < need:
        return 50.0
    segment = px_arr[idx - need:idx].astype(float)
    deltas = np.diff(segment)
    gains = float(np.sum(deltas[deltas > 0]))
    losses = float(-np.sum(deltas[deltas < 0]))
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ma_gap(ts_arr: np.ndarray, px_arr: np.ndarray, now: float, n: int, btc_mid: float) -> float:
    idx = int(np.searchsorted(ts_arr, now, side="right"))
    if idx == 0:
        return 0.0
    start = max(0, idx - n)
    ma = float(np.mean(px_arr[start:idx]))
    if ma <= 0:
        return 0.0
    return (btc_mid - ma) / ma
