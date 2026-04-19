"""Logistic regression v4 model loader — BTC-only feature family.

The v4 model adds `ob_imbalance` + `ob_cross_imbalance` (18 features total)
plus isotonic calibration on top of the v3 feature set. This loader is also
forward-compatible with retrains that include the v5 microstructure features
(microprice_delta, imbalance_mean_10s, imbalance_slope_10s,
bid_ask_size_ratio_change_5s) and backward-compatible with v3 checkpoints
(16 features, no calibrator).

Feature selection is **meta-driven**: the set of features fed to the scaler at
inference time comes from `logreg_meta.json["features"]`, not a hard-coded list.
This guarantees the inference feature vector matches the shape the trained
scaler/model expect, regardless of which version is on disk.

Same predict(snapshot) -> PredictionResult interface as other models, so it
plugs directly into LogRegEdgeStrategy.

Usage:
    model = LogRegV4Model.load("models/logreg_v4")
    result = model.predict(snapshot)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..api.types import OrderBook
from .feature_builder import parse_strike_price
from .prediction import PredictionResult


# Default feature set used when a loaded meta.json has no `features` list
# (old v3 checkpoints). New checkpoints drive the live feature vector via
# meta — see LogRegV4Model.load / predict.
V3_FEATURES = [
    "time_to_expiry",
    "ret_5s", "ret_15s", "ret_30s", "ret_60s", "ret_180s",
    "vol_15s", "vol_30s", "vol_60s", "vol_ratio_15_60",
    "volume_surge_ratio", "vwap_deviation", "cvd_60s",
    "rsi_14", "td_setup_net", "spread",
]

# v4 additions (order-book imbalance).
V4_FEATURES_EXTRA = ["ob_imbalance", "ob_cross_imbalance"]

# v5 additions (rolling microstructure from yes_ob_history). Forward-
# compatible: only computed/emitted when the loaded model lists them.
V5_FEATURES_EXTRA = [
    "microprice_delta",
    "imbalance_mean_10s",
    "imbalance_slope_10s",
    "bid_ask_size_ratio_change_5s",
]

# Superset of every feature this loader knows how to build live. The actual
# vector at inference is projected onto `self.feature_names` (from meta).
_KNOWN_FEATURES = set(V3_FEATURES) | set(V4_FEATURES_EXTRA) | set(V5_FEATURES_EXTRA)


class LogRegV4Model:
    """LogReg v4 (and compatible v3/v5) loader with the standard predict() interface."""

    def __init__(
        self,
        model=None,
        scaler=None,
        calibrator=None,
        model_version: str = "logreg_v4",
        feature_names: Optional[Sequence[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._model = model
        self._scaler = scaler
        self._calibrator = calibrator  # isotonic calibration (optional)
        self._model_version = model_version
        # Feature names drive the live feature vector — must match the order
        # the scaler/model were trained with. Falls back to V3_FEATURES for
        # old checkpoints without a features list in meta.json.
        self.feature_names: List[str] = list(feature_names) if feature_names else list(V3_FEATURES)
        self.logger = logger or logging.getLogger(__name__)
        self.ready = model is not None and scaler is not None
        if self.ready:
            unknown = [f for f in self.feature_names if f not in _KNOWN_FEATURES]
            if unknown:
                self.logger.error(
                    "logreg model %s declares features this loader cannot build: %s — "
                    "disabling model", self._model_version, unknown,
                )
                self.ready = False
            else:
                expected = getattr(self._scaler, "n_features_in_", len(self.feature_names))
                if expected != len(self.feature_names):
                    self.logger.error(
                        "logreg %s feature count mismatch: meta lists %d, scaler expects %d — "
                        "disabling model", self._model_version, len(self.feature_names), expected,
                    )
                    self.ready = False
        self.last_features: dict = {}  # populated on each predict() for logging

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
            "min_seconds_to_expiry": 10.0,
            "max_seconds_to_expiry": 295.0,
        }

    # ------------------------------------------------------------------
    # Live prediction
    # ------------------------------------------------------------------

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        if not self.ready:
            return PredictionResult(None, self._model_version, "model_not_loaded")

        btc_prices: list = list(snapshot.get("btc_prices") or [])
        if len(btc_prices) < 15:
            return PredictionResult(None, self._model_version, "insufficient_btc_history")

        now_ts = float(snapshot.get("now_ts") or btc_prices[-1][0])
        btc_mid = float(btc_prices[-1][1])
        if btc_mid <= 0:
            return PredictionResult(None, self._model_version, "invalid_btc_price")

        # Strike price
        strike_price = snapshot.get("strike_price")
        if strike_price is None:
            question = str(snapshot.get("question") or "")
            strike_price = parse_strike_price(question)
        if strike_price is None or float(strike_price) <= 0:
            return PredictionResult(None, self._model_version, "missing_strike")
        strike_price = float(strike_price)

        # Time to expiry
        slot_expiry_ts = snapshot.get("slot_expiry_ts")
        if slot_expiry_ts is None:
            return PredictionResult(None, self._model_version, "missing_expiry")
        tte = max(0.0, float(slot_expiry_ts) - now_ts)

        # Build numpy arrays from btc_prices (supports 2-tuple and 3-tuple)
        ts_arr = np.array([float(p[0]) for p in btc_prices])
        px_arr = np.array([float(p[1]) for p in btc_prices])
        has_vol = len(btc_prices[0]) >= 3
        vol_arr = np.array([float(p[2]) for p in btc_prices]) if has_vol else np.ones(len(btc_prices))

        # Spread + orderbook imbalance from live orderbook
        spread = 0.01
        ob_imbalance = 0.0
        ob_cross_imbalance = 0.5
        yes_bid = 0.0
        yes_ask = 0.0
        yes_bid_d = 0.0
        yes_ask_d = 0.0
        yes_book = snapshot.get("yes_book")
        no_book = snapshot.get("no_book")
        if isinstance(yes_book, OrderBook) and yes_book.bids and yes_book.asks:
            yes_bid = float(yes_book.bids[0].price)
            yes_ask = float(yes_book.asks[0].price)
            spread = max(0.0, yes_ask - yes_bid)
            # OB imbalance: sum top-3 bid depth vs ask depth
            yes_bid_d = sum(float(l.size) for l in yes_book.bids[:3])
            yes_ask_d = sum(float(l.size) for l in yes_book.asks[:3])
            ob_total = yes_bid_d + yes_ask_d
            ob_imbalance = (yes_bid_d - yes_ask_d) / ob_total if ob_total > 0 else 0.0
            # Cross-side imbalance: YES bid depth vs NO bid depth
            if isinstance(no_book, OrderBook) and no_book.bids:
                no_bid_d = sum(float(l.size) for l in no_book.bids[:3])
                cross_total = yes_bid_d + no_bid_d
                ob_cross_imbalance = yes_bid_d / cross_total if cross_total > 0 else 0.5

        # Compute features into a dict. `predict()` will project onto
        # self.feature_names (from the trained model's meta.json) to build
        # the final vector — guaranteeing shape compatibility with the
        # trained scaler regardless of which version is loaded.
        v15 = _vol(ts_arr, px_arr, now_ts, 15)
        v30 = _vol(ts_arr, px_arr, now_ts, 30)
        v60 = _vol(ts_arr, px_arr, now_ts, 60)

        feats: Dict[str, float] = {
            "time_to_expiry": tte,
            "ret_5s":   _ret(ts_arr, px_arr, now_ts, 5),
            "ret_15s":  _ret(ts_arr, px_arr, now_ts, 15),
            "ret_30s":  _ret(ts_arr, px_arr, now_ts, 30),
            "ret_60s":  _ret(ts_arr, px_arr, now_ts, 60),
            "ret_180s": _ret(ts_arr, px_arr, now_ts, 180),
            "vol_15s": v15,
            "vol_30s": v30,
            "vol_60s": v60,
            "vol_ratio_15_60": (v15 / v60) if v60 > 0 else 0.0,
            "volume_surge_ratio": _volume_surge(ts_arr, vol_arr, now_ts),
            "vwap_deviation": _vwap_deviation(ts_arr, px_arr, vol_arr, now_ts),
            "cvd_60s": _cvd(ts_arr, px_arr, vol_arr, now_ts),
            "rsi_14": _rsi(ts_arr, px_arr, now_ts),
            "td_setup_net": _td_setup_net(ts_arr, px_arr, now_ts),
            "spread": spread,
            "ob_imbalance": ob_imbalance,
            "ob_cross_imbalance": ob_cross_imbalance,
        }

        # v5 rolling OB microstructure features — only computed when the
        # loaded model actually declares them (avoids wasted work on v3/v4).
        if any(f in self.feature_names for f in V5_FEATURES_EXTRA):
            ob_history = snapshot.get("yes_ob_history") or []
            feats["microprice_delta"] = _microprice_delta(yes_bid, yes_ask, yes_bid_d, yes_ask_d)
            feats["imbalance_mean_10s"] = _imbalance_mean(ob_history, now_ts, 10.0)
            feats["imbalance_slope_10s"] = _imbalance_slope(ob_history, now_ts, 10.0)
            feats["bid_ask_size_ratio_change_5s"] = _ratio_change(ob_history, now_ts, 5.0)

        # Project dict → ordered vector matching the trained scaler.
        try:
            features = np.array([[feats[name] for name in self.feature_names]], dtype=float)
        except KeyError as exc:
            self.logger.warning("logreg_v4 missing feature %s for version %s",
                                exc, self._model_version)
            return PredictionResult(None, self._model_version, "feature_missing")

        self.last_features = {name: float(feats[name]) for name in self.feature_names}

        try:
            scaled = self._scaler.transform(features)
            prob_yes = float(self._model.predict_proba(scaled)[0, 1])
            if self._calibrator is not None:
                prob_yes = float(self._calibrator.predict(np.array([prob_yes]))[0])
        except Exception as exc:
            self.logger.warning("logreg_v4 predict failed: %s", exc)
            return PredictionResult(None, self._model_version, "predict_failed")

        prob_yes = max(0.0, min(1.0, prob_yes))
        return PredictionResult(prob_yes, self._model_version, "ready")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "logreg_model.pkl"), "wb") as f:
            pickle.dump(self._model, f)
        with open(os.path.join(directory, "logreg_scaler.pkl"), "wb") as f:
            pickle.dump(self._scaler, f)
        meta = {
            "model_version": self._model_version,
            "features": list(self.feature_names),
            "coef": self._model.coef_[0].tolist() if self._model else [],
            "intercept": float(self._model.intercept_[0]) if self._model else 0.0,
        }
        with open(os.path.join(directory, "logreg_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str, logger: Optional[logging.Logger] = None) -> LogRegV4Model:
        log = logger or logging.getLogger(__name__)
        try:
            with open(os.path.join(directory, "logreg_model.pkl"), "rb") as f:
                model = pickle.load(f)
            with open(os.path.join(directory, "logreg_scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)
            # Load calibrator if available (v4+)
            calibrator = None
            cal_path = os.path.join(directory, "logreg_calibrator.pkl")
            if os.path.exists(cal_path):
                with open(cal_path, "rb") as f:
                    calibrator = pickle.load(f)
                log.info("Loaded isotonic calibrator from %s", cal_path)
            meta_path = os.path.join(directory, "logreg_meta.json")
            version = "logreg_v4"
            feature_names: Optional[List[str]] = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                version = meta.get("model_version", version)
                if isinstance(meta.get("features"), list) and meta["features"]:
                    feature_names = [str(f) for f in meta["features"]]
            log.info("Loaded logreg model %s from %s (%d features)",
                     version, directory,
                     len(feature_names) if feature_names else len(V3_FEATURES))
            return cls(model=model, scaler=scaler, calibrator=calibrator,
                       model_version=version, feature_names=feature_names, logger=log)
        except Exception as exc:
            log.error("Failed to load logreg model from %s: %s", directory, exc)
            return cls(logger=log)


# -- Feature helpers (numpy, same as train_logreg_v3.py) -------------------

def _asof(ts: np.ndarray, target: float) -> int:
    return int(np.searchsorted(ts, target, side="right")) - 1


def _ret(ts: np.ndarray, px: np.ndarray, now: float, lookback: int) -> float:
    ci = _asof(ts, now)
    pi = _asof(ts, now - lookback)
    if ci < 0 or pi < 0 or px[pi] <= 0:
        return 0.0
    return (float(px[ci]) - float(px[pi])) / float(px[pi])


def _vol(ts: np.ndarray, px: np.ndarray, now: float, lookback: int) -> float:
    mask = (ts >= now - lookback) & (ts <= now)
    prices = px[mask]
    if len(prices) < 3:
        return 0.0
    r = np.diff(np.log(prices))
    return float(np.std(r, ddof=1))


def _rsi(ts: np.ndarray, px: np.ndarray, now: float, period: int = 14) -> float:
    idx = int(np.searchsorted(ts, now, side="right"))
    need = period + 1
    if idx < need:
        return 50.0
    seg = px[idx - need:idx].astype(float)
    d = np.diff(seg)
    g = float(np.sum(d[d > 0])) / period
    l = float(-np.sum(d[d < 0])) / period
    if l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + g / l)


def _td_setup_net(ts: np.ndarray, px: np.ndarray, now: float, lookback: int = 4) -> float:
    idx = int(np.searchsorted(ts, now, side="right"))
    if idx < lookback + 1:
        return 0.0
    n = min(idx, 13 + lookback)
    seg = px[idx - n:idx].astype(float)
    bull = bear = 0
    for i in range(lookback, len(seg)):
        if seg[i] < seg[i - lookback]:
            bull += 1; bear = 0
        elif seg[i] > seg[i - lookback]:
            bear += 1; bull = 0
        else:
            bull = bear = 0
    return float(bear - bull)


def _volume_surge(ts: np.ndarray, vol: np.ndarray, now: float) -> float:
    v15 = vol[(ts >= now - 15) & (ts <= now)]
    v60 = vol[(ts >= now - 60) & (ts <= now)]
    a15 = float(v15.mean()) if len(v15) > 0 else 0.0
    a60 = float(v60.mean()) if len(v60) > 0 else 0.0
    return (a15 / a60) if a60 > 0 else 0.0


def _vwap_deviation(ts: np.ndarray, px: np.ndarray, vol: np.ndarray, now: float, lookback: int = 60) -> float:
    mask = (ts >= now - lookback) & (ts <= now)
    p, v = px[mask], vol[mask]
    if len(p) < 2 or v.sum() == 0:
        return 0.0
    vwap = float(np.sum(p * v) / np.sum(v))
    spot = float(p[-1])
    return (spot - vwap) / spot if spot > 0 else 0.0


def _cvd(ts: np.ndarray, px: np.ndarray, vol: np.ndarray, now: float, lookback: int = 60) -> float:
    mask = (ts >= now - lookback) & (ts <= now)
    p, v = px[mask], vol[mask]
    if len(p) < 2:
        return 0.0
    d = np.diff(p)
    buy = float(np.sum(v[1:][d >= 0]))
    sell = float(np.sum(v[1:][d < 0]))
    total = buy + sell
    return (buy - sell) / total if total > 0 else 0.0


# -- v5 rolling orderbook microstructure helpers --------------------------
#
# Shared between live (LogRegV4Model.predict) and training (train_logreg_v3.py
# imports these directly) to guarantee train/serve parity.
#
# ob_history is a sequence of (ts, bid_d, ask_d) tuples sorted ascending by ts,
# where bid_d/ask_d are top-3 depth aggregates matching the live ob_imbalance
# computation. A minimum window of history is required for rolling features;
# when insufficient, all helpers return 0.0.


def _microprice_delta(bid: float, ask: float, bid_d: float, ask_d: float) -> float:
    """Depth-weighted microprice skew relative to mid. Stateless.

    microprice = (ask * bid_d + bid * ask_d) / (bid_d + ask_d)
    Positive → more bid-side depth (book leans up).
    """
    total = bid_d + ask_d
    if total <= 0 or bid <= 0 or ask <= 0:
        return 0.0
    microprice = (ask * bid_d + bid * ask_d) / total
    mid = 0.5 * (bid + ask)
    return float(microprice - mid)


def _ob_imbalance_value(bid_d: float, ask_d: float) -> float:
    total = bid_d + ask_d
    return (bid_d - ask_d) / total if total > 0 else 0.0


def _imbalance_mean(history: Sequence, now_ts: float, window: float) -> float:
    if not history:
        return 0.0
    cutoff = now_ts - window
    vals = [_ob_imbalance_value(float(h[1]), float(h[2]))
            for h in history if float(h[0]) >= cutoff and float(h[0]) <= now_ts]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _imbalance_slope(history: Sequence, now_ts: float, window: float) -> float:
    """OLS slope of ob_imbalance vs time (seconds) over the trailing window."""
    if not history:
        return 0.0
    cutoff = now_ts - window
    pts = [(float(h[0]), _ob_imbalance_value(float(h[1]), float(h[2])))
           for h in history if float(h[0]) >= cutoff and float(h[0]) <= now_ts]
    if len(pts) < 2:
        return 0.0
    ts = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    ts = ts - ts[0]  # numerically stable
    if float(np.ptp(ts)) <= 0:
        return 0.0
    # slope only from polyfit
    slope, _ = np.polyfit(ts, ys, 1)
    return float(slope)


def _ratio_change(history: Sequence, now_ts: float, lag: float) -> float:
    """Change in (bid_d/ask_d) between now and the sample nearest `now - lag`.

    Uses the current (last) sample as "now" and the snapshot whose timestamp
    is closest to `now - lag` (within `now - 2*lag` to `now`) as the lagged
    reference. Returns 0.0 when no lagged sample exists.
    """
    if not history:
        return 0.0
    hist = [(float(h[0]), float(h[1]), float(h[2])) for h in history
            if float(h[0]) <= now_ts]
    if len(hist) < 2:
        return 0.0
    # Current = most recent sample at or before now_ts
    cur_ts, cur_bid_d, cur_ask_d = hist[-1]
    cur_ratio = (cur_bid_d / cur_ask_d) if cur_ask_d > 0 else 0.0
    # Find sample with ts closest to (cur_ts - lag) within [cur_ts - 2*lag, cur_ts - 0.5*lag]
    target = cur_ts - lag
    lo = cur_ts - 2.0 * lag
    hi = cur_ts - 0.5 * lag
    candidates = [h for h in hist[:-1] if lo <= h[0] <= hi]
    if not candidates:
        return 0.0
    ref = min(candidates, key=lambda h: abs(h[0] - target))
    _, ref_bid_d, ref_ask_d = ref
    ref_ratio = (ref_bid_d / ref_ask_d) if ref_ask_d > 0 else 0.0
    return float(cur_ratio - ref_ratio)
