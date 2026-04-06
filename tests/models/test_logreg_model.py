"""Tests for LogRegModel — training, prediction, save/load."""

import numpy as np
import pytest

from src.models.logreg_model import LogRegModel, LR_FEATURES
from src.models.xgb_model import PredictionResult


def _make_snapshot(btc_prices, strike_price=50000.0, expiry_offset=300.0):
    """Helper to build a minimal snapshot dict."""
    now_ts = btc_prices[-1][0] if btc_prices else 0.0
    return {
        "btc_prices": btc_prices,
        "question": "Bitcoin Up or Down",
        "strike_price": strike_price,
        "slot_expiry_ts": now_ts + expiry_offset,
        "now_ts": now_ts,
    }


def _train_toy_model():
    """Train a model on synthetic data — just enough to verify the pipeline."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, len(LR_FEATURES))
    # dist_to_strike (index 5) is the main signal
    y = (X[:, 5] > 0).astype(float)
    return LogRegModel.train(X, y, model_version="test_v1")


def test_train_and_predict():
    model = _train_toy_model()
    assert model.ready

    # Build a plausible BTC price series
    base_price = 50000.0
    btc_prices = [(float(t), base_price + t * 0.1) for t in range(100)]
    snapshot = _make_snapshot(btc_prices, strike_price=base_price)

    result = model.predict(snapshot)
    assert isinstance(result, PredictionResult)
    assert result.prob_yes is not None
    assert 0.0 <= result.prob_yes <= 1.0
    assert result.feature_status == "ready"
    assert result.model_version == "test_v1"


def test_predict_insufficient_btc():
    model = _train_toy_model()
    snapshot = _make_snapshot([], strike_price=50000.0)
    result = model.predict(snapshot)
    assert result.prob_yes is None
    assert result.feature_status == "insufficient_btc_history"


def test_predict_missing_strike():
    model = _train_toy_model()
    btc_prices = [(float(t), 50000.0) for t in range(100)]
    snapshot = _make_snapshot(btc_prices)
    snapshot["strike_price"] = None
    snapshot["question"] = "no strike here"
    result = model.predict(snapshot)
    assert result.prob_yes is None
    assert result.feature_status == "missing_strike"


def test_predict_missing_expiry():
    model = _train_toy_model()
    btc_prices = [(float(t), 50000.0) for t in range(100)]
    snapshot = _make_snapshot(btc_prices)
    snapshot["slot_expiry_ts"] = None
    result = model.predict(snapshot)
    assert result.prob_yes is None
    assert result.feature_status == "missing_expiry"


def test_predict_not_loaded():
    model = LogRegModel()  # no model/scaler
    assert not model.ready
    btc_prices = [(float(t), 50000.0) for t in range(100)]
    snapshot = _make_snapshot(btc_prices)
    result = model.predict(snapshot)
    assert result.prob_yes is None
    assert result.feature_status == "model_not_loaded"


def test_save_and_load(tmp_path):
    model = _train_toy_model()
    model.save(str(tmp_path))

    loaded = LogRegModel.load(str(tmp_path))
    assert loaded.ready
    assert loaded.model_version == "test_v1"

    # Both should produce the same prediction
    btc_prices = [(float(t), 50000.0 + t * 0.05) for t in range(100)]
    snapshot = _make_snapshot(btc_prices)
    r1 = model.predict(snapshot)
    r2 = loaded.predict(snapshot)
    assert abs(r1.prob_yes - r2.prob_yes) < 1e-9


def test_prob_responds_to_dist_to_strike():
    """Model should predict higher P(Up) when BTC is above strike."""
    model = _train_toy_model()

    btc_above = [(float(t), 50100.0) for t in range(100)]
    btc_below = [(float(t), 49900.0) for t in range(100)]

    p_above = model.predict(_make_snapshot(btc_above, strike_price=50000.0)).prob_yes
    p_below = model.predict(_make_snapshot(btc_below, strike_price=50000.0)).prob_yes

    # dist_to_strike is the dominant feature, so p_above should be higher
    assert p_above > p_below
