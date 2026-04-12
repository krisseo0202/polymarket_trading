"""Tests for LogRegV4Model — the meta-driven logreg loader.

Regression context: v4 on disk was trained with 18 features. An earlier
rework appended 4 v5 microstructure features to the hard-coded predict
path, producing a 22-col feature vector that didn't match the 18-col
scaler. `scaler.transform(features)` raised `ValueError: X has 22 features,
but StandardScaler is expecting 18` and v4 was entirely broken for live
trading. The fix was to project features onto `feature_names` from the
model's meta.json at runtime.

These tests pin that behavior so the bug cannot recur.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.types import OrderBook, OrderBookEntry
from src.models.logreg_v4_model import (
    LogRegV4Model,
    V3_FEATURES,
    V4_FEATURES_EXTRA,
    V5_FEATURES_EXTRA,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _synthetic_snapshot() -> dict:
    """A minimal well-formed live snapshot the predict path can consume."""
    now = time.time()
    btc_prices = [(now - 200 + i, 60000.0 + (i % 5) - 2, 1.0) for i in range(200)]
    yb = [
        OrderBookEntry(price=0.55, size=100.0),
        OrderBookEntry(price=0.54, size=50.0),
        OrderBookEntry(price=0.53, size=40.0),
    ]
    ya = [
        OrderBookEntry(price=0.56, size=80.0),
        OrderBookEntry(price=0.57, size=70.0),
        OrderBookEntry(price=0.58, size=60.0),
    ]
    nb = [
        OrderBookEntry(price=0.44, size=100.0),
        OrderBookEntry(price=0.43, size=60.0),
        OrderBookEntry(price=0.42, size=40.0),
    ]
    na = [
        OrderBookEntry(price=0.45, size=90.0),
        OrderBookEntry(price=0.46, size=70.0),
        OrderBookEntry(price=0.47, size=50.0),
    ]
    yes_book = OrderBook(market_id="m", token_id="y", bids=yb, asks=ya, timestamp=now)
    no_book = OrderBook(market_id="m", token_id="n", bids=nb, asks=na, timestamp=now)

    return {
        "btc_prices": btc_prices,
        "now_ts": now,
        "strike_price": 60000.0,
        "slot_expiry_ts": now + 120.0,
        "yes_book": yes_book,
        "no_book": no_book,
        "yes_ob_history": [
            (now - 9, 190.0, 210.0),
            (now - 5, 195.0, 215.0),
            (now - 1, 200.0, 220.0),
        ],
        "question": "BTC Up or Down $60,000",
    }


@pytest.fixture
def v4_model_dir() -> str:
    """Path to the on-disk v4 checkpoint — the real artifact we ship with."""
    root = Path(__file__).resolve().parents[2]
    path = root / "models" / "logreg_v4"
    if not path.exists():
        pytest.skip(f"models/logreg_v4 not present at {path}")
    return str(path)


# ──────────────────────────────────────────────────────────────────────────
# Regression: the bug that motivated this whole session
# ──────────────────────────────────────────────────────────────────────────


def test_v4_model_loads_and_predicts_with_18_features(v4_model_dir):
    """REGRESSION: load real v4 checkpoint, run predict(), verify the
    feature vector matches the 18-feature scaler the model was trained
    with. This is the exact bug that broke v4 live trading."""
    model = LogRegV4Model.load(v4_model_dir)
    assert model.ready, "v4 model failed to load"
    assert model.model_version == "logreg_v4"
    assert len(model.feature_names) == 18
    assert model.feature_names[0] == "time_to_expiry"
    assert "ob_imbalance" in model.feature_names
    assert "ob_cross_imbalance" in model.feature_names

    result = model.predict(_synthetic_snapshot())
    assert result.prob_yes is not None, f"predict failed: {result.feature_status}"
    assert 0.0 <= result.prob_yes <= 1.0
    assert result.feature_status == "ready"
    assert result.model_version == "logreg_v4"
    # The live vector must be exactly 18 features — NOT 22.
    assert len(model.last_features) == 18


def test_v4_model_does_not_compute_v5_features_when_not_declared(v4_model_dir):
    """v5 microstructure features must NOT appear in last_features when
    the loaded model's meta doesn't declare them. This is the forward-
    compatibility gate — skipping this work when unused is a correctness
    property, not just an optimization."""
    model = LogRegV4Model.load(v4_model_dir)
    model.predict(_synthetic_snapshot())
    for f in V5_FEATURES_EXTRA:
        assert f not in model.last_features, (
            f"v5 feature {f} leaked into v4 inference"
        )


# ──────────────────────────────────────────────────────────────────────────
# Startup validation — reject broken checkpoints loudly
# ──────────────────────────────────────────────────────────────────────────


def _make_sklearn_stub(n_features: int):
    """Produces a scaler + model pair that together look trained on N features."""
    scaler = MagicMock()
    scaler.n_features_in_ = n_features
    scaler.transform.side_effect = lambda X: X

    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.coef_ = np.zeros((1, n_features))
    model.intercept_ = np.array([0.0])
    return scaler, model


def test_v4_refuses_load_on_feature_count_mismatch():
    """Meta declares 18 features but scaler was trained with 22 → ready=False.
    This is the defense against the exact bug class we hit."""
    scaler, model = _make_sklearn_stub(n_features=22)
    m = LogRegV4Model(
        model=model,
        scaler=scaler,
        feature_names=list(V3_FEATURES) + list(V4_FEATURES_EXTRA),  # 18
        model_version="logreg_v4_broken",
    )
    assert not m.ready, "model should refuse to load with feature-count mismatch"


def test_v4_refuses_load_on_unknown_feature_in_meta():
    """Meta declares a feature name the loader can't build → ready=False."""
    scaler, model = _make_sklearn_stub(n_features=19)
    m = LogRegV4Model(
        model=model,
        scaler=scaler,
        feature_names=list(V3_FEATURES) + list(V4_FEATURES_EXTRA) + ["typo_feature"],
        model_version="logreg_v4_typo",
    )
    assert not m.ready, "model should refuse to load with unknown feature"


def test_v4_backwards_compat_v3_checkpoint_loads_with_16_features():
    """Legacy v3 checkpoints (no `features` in meta.json) fall back to the
    V3_FEATURES default. This is the backcompat path for old checkpoints
    without explicit feature lists."""
    with tempfile.TemporaryDirectory() as tmp:
        # Write a minimal v3-style checkpoint: meta has no features key.
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.random.RandomState(0).randn(50, 16)
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression().fit(scaler.transform(X), y)

        with open(os.path.join(tmp, "logreg_model.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(tmp, "logreg_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(tmp, "logreg_meta.json"), "w") as f:
            # No `features` key — forces the V3_FEATURES fallback.
            json.dump({"model_version": "logreg_v3"}, f)

        m = LogRegV4Model.load(tmp)
        assert m.ready
        assert m.model_version == "logreg_v3"
        assert len(m.feature_names) == 16
        assert m.feature_names == list(V3_FEATURES)
