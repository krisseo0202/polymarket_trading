"""Smoke tests for XGBFBModel — the feature-builder-driven XGB loader."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from src.api.types import OrderBook, OrderBookEntry
from src.models.xgb_fb_model import XGBFBModel


def _write_artifacts(model_dir: Path, features: List[str]) -> None:
    """Produce a valid artifact layout using a tiny fitted XGB booster."""
    from xgboost import XGBClassifier

    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, len(features)))
    y = (X[:, 0] > 0).astype(int)

    model = XGBClassifier(
        max_depth=3, n_estimators=20, eval_metric="logloss",
        tree_method="hist", verbosity=0,
    ).fit(X, y)
    raw_p = model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y)

    (model_dir / "schema.json").write_text(json.dumps({"features": features}))
    model.save_model(str(model_dir / "xgb_model.json"))
    with (model_dir / "xgb_calibrator.pkl").open("wb") as f:
        pickle.dump(calibrator, f)


def _book(bid: float, ask: float) -> OrderBook:
    return OrderBook(
        market_id="m",
        token_id="t",
        bids=[OrderBookEntry(price=bid, size=100.0)],
        asks=[OrderBookEntry(price=ask, size=100.0)],
        tick_size=0.001,
    )


def _snapshot(now: float, strike: float = 100_000.0, btc_drift: float = 50.0) -> dict:
    btc_prices = [(now - 100 + i, strike + (i / 100.0) * btc_drift) for i in range(101)]
    return {
        "btc_prices": btc_prices,
        "yes_book": _book(0.52, 0.54),
        "no_book": _book(0.46, 0.48),
        "yes_history": [],
        "no_history": [],
        "question": "",
        "strike_price": strike,
        "slot_expiry_ts": now + 120,
        "now_ts": now,
    }


def test_load_ready_from_valid_artifacts(tmp_path):
    _write_artifacts(tmp_path, ["btc_mid", "moneyness", "slot_drift_bps"])
    model = XGBFBModel.load(str(tmp_path))
    assert model.ready
    assert model.feature_names == ["btc_mid", "moneyness", "slot_drift_bps"]


def test_load_missing_dir_returns_not_ready(tmp_path):
    model = XGBFBModel.load(str(tmp_path / "missing"))
    assert not model.ready


def test_predict_returns_bounded_probability(tmp_path):
    _write_artifacts(tmp_path, ["btc_mid", "moneyness", "slot_drift_bps"])
    model = XGBFBModel.load(str(tmp_path))
    prediction = model.predict(_snapshot(time.time()))
    assert prediction.prob_yes is not None
    assert 0.0 <= prediction.prob_yes <= 1.0


def test_load_on_real_trained_artifact_if_present():
    """Smoke-test against models/signed_v1_trim if it exists."""
    artifact = Path("models/signed_v1_trim")
    if not artifact.exists():
        pytest.skip("trained artifact not present")
    model = XGBFBModel.load(str(artifact))
    assert model.ready
    assert len(model.feature_names) > 0
    # All declared features must be in the main schema so build_live_features
    # can produce them.
    from src.models.schema import FEATURE_COLUMNS
    missing = [f for f in model.feature_names if f not in FEATURE_COLUMNS]
    assert not missing, f"artifact references features missing from schema: {missing}"
