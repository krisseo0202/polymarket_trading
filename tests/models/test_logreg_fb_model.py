"""Smoke tests for LogRegFBModel — the feature-builder-driven LogReg loader."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.api.types import OrderBook, OrderBookEntry
from src.models.logreg_fb_model import LogRegFBModel


def _write_artifacts(model_dir: Path, features: List[str]) -> None:
    """Produce a valid artifact layout using a tiny fitted pipeline."""
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, len(features)))
    y = (X[:, 0] > 0).astype(float)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    raw_p = model.predict_proba(scaler.transform(X))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y)

    (model_dir / "schema.json").write_text(json.dumps({"features": features}))
    with (model_dir / "logreg_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    with (model_dir / "logreg_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    with (model_dir / "logreg_calibrator.pkl").open("wb") as f:
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
    model = LogRegFBModel.load(str(tmp_path))
    assert model.ready
    assert model.feature_names == ["btc_mid", "moneyness", "slot_drift_bps"]


def test_load_missing_dir_returns_not_ready(tmp_path):
    model = LogRegFBModel.load(str(tmp_path / "missing"))
    assert not model.ready


def test_predict_returns_bounded_probability(tmp_path):
    _write_artifacts(tmp_path, ["btc_mid", "moneyness", "slot_drift_bps"])
    model = LogRegFBModel.load(str(tmp_path))
    prediction = model.predict(_snapshot(time.time()))
    assert prediction.prob_yes is not None
    assert 0.0 <= prediction.prob_yes <= 1.0
    assert prediction.feature_status in {"ready", "ready_indicator_warmup", "ready_indicator_error"}


def test_slot_path_state_resets_on_new_slot(tmp_path):
    _write_artifacts(
        tmp_path,
        ["slot_high_excursion_bps", "slot_low_excursion_bps", "slot_drift_bps"],
    )
    model = LogRegFBModel.load(str(tmp_path))

    now = time.time()
    # Slot 1: BTC ranges 100_000 → 100_050.
    model.predict(_snapshot(now, strike=100_000.0, btc_drift=50.0))
    first_slot_ts = model._slot_state_ts
    assert first_slot_ts > 0
    assert model._slot_state.slot_max > 0

    # Slot 2: new slot boundary (+300s), different price regime.
    model.predict(_snapshot(now + 300, strike=101_000.0, btc_drift=10.0))
    second_slot_ts = model._slot_state_ts
    assert second_slot_ts == first_slot_ts + 300
    # State must be anchored to the new slot (slot_ts field aligns), not the old one.
    assert model._slot_state.slot_ts == second_slot_ts
    # And the new max tracks the new slot's price regime (≥101_000), confirming
    # we didn't keep folding slot-1 prices into state.
    assert model._slot_state.slot_max >= 101_000.0
    assert model._slot_state.slot_min >= 101_000.0


def test_predict_surfaces_feature_status_when_unready(tmp_path):
    _write_artifacts(tmp_path, ["btc_mid"])
    model = LogRegFBModel.load(str(tmp_path))
    # Snapshot missing btc_prices → build_live_features should fail early.
    prediction = model.predict({"yes_book": _book(0.5, 0.5), "no_book": _book(0.5, 0.5)})
    assert prediction.prob_yes is None
    assert prediction.feature_status == "insufficient_btc_history"


def test_load_on_real_trained_artifact_if_present():
    """Smoke-test against the artifact produced by scripts/train.py if it exists."""
    artifact = Path("models/week1_v2_realfills")
    if not artifact.exists():
        pytest.skip("trained artifact not present")
    model = LogRegFBModel.load(str(artifact))
    assert model.ready
    assert len(model.feature_names) > 0
    # All declared features must be in the main schema so build_live_features can produce them.
    from src.models.schema import FEATURE_COLUMNS
    missing = [f for f in model.feature_names if f not in FEATURE_COLUMNS]
    assert not missing, f"artifact references features missing from schema: {missing}"
