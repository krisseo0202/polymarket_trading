import json

import numpy as np

from src.models.schema import FEATURE_COLUMNS
from src.models.xgb_model import BTCUpDownXGBModel
import src.models.xgb_model as xgb_model_module


class _FakeDMatrix:
    def __init__(self, data, feature_names=None):
        self.data = data
        self.feature_names = feature_names


class _FakeBooster:
    def load_model(self, path):
        self.path = path

    def predict(self, dmatrix):
        assert dmatrix.feature_names == FEATURE_COLUMNS
        return np.asarray([0.62], dtype=float)


class _FakeXGBModule:
    Booster = _FakeBooster
    DMatrix = _FakeDMatrix


def test_model_fail_closed_when_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(xgb_model_module, "xgb", _FakeXGBModule)

    feature_path = tmp_path / "features.json"
    meta_path = tmp_path / "meta.json"
    feature_path.write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")
    meta_path.write_text(json.dumps({"model_version": "test"}), encoding="utf-8")

    model = BTCUpDownXGBModel(
        model_path=str(tmp_path / "missing.json"),
        feature_schema_path=str(feature_path),
        metadata_path=str(meta_path),
    )

    assert model.ready is False
    assert model.load_error == "model_artifact_missing"


def test_model_predict_from_features_with_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(xgb_model_module, "xgb", _FakeXGBModule)

    model_path = tmp_path / "model.json"
    feature_path = tmp_path / "features.json"
    meta_path = tmp_path / "meta.json"

    model_path.write_text("{}", encoding="utf-8")
    feature_path.write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "model_version": "test_model",
                "calibration": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "bin_values": [0.4, 0.8],
                },
                "thresholds": {},
            }
        ),
        encoding="utf-8",
    )

    model = BTCUpDownXGBModel(
        model_path=str(model_path),
        feature_schema_path=str(feature_path),
        metadata_path=str(meta_path),
    )

    features = {name: 0.0 for name in FEATURE_COLUMNS}
    prob_yes = model.predict_from_features(features)

    assert model.ready is True
    assert prob_yes == 0.8
