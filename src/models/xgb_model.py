"""Model loading, calibration, and live prediction for BTC Up/Down."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - exercised in tests via monkeypatch
    xgb = None

from .feature_builder import build_live_features
from .schema import FEATURE_COLUMNS, build_default_metadata


@dataclass
class PredictionResult:
    """Live prediction result consumed by the strategy."""

    prob_yes: Optional[float]
    prob_no: Optional[float]
    model_version: str
    feature_status: str


class BTCUpDownXGBModel:
    """Load an XGBoost booster plus calibration metadata."""

    def __init__(
        self,
        model_path: str,
        feature_schema_path: str,
        metadata_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_path = model_path
        self.feature_schema_path = feature_schema_path
        self.metadata_path = metadata_path
        self.logger = logger or logging.getLogger(__name__)

        self.metadata: Dict[str, object] = build_default_metadata()
        self.feature_columns = list(FEATURE_COLUMNS)
        self.booster = None
        self.ready = False
        self.load_error: Optional[str] = None
        self._load()

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unversioned"))

    @property
    def thresholds(self) -> Dict[str, float]:
        thresholds = self.metadata.get("thresholds", {})
        return thresholds if isinstance(thresholds, dict) else {}

    def _load(self) -> None:
        self.ready = False
        self.load_error = None

        feature_columns = self._load_feature_schema()
        if feature_columns is not None:
            self.feature_columns = feature_columns

        metadata = self._load_json(self.metadata_path)
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        if xgb is None:
            self.load_error = "xgboost_not_installed"
            self.logger.warning("XGBoost dependency not installed; model service will stay fail-closed")
            return

        if not os.path.exists(self.model_path):
            self.load_error = "model_artifact_missing"
            return

        try:
            booster = xgb.Booster()
            booster.load_model(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive
            self.load_error = f"model_load_failed:{exc.__class__.__name__}"
            self.logger.warning("Failed to load XGBoost model: %s", exc)
            return

        self.booster = booster
        self.ready = True

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        """Predict live probability from a raw snapshot."""
        if not self.ready or self.booster is None:
            return PredictionResult(
                prob_yes=None,
                prob_no=None,
                model_version=self.model_version,
                feature_status=self.load_error or "model_unavailable",
            )

        built = build_live_features(snapshot)
        if not built.ready:
            return PredictionResult(
                prob_yes=None,
                prob_no=None,
                model_version=self.model_version,
                feature_status=built.status,
            )

        prob_yes = self.predict_from_features(built.features)
        if prob_yes is None:
            return PredictionResult(
                prob_yes=None,
                prob_no=None,
                model_version=self.model_version,
                feature_status="predict_failed",
            )

        return PredictionResult(
            prob_yes=prob_yes,
            prob_no=max(0.0, min(1.0, 1.0 - prob_yes)),
            model_version=self.model_version,
            feature_status=built.status,
        )

    def predict_from_features(self, features: Mapping[str, float]) -> Optional[float]:
        """Predict from an already-built fixed feature vector."""
        if not self.ready or self.booster is None:
            return None

        try:
            row = np.asarray([[float(features[name]) for name in self.feature_columns]], dtype=float)
            dmatrix = xgb.DMatrix(row, feature_names=self.feature_columns)
            raw = float(self.booster.predict(dmatrix)[0])
            calibrated = self._apply_calibration(raw)
            return max(0.0, min(1.0, calibrated))
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Prediction failed: %s", exc)
            return None

    def _apply_calibration(self, raw_prob: float) -> float:
        calibration = self.metadata.get("calibration", {})
        if not isinstance(calibration, dict):
            return raw_prob

        edges = calibration.get("bin_edges")
        values = calibration.get("bin_values")
        if not isinstance(edges, list) or not isinstance(values, list) or len(edges) < 2 or len(values) < 1:
            return raw_prob

        raw_prob = max(0.0, min(1.0, raw_prob))
        idx = int(np.searchsorted(edges, raw_prob, side="right") - 1)
        idx = max(0, min(idx, len(values) - 1))
        try:
            return float(values[idx])
        except (TypeError, ValueError):
            return raw_prob

    def _load_feature_schema(self) -> Optional[list]:
        data = self._load_json(self.feature_schema_path)
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            return data
        return None

    def _load_json(self, path: str) -> Optional[object]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to load JSON from %s: %s", path, exc)
            return None

