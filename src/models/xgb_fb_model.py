"""XGBoost loader mirroring LogRegFBModel.

Artifact layout produced by ``scripts/train.py``:

    <model_dir>/
        schema.json           # {"features": [...]}
        xgb_model.json        # XGBClassifier booster
        xgb_calibrator.pkl    # sklearn IsotonicRegression (optional)

Same ``predict(snapshot) -> PredictionResult`` interface as the LogReg
loaders, so it drops into ``LogRegEdgeStrategy`` via the strategy's
``model_service`` slot without any strategy-side changes. Slot-path state
is managed internally so Family C features are live.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .feature_builder import build_live_features
from .prediction import PredictionResult
from .slot_path_state import SlotPathState, advance_from_snapshot, features_from_snapshot


_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_edge": 0.05,
    "min_prob_yes": 0.54,
    # Asymmetric: NO-side calibration was anti-calibrated and the only
    # profitable bucket was p_hat ≤ 0.32. See tasks/postmortem_2026-04-25.md.
    "max_prob_yes_for_no": 0.32,
    "max_spread_pct": 0.06,
    "exit_edge": -0.01,
    "min_seconds_to_expiry": 10.0,
    "max_seconds_to_expiry": 295.0,
}


class XGBFBModel:
    """Meta-driven XGB loader that consumes the scripts/train.py artifact."""

    def __init__(
        self,
        model: Any = None,
        calibrator: Any = None,
        feature_names: Optional[List[str]] = None,
        model_version: str = "xgb_fb",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._model = model
        self._calibrator = calibrator
        self._feature_names: List[str] = list(feature_names) if feature_names else []
        self._model_version = model_version
        self.logger = logger or logging.getLogger(__name__)

        self.ready = model is not None and len(self._feature_names) > 0

        self._slot_state = SlotPathState()
        self._slot_state_ts: int = 0
        self.last_features: Dict[str, float] = {}

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def thresholds(self) -> Dict[str, float]:
        return dict(_DEFAULT_THRESHOLDS)

    @classmethod
    def load(
        cls, model_dir: str, logger: Optional[logging.Logger] = None
    ) -> "XGBFBModel":
        log = logger or logging.getLogger(__name__)
        root = Path(model_dir)
        if not root.exists():
            log.warning("xgb_fb: model dir %s does not exist", model_dir)
            return cls(logger=log, model_version=f"xgb_fb:{root.name}")

        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            log.error("xgb_fb: xgboost not installed — %s", exc)
            return cls(logger=log, model_version=f"xgb_fb:{root.name}")

        try:
            schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
            feature_names = list(schema.get("features") or [])
            model = XGBClassifier()
            model.load_model(str(root / "xgb_model.json"))
            calibrator = None
            cal_path = root / "xgb_calibrator.pkl"
            if cal_path.exists():
                with cal_path.open("rb") as f:
                    calibrator = pickle.load(f)
        except Exception as exc:
            log.warning("xgb_fb: failed to load from %s: %s", model_dir, exc)
            return cls(logger=log, model_version=f"xgb_fb:{root.name}")

        return cls(
            model=model,
            calibrator=calibrator,
            feature_names=feature_names,
            model_version=f"xgb_fb:{root.name}",
            logger=log,
        )

    def predict(self, snapshot: Mapping[str, object]) -> PredictionResult:
        if not self.ready:
            return PredictionResult(None, self._model_version, "model_not_loaded")

        self._slot_state_ts = advance_from_snapshot(
            self._slot_state, self._slot_state_ts, snapshot,
        )
        snapshot_with_path = dict(snapshot)
        snapshot_with_path["slot_path_features"] = features_from_snapshot(
            self._slot_state, snapshot,
        )

        built = build_live_features(snapshot_with_path)
        if not built.ready:
            return PredictionResult(None, self._model_version, built.status)

        features = built.features
        self.last_features = features

        try:
            x = np.asarray(
                [[float(features.get(name, 0.0) or 0.0) for name in self._feature_names]],
                dtype=float,
            )
            raw_p = float(self._model.predict_proba(x)[0, 1])
        except Exception as exc:
            self.logger.warning("xgb_fb predict failed: %s", exc)
            return PredictionResult(None, self._model_version, "predict_error")

        prob = raw_p
        if self._calibrator is not None:
            try:
                prob = float(self._calibrator.transform([raw_p])[0])
            except Exception as exc:
                self.logger.warning("xgb_fb calibrator failed: %s", exc)

        prob = max(0.0, min(1.0, prob))
        return PredictionResult(
            prob_yes=prob,
            model_version=self._model_version,
            feature_status=built.status,
        )

