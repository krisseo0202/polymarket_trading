"""Feature-builder-driven LogReg loader.

Matches the artifact layout produced by ``scripts/train.py``:

    <model_dir>/
        schema.json               # {"features": [...]}
        logreg_model.pkl          # sklearn LogisticRegression
        logreg_scaler.pkl         # sklearn StandardScaler
        logreg_calibrator.pkl     # sklearn IsotonicRegression

Differs from ``LogRegV4Model`` in two important ways:

1. **Features come from the main feature builder**
   (``src.models.feature_builder.build_live_features``), so anything that lives
   in ``schema.FEATURE_COLUMNS`` — including Family A/B/C — is available.

2. **Within-slot path state is managed internally.** The loader owns a
   ``SlotPathState`` and resets it on slot rollover. Each ``predict()`` folds
   the latest BTC tick in and injects ``slot_path_features`` into the snapshot
   before calling ``build_live_features``, so Family C features are live too.

Same ``predict(snapshot) -> PredictionResult`` interface as other loaders —
plugs straight into ``LogRegEdgeStrategy`` via the strategy's
``model_service`` slot.
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
    "max_prob_yes_for_no": 0.46,
    "max_spread_pct": 0.06,
    "exit_edge": -0.01,
    "min_seconds_to_expiry": 10.0,
    "max_seconds_to_expiry": 295.0,
}


class LogRegFBModel:
    """Meta-driven LogReg loader that consumes the new train.py artifact."""

    def __init__(
        self,
        model: Any = None,
        scaler: Any = None,
        calibrator: Any = None,
        feature_names: Optional[List[str]] = None,
        model_version: str = "logreg_fb",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._model = model
        self._scaler = scaler
        self._calibrator = calibrator
        self._feature_names: List[str] = list(feature_names) if feature_names else []
        self._model_version = model_version
        self.logger = logger or logging.getLogger(__name__)

        self.ready = (
            model is not None
            and scaler is not None
            and len(self._feature_names) > 0
        )
        if self.ready:
            expected = getattr(self._scaler, "n_features_in_", len(self._feature_names))
            if expected != len(self._feature_names):
                self.logger.error(
                    "logreg_fb %s feature count mismatch: schema=%d, scaler=%d — disabling",
                    self._model_version, len(self._feature_names), expected,
                )
                self.ready = False

        # Slot-path state is owned by the loader so every predict() call can
        # update it without asking the strategy to know anything new.
        self._slot_state = SlotPathState()
        self._slot_state_ts: int = 0
        self.last_features: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public properties used by LogRegEdgeStrategy
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def thresholds(self) -> Dict[str, float]:
        return dict(_DEFAULT_THRESHOLDS)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls, model_dir: str, logger: Optional[logging.Logger] = None
    ) -> "LogRegFBModel":
        """Load the four artifacts produced by scripts/train.py."""
        log = logger or logging.getLogger(__name__)
        root = Path(model_dir)
        if not root.exists():
            log.warning("logreg_fb: model dir %s does not exist", model_dir)
            return cls(logger=log, model_version=f"logreg_fb:{root.name}")

        try:
            schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
            feature_names = list(schema.get("features") or [])
            with (root / "logreg_model.pkl").open("rb") as f:
                model = pickle.load(f)
            with (root / "logreg_scaler.pkl").open("rb") as f:
                scaler = pickle.load(f)
            calibrator = None
            cal_path = root / "logreg_calibrator.pkl"
            if cal_path.exists():
                with cal_path.open("rb") as f:
                    calibrator = pickle.load(f)
        except Exception as exc:
            log.warning("logreg_fb: failed to load from %s: %s", model_dir, exc)
            return cls(logger=log, model_version=f"logreg_fb:{root.name}")

        return cls(
            model=model,
            scaler=scaler,
            calibrator=calibrator,
            feature_names=feature_names,
            model_version=f"logreg_fb:{root.name}",
            logger=log,
        )

    # ------------------------------------------------------------------
    # Live prediction
    # ------------------------------------------------------------------

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
            x_scaled = self._scaler.transform(x)
            raw_p = float(self._model.predict_proba(x_scaled)[0, 1])
        except Exception as exc:
            self.logger.warning("logreg_fb predict failed: %s", exc)
            return PredictionResult(None, self._model_version, "predict_error")

        prob = raw_p
        if self._calibrator is not None:
            try:
                prob = float(self._calibrator.transform([raw_p])[0])
            except Exception as exc:
                self.logger.warning("logreg_fb calibrator failed: %s", exc)

        prob = max(0.0, min(1.0, prob))
        return PredictionResult(
            prob_yes=prob,
            model_version=self._model_version,
            feature_status=built.status,
        )

