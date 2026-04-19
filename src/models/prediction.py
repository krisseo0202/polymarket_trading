"""Shared prediction result type used by all probability models."""

from __future__ import annotations

from typing import Optional


class PredictionResult:
    """Live prediction result consumed by strategies."""

    __slots__ = ("prob_yes", "model_version", "feature_status", "edge_yes", "edge_no")

    def __init__(
        self,
        prob_yes: Optional[float],
        model_version: str,
        feature_status: str,
        edge_yes: Optional[float] = None,
        edge_no: Optional[float] = None,
    ):
        # prob_no is derived as 1 - prob_yes; no separate storage.
        self.prob_yes = prob_yes
        self.model_version = model_version
        self.feature_status = feature_status
        self.edge_yes = edge_yes
        self.edge_no = edge_no

    @property
    def prob_no(self) -> Optional[float]:
        if self.prob_yes is None:
            return None
        return max(0.0, min(1.0, 1.0 - self.prob_yes))
