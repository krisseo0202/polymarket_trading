"""Time-to-expiry (TTE) bucket definitions + sample weights for training.

The 5-min market has very different predictability characteristics along the
slot. Treating every snapshot equally means the loss is dominated by easy,
near-resolution rows (where BTC has basically no time to move and the outcome
is nearly deterministic). That hurts the model where it actually needs to
perform — the 60-180s window where we trade.

These buckets + weights bias training toward the tradable window.

## Buckets

- ``very_early``  (240-300s): Family C features still warming up. Some signal.
- ``early_mid``   (180-240s): entry-eligible window start; book settled.
- ``core``        (60-180s):  prime trading window. Emphasize.
- ``late``        (20-60s):   execution risk high, spreads tight. Down-weight.
- ``very_late``   (0-20s):    outcome nearly locked. Near-trivial. Down-weight hard.

Validate with the probe's bucket analysis (`scripts/feature_probe.py`). The
report prints per-bucket Brier, log loss, calibration error, mean spread, and
realized edge. If ``core`` is NOT the best tradable-edge bucket on real data,
re-tune these weights.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


# (name, lo_inclusive, hi_exclusive). 301 caps past 300 to catch clock skew.
BUCKET_BOUNDS: List[tuple] = [
    ("very_late",  0.0,   20.0),
    ("late",       20.0,  60.0),
    ("core",       60.0,  180.0),
    ("early_mid",  180.0, 240.0),
    ("very_early", 240.0, 301.0),
]


# Mean ≈ 1.0 across a slot with evenly-distributed snapshot times.
BUCKET_WEIGHTS: Dict[str, float] = {
    "very_early": 0.6,
    "early_mid":  1.2,
    "core":       1.5,
    "late":       0.5,
    "very_late":  0.1,
}


def tte_bucket(tte_s: float) -> str:
    for name, lo, hi in BUCKET_BOUNDS:
        if lo <= tte_s < hi:
            return name
    return "very_late" if tte_s < 0 else "very_early"


def tte_weight(tte_s: float) -> float:
    return BUCKET_WEIGHTS.get(tte_bucket(tte_s), 1.0)


def bucket_names() -> List[str]:
    """Bucket names in display order (very_early → very_late)."""
    return ["very_early", "early_mid", "core", "late", "very_late"]


def bucket_range(name: str) -> tuple:
    for n, lo, hi in BUCKET_BOUNDS:
        if n == name:
            return (lo, hi)
    raise ValueError(f"unknown bucket: {name}")


def tte_series_to_weights(tte: np.ndarray) -> np.ndarray:
    out = np.ones_like(tte, dtype=float)
    for name, lo, hi in BUCKET_BOUNDS:
        mask = (tte >= lo) & (tte < hi)
        out[mask] = BUCKET_WEIGHTS[name]
    out[tte < 0] = BUCKET_WEIGHTS["very_late"]
    out[tte >= 301.0] = BUCKET_WEIGHTS["very_early"]
    return out


def tte_series_to_buckets(tte: np.ndarray) -> np.ndarray:
    out = np.full(tte.shape, "very_late", dtype=object)
    for name, lo, hi in BUCKET_BOUNDS:
        mask = (tte >= lo) & (tte < hi)
        out[mask] = name
    out[tte >= 301.0] = "very_early"
    return out
