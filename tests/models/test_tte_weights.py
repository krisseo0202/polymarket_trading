"""Tests for src.models.tte_weights: bucket boundaries + sample weights."""

import numpy as np
import pytest

from src.models.tte_weights import (
    BUCKET_BOUNDS,
    BUCKET_WEIGHTS,
    bucket_names,
    bucket_range,
    tte_bucket,
    tte_series_to_buckets,
    tte_series_to_weights,
    tte_weight,
)


# ---------------------------------------------------------------------------
# Bucket boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tte_s,expected", [
    (0.0,   "very_late"),
    (10.0,  "very_late"),
    (19.999, "very_late"),
    (20.0,  "late"),
    (45.0,  "late"),
    (59.999, "late"),
    (60.0,  "core"),
    (120.0, "core"),
    (179.999, "core"),
    (180.0, "early_mid"),
    (210.0, "early_mid"),
    (239.999, "early_mid"),
    (240.0, "very_early"),
    (290.0, "very_early"),
    (300.999, "very_early"),
])
def test_tte_bucket_boundaries(tte_s, expected):
    assert tte_bucket(tte_s) == expected


def test_tte_bucket_out_of_range_clamps():
    """Clock skew guards: negative → very_late, >301 → very_early."""
    assert tte_bucket(-5.0) == "very_late"
    assert tte_bucket(500.0) == "very_early"


def test_bucket_names_in_display_order():
    """Display order is very_early → very_late (slot start → slot end)."""
    assert bucket_names() == ["very_early", "early_mid", "core", "late", "very_late"]


@pytest.mark.parametrize("name,expected_range", [
    ("very_late",  (0.0,   20.0)),
    ("late",       (20.0,  60.0)),
    ("core",       (60.0,  180.0)),
    ("early_mid",  (180.0, 240.0)),
    ("very_early", (240.0, 301.0)),
])
def test_bucket_range(name, expected_range):
    assert bucket_range(name) == expected_range


def test_bucket_range_rejects_unknown():
    with pytest.raises(ValueError):
        bucket_range("nonsense")


# ---------------------------------------------------------------------------
# Weight lookup
# ---------------------------------------------------------------------------


def test_core_bucket_has_highest_weight():
    """The core trading window must have the highest weight."""
    weights = [BUCKET_WEIGHTS[n] for n in bucket_names()]
    max_weight = max(weights)
    assert BUCKET_WEIGHTS["core"] == max_weight


def test_very_late_is_most_downweighted():
    """Near-resolution rows must be the most aggressively down-weighted."""
    weights = [BUCKET_WEIGHTS[n] for n in bucket_names()]
    min_weight = min(weights)
    assert BUCKET_WEIGHTS["very_late"] == min_weight


def test_tte_weight_returns_correct_bucket_weight():
    assert tte_weight(10.0) == BUCKET_WEIGHTS["very_late"]
    assert tte_weight(100.0) == BUCKET_WEIGHTS["core"]
    assert tte_weight(250.0) == BUCKET_WEIGHTS["very_early"]


# ---------------------------------------------------------------------------
# Vectorized lookups
# ---------------------------------------------------------------------------


def test_tte_series_to_weights_matches_scalar_lookup():
    arr = np.array([5.0, 30.0, 100.0, 200.0, 270.0], dtype=float)
    vec = tte_series_to_weights(arr)
    expected = np.array([tte_weight(x) for x in arr])
    np.testing.assert_allclose(vec, expected)


def test_tte_series_to_buckets_matches_scalar_lookup():
    arr = np.array([5.0, 30.0, 100.0, 200.0, 270.0], dtype=float)
    vec = tte_series_to_buckets(arr)
    expected = [tte_bucket(x) for x in arr]
    assert list(vec) == expected


def test_tte_series_to_weights_handles_out_of_range():
    arr = np.array([-10.0, 0.0, 500.0], dtype=float)
    vec = tte_series_to_weights(arr)
    # Negative clamps to very_late; >=301 clamps to very_early.
    assert vec[0] == BUCKET_WEIGHTS["very_late"]
    assert vec[1] == BUCKET_WEIGHTS["very_late"]
    assert vec[2] == BUCKET_WEIGHTS["very_early"]


def test_mean_weight_approximates_one_on_uniform_tte_grid():
    """Sanity: weights were picked so a typical slot distribution averages ~1.

    If this fails, effective training-set size will shift noticeably when the
    --tte-weighted flag flips, and promote gates may become unstable.
    """
    # Uniform sampling across the 0-300s slot.
    tte = np.linspace(0.0, 300.0, 301)
    w = tte_series_to_weights(tte)
    assert 0.7 <= w.mean() <= 1.3, (
        f"mean weight {w.mean():.3f} drifted; re-tune BUCKET_WEIGHTS"
    )
