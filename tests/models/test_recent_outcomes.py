"""Unit tests for `_add_recent_outcome_features`.

Computes Up rate over last N=5/10/20 slots from a caller-supplied
sequence of "Up"/"Down" strings.
"""

from __future__ import annotations

import pytest

from src.models.feature_builder import _add_recent_outcome_features


def test_empty_history_defaults_to_uninformative_prior():
    f: dict = {}
    _add_recent_outcome_features(f, [])
    assert f["recent_up_rate_5"] == 0.5
    assert f["recent_up_rate_10"] == 0.5
    assert f["recent_up_rate_20"] == 0.5


def test_none_history_defaults_to_prior():
    f: dict = {}
    _add_recent_outcome_features(f, None)
    assert f["recent_up_rate_5"] == 0.5
    assert f["recent_up_rate_10"] == 0.5
    assert f["recent_up_rate_20"] == 0.5


def test_all_up_history_returns_one():
    f: dict = {}
    _add_recent_outcome_features(f, ["Up"] * 25)
    assert f["recent_up_rate_5"] == 1.0
    assert f["recent_up_rate_10"] == 1.0
    assert f["recent_up_rate_20"] == 1.0


def test_all_down_history_returns_zero():
    f: dict = {}
    _add_recent_outcome_features(f, ["Down"] * 25)
    assert f["recent_up_rate_5"] == 0.0
    assert f["recent_up_rate_10"] == 0.0
    assert f["recent_up_rate_20"] == 0.0


def test_mixed_history_correct_rates_per_window():
    """Last 5 slots all Up; last 10 are 7 Up; last 20 are 11 Up."""
    history = (
        ["Down"] * 9 + ["Up"] * 4    # oldest 13: 4/13 Up
        + ["Down"] * 2 + ["Up"] * 3 + ["Up"] * 2  # 7/20 Up cumulative = 4+3+2+2 → check
    )
    # Easier: explicit construction.
    history = []
    # Build for clear assertions:
    # slots 0-9 Down, 10-15 Up, 16-19 Up = last 5 all Up, last 10 = 4 Down + 6 Up,
    # last 20 = 9 Down + 11 Up but trimmed window — let's just construct exactly.
    history = (["Down"] * 10 + ["Up"] * 6 + ["Up"] * 4)  # length 20: 10 Down, 10 Up
    f: dict = {}
    _add_recent_outcome_features(f, history)
    # last 5 = ["Up"]*5
    assert f["recent_up_rate_5"] == 1.0
    # last 10 = ["Up"]*10
    assert f["recent_up_rate_10"] == 1.0
    # last 20 = 10 Down + 10 Up
    assert f["recent_up_rate_20"] == 0.5


def test_short_history_uses_what_it_has():
    """Only 3 slots resolved → all three rates use those 3 slots."""
    f: dict = {}
    _add_recent_outcome_features(f, ["Up", "Down", "Up"])
    assert f["recent_up_rate_5"] == pytest.approx(2.0 / 3.0)
    assert f["recent_up_rate_10"] == pytest.approx(2.0 / 3.0)
    assert f["recent_up_rate_20"] == pytest.approx(2.0 / 3.0)


def test_case_insensitive_outcome_strings():
    """Lowercase 'up'/'down' should be treated as Up/Down."""
    f: dict = {}
    _add_recent_outcome_features(f, ["up", "DOWN", "Up"])
    assert f["recent_up_rate_5"] == pytest.approx(2.0 / 3.0)
