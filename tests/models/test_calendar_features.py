"""Unit tests for the calendar/cyclical features.

`hour_sin`, `hour_cos`, `is_weekend` are derived from `slot_expiry_ts`
in `feature_builder._add_calendar_features`.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.models.feature_builder import _add_calendar_features


def _ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


def test_midnight_utc_sin_zero_cos_one():
    """At 00:00 UTC, hour=0, so sin(0)=0 and cos(0)=1."""
    f: dict = {}
    _add_calendar_features(f, _ts(2026, 4, 27, 0, 0))
    assert f["hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert f["hour_cos"] == pytest.approx(1.0, abs=1e-9)


def test_six_am_utc_quarter_phase():
    """At 06:00, hour=6/24, so sin=1 (peak), cos=0."""
    f: dict = {}
    _add_calendar_features(f, _ts(2026, 4, 27, 6, 0))
    assert f["hour_sin"] == pytest.approx(1.0, abs=1e-9)
    assert f["hour_cos"] == pytest.approx(0.0, abs=1e-9)


def test_noon_utc_half_phase():
    """At 12:00, hour=12, sin(π)=0, cos(π)=-1."""
    f: dict = {}
    _add_calendar_features(f, _ts(2026, 4, 27, 12, 0))
    assert f["hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert f["hour_cos"] == pytest.approx(-1.0, abs=1e-9)


def test_weekday_flag_monday():
    f: dict = {}
    # 2026-04-27 is a Monday (weekday()=0).
    _add_calendar_features(f, _ts(2026, 4, 27))
    assert f["is_weekend"] == 0.0


def test_weekday_flag_saturday():
    f: dict = {}
    # 2026-04-25 is a Saturday (weekday()=5).
    _add_calendar_features(f, _ts(2026, 4, 25))
    assert f["is_weekend"] == 1.0


def test_weekday_flag_sunday():
    f: dict = {}
    # 2026-04-26 is a Sunday (weekday()=6).
    _add_calendar_features(f, _ts(2026, 4, 26))
    assert f["is_weekend"] == 1.0


def test_fractional_hour_continuous_encoding():
    """At 00:30 UTC, hour=0.5, so sin should be small positive, not 0."""
    f: dict = {}
    _add_calendar_features(f, _ts(2026, 4, 27, 0, 30))
    expected_sin = math.sin(2 * math.pi * 0.5 / 24)
    assert f["hour_sin"] == pytest.approx(expected_sin, abs=1e-9)


def test_zero_or_missing_ts_does_not_crash():
    """Defensive: defaults to epoch (Thursday 1970-01-01)."""
    f: dict = {}
    _add_calendar_features(f, 0.0)
    # No exception. is_weekend=0 because 1970-01-01 is a Thursday.
    assert f["is_weekend"] == 0.0
