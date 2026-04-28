"""Unit tests for `_add_ut_trend_disagreement`.

The feature is 1.0 when 1m and 5m UT trends point opposite directions
AND both are non-zero (cold timeframes don't count as disagreement).
"""

from __future__ import annotations

from src.models.feature_builder import _add_ut_trend_disagreement


def test_agree_both_up():
    f = {"ut_1m_trend": 1.0, "ut_5m_trend": 1.0}
    _add_ut_trend_disagreement(f)
    assert f["ut_trend_disagreement"] == 0.0


def test_agree_both_down():
    f = {"ut_1m_trend": -1.0, "ut_5m_trend": -1.0}
    _add_ut_trend_disagreement(f)
    assert f["ut_trend_disagreement"] == 0.0


def test_disagree_1m_up_5m_down():
    f = {"ut_1m_trend": 1.0, "ut_5m_trend": -1.0}
    _add_ut_trend_disagreement(f)
    assert f["ut_trend_disagreement"] == 1.0


def test_disagree_1m_down_5m_up():
    f = {"ut_1m_trend": -1.0, "ut_5m_trend": 1.0}
    _add_ut_trend_disagreement(f)
    assert f["ut_trend_disagreement"] == 1.0


def test_one_cold_does_not_disagree():
    """If 1m hasn't warmed (=0), no disagreement."""
    f = {"ut_1m_trend": 0.0, "ut_5m_trend": -1.0}
    _add_ut_trend_disagreement(f)
    assert f["ut_trend_disagreement"] == 0.0


def test_both_cold_no_disagreement():
    f = {"ut_1m_trend": 0.0, "ut_5m_trend": 0.0}
    _add_ut_trend_disagreement(f)
    assert f["ut_trend_disagreement"] == 0.0


def test_missing_keys_default_to_zero():
    """If multi-TF features were never populated, no crash, no signal."""
    f: dict = {}
    _add_ut_trend_disagreement(f)
    assert f["ut_trend_disagreement"] == 0.0
