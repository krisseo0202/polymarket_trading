"""Tests for kelly_fraction()."""

import pytest
from src.utils.kelly import kelly_fraction


def test_positive_edge():
    # p=0.6, q=0.4, odds=1.0 -> f = (0.6*1 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 0.4, odds=1.0, max_fraction=1.0) == pytest.approx(0.2)


def test_no_edge_returns_zero():
    # p=0.5, q=0.5, odds=1.0 -> f = 0
    assert kelly_fraction(0.5, 0.5, odds=1.0) == 0.0


def test_negative_edge_returns_zero():
    # p=0.3, q=0.7, odds=1.0 -> f = -0.4, clamped to 0
    assert kelly_fraction(0.3, 0.7, odds=1.0) == 0.0


def test_clamped_to_max_fraction():
    # Large edge but max_fraction=0.05
    assert kelly_fraction(0.9, 0.1, odds=1.0, max_fraction=0.05) == pytest.approx(0.05)


def test_higher_odds():
    # p=0.4, q=0.6, odds=3.0 -> f = (0.4*3 - 0.6)/3 = 0.6/3 = 0.2
    assert kelly_fraction(0.4, 0.6, odds=3.0, max_fraction=1.0) == pytest.approx(0.2)


def test_invalid_odds_raises():
    with pytest.raises(ValueError, match="odds must be positive"):
        kelly_fraction(0.5, 0.5, odds=0.0)


def test_default_max_fraction():
    # p=0.9, q=0.1, odds=1.0 -> f=0.8, clamped to default 0.05
    assert kelly_fraction(0.9, 0.1, odds=1.0) == pytest.approx(0.05)
