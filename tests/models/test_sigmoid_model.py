"""Unit tests for BTCSigmoidModel."""

import math
import time
import pytest

from src.models.sigmoid_model import BTCSigmoidModel


def _make_prices(now_ts: float, price: float, count: int = 60, step: float = 5.0):
    """Flat price history from (now_ts - count*step) to now_ts."""
    return [(now_ts - (count - i) * step, price) for i in range(count)]


def _snapshot(price_now=84000.0, strike=84000.0, tte=150.0, price_history=None, td_signal=None):
    now_ts = time.time()
    prices = price_history if price_history is not None else _make_prices(now_ts, price_now)
    snap = {
        "btc_prices": prices,
        "strike_price": strike,
        "slot_expiry_ts": now_ts + tte,
        "now_ts": now_ts,
    }
    if td_signal is not None:
        snap["td_signal"] = td_signal
    return snap


class TestBTCSigmoidModel:
    def setup_method(self):
        self.model = BTCSigmoidModel()

    # ── Sanity: at-the-money, flat history → prob near 0.5 ───────────────────
    def test_atm_flat_returns_near_half(self):
        result = self.model.predict(_snapshot(84000.0, 84000.0))
        assert result.prob_yes is not None
        assert 0.4 < result.prob_yes < 0.6

    # ── Price clearly above strike → prob_yes > 0.5 ──────────────────────────
    def test_above_strike_bullish(self):
        result = self.model.predict(_snapshot(price_now=85000.0, strike=84000.0))
        assert result.prob_yes is not None
        assert result.prob_yes > 0.5

    # ── Price clearly below strike → prob_yes < 0.5 ──────────────────────────
    def test_below_strike_bearish(self):
        result = self.model.predict(_snapshot(price_now=83000.0, strike=84000.0))
        assert result.prob_yes is not None
        assert result.prob_yes < 0.5

    # ── Positive momentum lifts prob relative to flat baseline ───────────────
    def test_positive_momentum_increases_prob(self):
        now_ts = time.time()
        strike = 84000.0
        flat_prices = _make_prices(now_ts, 84000.0)
        # Rising prices: linear climb from 83500 to 84000 over last 5 minutes
        rising_prices = [
            (now_ts - 300 + i * 5, 83500.0 + i * (500.0 / 60))
            for i in range(61)
        ]
        snap_flat = {
            "btc_prices": flat_prices,
            "strike_price": strike,
            "slot_expiry_ts": now_ts + 150,
            "now_ts": now_ts,
        }
        snap_rising = {
            "btc_prices": rising_prices,
            "strike_price": strike,
            "slot_expiry_ts": now_ts + 150,
            "now_ts": now_ts,
        }
        r_flat = self.model.predict(snap_flat)
        r_rising = self.model.predict(snap_rising)
        assert r_rising.prob_yes > r_flat.prob_yes

    # ── Near expiry amplifies the distance signal ─────────────────────────────
    def test_near_expiry_amplifies(self):
        above_far = self.model.predict(_snapshot(price_now=84100.0, strike=84000.0, tte=290.0))
        above_near = self.model.predict(_snapshot(price_now=84100.0, strike=84000.0, tte=5.0))
        assert above_near.prob_yes > above_far.prob_yes

    # ── td_signal positive nudges probability up ──────────────────────────────
    def test_td_signal_positive_increases_prob(self):
        base = self.model.predict(_snapshot(td_signal=0))
        bullish = self.model.predict(_snapshot(td_signal=2))
        assert bullish.prob_yes > base.prob_yes

    # ── Missing strike → None ────────────────────────────────────────────────
    def test_missing_strike_returns_none(self):
        now_ts = time.time()
        snap = {
            "btc_prices": _make_prices(now_ts, 84000.0),
            "strike_price": None,
            "question": "no strike here",
            "slot_expiry_ts": now_ts + 150,
            "now_ts": now_ts,
        }
        result = self.model.predict(snap)
        assert result.prob_yes is None
        assert result.feature_status == "missing_strike"

    # ── Insufficient history → None ──────────────────────────────────────────
    def test_insufficient_history_returns_none(self):
        now_ts = time.time()
        snap = {
            "btc_prices": [(now_ts, 84000.0), (now_ts - 1, 84000.0)],
            "strike_price": 84000.0,
            "slot_expiry_ts": now_ts + 150,
            "now_ts": now_ts,
        }
        result = self.model.predict(snap)
        assert result.prob_yes is None
        assert result.feature_status == "insufficient_btc_history"

    # ── Expired market → None ────────────────────────────────────────────────
    def test_expired_market_returns_none(self):
        now_ts = time.time()
        snap = {
            "btc_prices": _make_prices(now_ts, 84000.0),
            "strike_price": 84000.0,
            "slot_expiry_ts": now_ts - 10,
            "now_ts": now_ts,
        }
        result = self.model.predict(snap)
        assert result.prob_yes is None
        assert result.feature_status == "market_expired"

    # ── prob_yes in [0, 1] ───────────────────────────────────────────────────
    def test_prob_yes_bounded(self):
        for price_now in [80000, 84000, 88000]:
            result = self.model.predict(_snapshot(price_now=price_now, strike=84000.0))
            if result.prob_yes is not None:
                assert 0.0 <= result.prob_yes <= 1.0

    # ── model_version exposed ────────────────────────────────────────────────
    def test_model_version(self):
        assert self.model.model_version == "btc_sigmoid_v1"
        result = self.model.predict(_snapshot())
        assert result.model_version == "btc_sigmoid_v1"
