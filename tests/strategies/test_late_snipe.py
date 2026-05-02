"""Tests for the late-slot snipe path in LogRegEdgeStrategy.

Covered:
- Disabled by default (no late_snipe config block → no signals)
- Fires when conditions met (YES side and NO side mirror)
- Blocked by low |moneyness| (the "don't fire every time at 0.90" guard)
- Blocked by price out of band
- Blocked by tte too high
- Once-per-slot
- Resets on slot rollover

Snipe is model-agnostic, so we don't need a real model. We give the
strategy a simple mock model that says ready=True; analyze() will run
_check_entry first (it'll skip due to thresholds at 0.5/0.5 prices) and
then fall through to _check_late_snipe.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock


from src.api.types import OrderBook, OrderBookEntry
from src.models.prediction import PredictionResult
from src.strategies.logreg_edge import LogRegEdgeStrategy


def _book(token_id: str, bid: float, ask: float, size: float = 1000.0) -> OrderBook:
    return OrderBook(
        market_id="m",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=size)],
        asks=[OrderBookEntry(price=ask, size=size)],
        tick_size=0.01,
    )


def _mock_model(prob_yes: float = 0.5):
    m = MagicMock()
    m.ready = True
    m.model_version = "test"
    m.feature_names = []
    m.thresholds = {}  # no model-side gates
    m.predict.return_value = PredictionResult(
        prob_yes=prob_yes, model_version="test", feature_status="ready"
    )
    return m


def _strategy(snipe_cfg: dict | None = None, model_prob: float = 0.5):
    cfg = {
        "delta": 0.025,
        "max_spread_pct": 0.05,
        "min_seconds_to_expiry": 50.0,
        "max_seconds_to_expiry": 290.0,
        "min_entry_price": 0.30,
        "max_entry_price": 0.70,
        "min_confidence": 0.0,
        "kelly_fraction": 0.10,
        "min_position_size_usdc": 5.0,
        "max_position_size_usdc": 50.0,
    }
    if snipe_cfg is not None:
        cfg["late_snipe"] = snipe_cfg
    s = LogRegEdgeStrategy(config=cfg, btc_feed=None, model_service=_mock_model(model_prob))
    s.set_tokens("m", "yes_t", "no_t")
    return s


def _market(
    yes_ask: float = 0.92, no_ask: float = 0.08,
    btc_mid: float = 100_500.0, strike: float = 100_000.0,
    tte: float = 20.0, spread: float = 0.01,
):
    """Build a market_data dict where the snipe COULD fire on YES if
    moneyness is high enough. NO side is symmetric (no_ask=0.08 implies
    YES dominant)."""
    yes_bid = max(0.01, yes_ask - spread)
    no_bid = max(0.01, no_ask - spread)
    return {
        "order_books": {
            "yes_t": _book("yes_t", yes_bid, yes_ask),
            "no_t": _book("no_t", no_bid, no_ask),
        },
        "positions": [],
        "balance": 1000.0,
        "slot_expiry_ts": time.time() + tte,
        "strike_price": strike,
        "btc_mid": btc_mid,
        "question": "Bitcoin Up or Down $100,000",
    }


# ---------------------------------------------------------------------------
# 1. Disabled by default
# ---------------------------------------------------------------------------


def test_snipe_disabled_by_default():
    """Without a late_snipe config block, strategy never fires snipe trades."""
    s = _strategy(snipe_cfg=None)
    # tte=20 + yes_ask=0.92 + moneyness=+50bps would be a perfect snipe.
    signals = s.analyze(_market(yes_ask=0.92, btc_mid=100_500.0, tte=20.0))
    assert signals == []


# ---------------------------------------------------------------------------
# 2-3. Fires when conditions met (YES + NO sides)
# ---------------------------------------------------------------------------


def test_snipe_fires_yes_at_092_with_high_moneyness():
    """Sweet spot: YES at 0.92, BTC 50bps above strike, 20s left → BUY YES."""
    s = _strategy(snipe_cfg={"enabled": True, "size_usdc": 5.0})
    signals = s.analyze(_market(
        yes_ask=0.92, btc_mid=100_500.0, strike=100_000.0, tte=20.0,
    ))
    assert len(signals) == 1
    sig = signals[0]
    assert sig.outcome == "YES"
    assert sig.action == "BUY"
    assert "late_snipe" in sig.reason
    assert sig.size > 0
    # $5 / 0.92 ≈ 5.43 shares
    assert abs(sig.size - 5.0 / 0.92) < 0.05


def test_snipe_fires_no_side_with_negative_moneyness():
    """Mirror of YES snipe: market thinks NO will win (NO ask at 0.92),
    BTC is 50bps below strike, 20s left → BUY NO."""
    s = _strategy(snipe_cfg={"enabled": True, "size_usdc": 5.0})
    # Flipped book: YES is the cheap side (0.08-0.09), NO is the expensive
    # side (0.91-0.92). The bot snipes NO at 0.92.
    signals = s.analyze(_market(
        yes_ask=0.09, no_ask=0.92, btc_mid=99_500.0, strike=100_000.0, tte=20.0,
    ))
    assert len(signals) == 1
    assert signals[0].outcome == "NO"
    assert signals[0].action == "BUY"


# ---------------------------------------------------------------------------
# 4. The "don't fire every time at 0.90" guard
# ---------------------------------------------------------------------------


def test_snipe_blocked_when_moneyness_too_small():
    """YES at 0.92 but BTC only 10bps above strike → reversal too plausible, skip.
    BTC at 100_100 on a 100_000 strike is exactly +10bps."""
    s = _strategy(snipe_cfg={"enabled": True, "min_moneyness_bps": 50.0})
    signals = s.analyze(_market(
        yes_ask=0.92, btc_mid=100_100.0, strike=100_000.0, tte=20.0,
    ))
    assert signals == []
    assert "snipe_no_match" in s.last_skip_reason
    assert "+10bps" in s.last_skip_reason


# ---------------------------------------------------------------------------
# 5. Price out of band
# ---------------------------------------------------------------------------


def test_snipe_blocked_when_price_below_floor():
    """YES at 0.85 — below the snipe floor of 0.90 — must skip."""
    s = _strategy(snipe_cfg={
        "enabled": True, "min_entry_price": 0.90, "max_entry_price": 0.95,
    })
    signals = s.analyze(_market(
        yes_ask=0.85, btc_mid=100_500.0, strike=100_000.0, tte=20.0,
    ))
    assert signals == []
    assert "snipe_no_match" in s.last_skip_reason


def test_snipe_blocked_when_price_above_ceiling():
    """YES at 0.97 — above the snipe ceiling of 0.95 — must skip.
    (At 0.97, the residual upside is only 3 cents; the trade is no longer
    asymmetric enough to be worth the slippage risk.)"""
    s = _strategy(snipe_cfg={
        "enabled": True, "min_entry_price": 0.90, "max_entry_price": 0.95,
    })
    signals = s.analyze(_market(
        yes_ask=0.97, btc_mid=100_500.0, strike=100_000.0, tte=20.0,
    ))
    assert signals == []


# ---------------------------------------------------------------------------
# 6. TTE too high
# ---------------------------------------------------------------------------


def test_snipe_blocked_when_tte_too_high():
    """tte=45s — outside the snipe window of ≤30s — must skip."""
    s = _strategy(snipe_cfg={"enabled": True, "max_tte_s": 30.0})
    signals = s.analyze(_market(
        yes_ask=0.92, btc_mid=100_500.0, strike=100_000.0, tte=45.0,
    ))
    assert signals == []
    assert "snipe_tte" in s.last_skip_reason


# ---------------------------------------------------------------------------
# 7. Spread gate
# ---------------------------------------------------------------------------


def test_snipe_blocked_when_spread_too_wide():
    """Snipe spread cap is 3% by default. YES bid 0.85 / ask 0.92 → 8% spread → skip."""
    s = _strategy(snipe_cfg={"enabled": True, "max_spread_pct": 0.03})
    signals = s.analyze(_market(
        yes_ask=0.92, btc_mid=100_500.0, strike=100_000.0, tte=20.0, spread=0.07,
    ))
    assert signals == []
    assert "snipe_spread_wide" in s.last_skip_reason


# ---------------------------------------------------------------------------
# 8. Once per slot, resets on rollover
# ---------------------------------------------------------------------------


def test_snipe_fires_only_once_per_slot():
    """Same slot, two analyze() calls → only the first fires.

    Reuses the same market dict so slot_expiry_ts (and therefore slot_ts)
    is identical across calls. Without the once-per-slot guard, the
    _auto_recover_position reconciliation would clear the optimistic
    position state (since the test inventory is empty) and the snipe
    would re-fire."""
    s = _strategy(snipe_cfg={"enabled": True})
    market = _market(yes_ask=0.92, btc_mid=100_500.0, strike=100_000.0, tte=20.0)

    signals1 = s.analyze(market)
    assert len(signals1) == 1

    # Same market dict → same slot_ts. Reconciliation will clear the
    # optimistic active_token_id since `positions` is empty, exposing the
    # snipe-guard as the only thing that should still block re-entry.
    signals2 = s.analyze(market)
    assert signals2 == []
    assert "snipe_already_fired_this_slot" in s.last_skip_reason


def test_snipe_resets_on_slot_rollover():
    """When slot_ts changes (new slot), the once-per-slot guard releases."""
    s = _strategy(snipe_cfg={"enabled": True})
    market = _market(yes_ask=0.92, btc_mid=100_500.0, strike=100_000.0, tte=20.0)
    signals1 = s.analyze(market)
    assert len(signals1) == 1
    fired_slot = s._snipe_fired_for_slot
    assert fired_slot is not None

    # Simulate slot rollover: pretend the strategy was last fired for an
    # OLDER slot than the current one. The current slot_ts (computed from
    # market["slot_expiry_ts"]) differs from `fired_slot - 300` so the
    # guard releases and the snipe fires again. We also clear the
    # optimistic position state so the in_position guard doesn't mask
    # what we're testing.
    s._snipe_fired_for_slot = fired_slot - 300
    s.active_token_id = None
    s.entry_price = None
    s.entry_size = None

    signals2 = s.analyze(market)
    assert len(signals2) == 1, "new slot must allow another snipe"
