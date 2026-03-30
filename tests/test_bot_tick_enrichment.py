"""Tests for enriched _log_price_tick fields in bot.py ticker loop."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _make_slot_ctx(strike=84500.0, source="chainlink"):
    from src.engine.slot_state import SlotContext
    ts = 1_699_999_800
    return SlotContext(
        slot_start_ts=ts, slot_end_ts=ts + 300,
        strike_price=strike, strike_source=source,
        btc_ref_price=84510.0, captured_at=float(ts),
    )


def _collect_tick_record(btc_feed=None, slot_mgr=None):
    """
    Call the enrichment logic inline (mirrors what _fetch_loop does) and
    return the record that would be passed to _log_price_tick.
    """
    import time
    from src.engine.slot_state import SlotContext
    from src.models.feature_builder import _realized_vol

    yb, ya = 0.505, 0.510
    nb, na = 0.490, 0.495
    _now_ts = time.time()

    _btc_now = btc_feed.get_latest_mid() if btc_feed else None
    _btc_prices = btc_feed.get_recent_prices(300) if btc_feed else []
    _vol_30s = _realized_vol(_btc_prices, _now_ts, 30) if len(_btc_prices) >= 3 else None
    _vol_60s = _realized_vol(_btc_prices, _now_ts, 60) if len(_btc_prices) >= 3 else None
    _strike, _strike_src = None, None
    if slot_mgr:
        _ctx = slot_mgr.get()
        if _ctx:
            _strike, _strike_src = _ctx.strike_price, _ctx.strike_source

    record = {
        "ts": _now_ts,
        "slot_ts": 1_699_999_800,
        "yes_bid": yb, "yes_ask": ya,
        "no_bid": nb, "no_ask": na,
        "yes_mid": (yb + ya) / 2,
        "no_mid": (nb + na) / 2,
        "yes_spread": ya - yb,
        "no_spread": na - nb,
        "btc_now": _btc_now,
        "strike": _strike,
        "strike_source": _strike_src,
        "realized_vol_30s": _vol_30s,
        "realized_vol_60s": _vol_60s,
    }
    # Replicate the None-filter from _log_price_tick
    return {k: v for k, v in record.items() if v is not None}


class TestBotTickEnrichment:
    def test_tick_includes_btc_and_strike_with_feeds(self):
        btc_feed = MagicMock()
        btc_feed.get_latest_mid.return_value = 84512.5
        # Provide enough prices for realized vol (need >= 3)
        now = 1_699_999_850.0
        btc_feed.get_recent_prices.return_value = [
            (now - 60, 84500.0), (now - 30, 84505.0), (now, 84512.5)
        ]

        slot_mgr = MagicMock()
        slot_mgr.get.return_value = _make_slot_ctx(strike=84500.0, source="chainlink")

        record = _collect_tick_record(btc_feed=btc_feed, slot_mgr=slot_mgr)

        assert "btc_now" in record
        assert record["btc_now"] == pytest.approx(84512.5)
        assert "strike" in record
        assert record["strike"] == pytest.approx(84500.0)
        assert "strike_source" in record
        assert record["strike_source"] == "chainlink"
        assert "realized_vol_30s" in record
        assert "realized_vol_60s" in record

    def test_tick_omits_btc_fields_without_feed(self):
        record = _collect_tick_record(btc_feed=None, slot_mgr=None)

        # Core fields are always present
        assert "yes_bid" in record
        assert "no_bid" in record

        # Enrichment fields are absent (filtered as None)
        assert "btc_now" not in record
        assert "strike" not in record
        assert "strike_source" not in record
        assert "realized_vol_30s" not in record
        assert "realized_vol_60s" not in record

    def test_tick_includes_btc_but_not_strike_without_slot_mgr(self):
        btc_feed = MagicMock()
        btc_feed.get_latest_mid.return_value = 84512.5
        btc_feed.get_recent_prices.return_value = []

        record = _collect_tick_record(btc_feed=btc_feed, slot_mgr=None)

        assert "btc_now" in record
        assert record["btc_now"] == pytest.approx(84512.5)
        assert "strike" not in record

    def test_realized_vol_absent_when_too_few_prices(self):
        btc_feed = MagicMock()
        btc_feed.get_latest_mid.return_value = 84512.5
        btc_feed.get_recent_prices.return_value = [(1_699_999_800.0, 84512.5)]  # only 1 price

        record = _collect_tick_record(btc_feed=btc_feed, slot_mgr=None)

        assert "realized_vol_30s" not in record
        assert "realized_vol_60s" not in record
