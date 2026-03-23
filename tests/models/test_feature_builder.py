import time

from src.api.types import OrderBook, OrderBookEntry
from src.models.feature_builder import build_live_features


def _book(token_id: str, bid: float, ask: float, bid_size: float = 100.0, ask_size: float = 90.0) -> OrderBook:
    return OrderBook(
        market_id="mkt",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=bid_size)],
        asks=[OrderBookEntry(price=ask, size=ask_size)],
        tick_size=0.001,
    )


def test_build_live_features_ready_with_indicator_warmup():
    now = time.time()
    btc_prices = [(now - 70 + idx, 100_000.0 + idx * 2.0) for idx in range(71)]
    snapshot = {
        "btc_prices": btc_prices,
        "yes_book": _book("yes", 0.52, 0.54),
        "no_book": _book("no", 0.46, 0.48),
        "yes_history": [(now - 40, 0.50), (now - 5, 0.53)],
        "no_history": [(now - 40, 0.50), (now - 5, 0.47)],
        "question": "Will BTC be above $100,050 at 12:05 PM ET?",
        "slot_expiry_ts": now + 120,
        "now_ts": now,
    }

    result = build_live_features(snapshot)

    assert result.ready is True
    assert result.status == "ready_indicator_warmup"
    assert result.features["btc_mid"] > 0
    assert result.features["moneyness"] != 0
    assert result.features["yes_spread"] > 0
    assert result.features["no_spread"] > 0
    assert result.features["active_bull_gap"] == 0.0
    assert result.features["bull_setup"] == 0.0


def test_build_live_features_fails_closed_without_strike():
    now = time.time()
    snapshot = {
        "btc_prices": [(now - 5, 100_000.0), (now, 100_010.0)],
        "yes_book": _book("yes", 0.50, 0.52),
        "no_book": _book("no", 0.48, 0.50),
        "yes_history": [],
        "no_history": [],
        "question": "Will BTC move higher soon?",
        "slot_expiry_ts": now + 120,
        "now_ts": now,
    }

    result = build_live_features(snapshot)

    assert result.ready is False
    assert result.status == "missing_strike"
