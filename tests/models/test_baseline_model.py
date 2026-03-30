import time

from src.api.types import OrderBook, OrderBookEntry
from src.models.baseline_model import BTCUpDownBaselineModel


def _book(token_id: str, bid: float, ask: float) -> OrderBook:
    return OrderBook(
        market_id="mkt",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=100.0)],
        asks=[OrderBookEntry(price=ask, size=120.0)],
        tick_size=0.001,
    )


def test_baseline_model_returns_probabilities_and_edges():
    now = time.time()
    btc_prices = [(now - 60 + idx, 100_000.0 + idx * 15.0) for idx in range(61)]
    model = BTCUpDownBaselineModel()

    result = model.predict(
        {
            "btc_prices": btc_prices,
            "yes_book": _book("yes", 0.54, 0.56),
            "no_book": _book("no", 0.43, 0.45),
            "strike_price": 100_050.0,
            "slot_expiry_ts": now + 120.0,
            "now_ts": now,
        }
    )

    assert result.feature_status == "ready"
    assert result.prob_yes is not None
    assert result.prob_no is not None
    assert result.edge_yes is not None
    assert result.edge_no is not None
    assert 0.0 <= result.prob_yes <= 1.0
    assert 0.0 <= result.prob_no <= 1.0
    assert abs(result.prob_yes + result.prob_no - 1.0) < 1e-9
    assert result.edge_yes == result.prob_yes - 0.56
    assert result.edge_no == result.prob_no - 0.45


def test_baseline_model_prob_yes_drops_when_spot_below_strike():
    now = time.time()
    btc_prices = [(now - 60 + idx, 99_500.0 + idx * 1.5) for idx in range(61)]
    model = BTCUpDownBaselineModel()

    result = model.predict(
        {
            "btc_prices": btc_prices,
            "yes_book": _book("yes", 0.52, 0.54),
            "no_book": _book("no", 0.46, 0.48),
            "strike_price": 100_200.0,
            "slot_expiry_ts": now + 90.0,
            "now_ts": now,
        }
    )

    assert result.prob_yes is not None
    assert result.prob_yes < 0.5
    assert result.prob_no > 0.5


def test_baseline_model_fails_closed_without_strike():
    now = time.time()
    btc_prices = [(now - 10 + idx, 100_000.0 + idx) for idx in range(11)]
    model = BTCUpDownBaselineModel()

    result = model.predict(
        {
            "btc_prices": btc_prices,
            "yes_book": _book("yes", 0.50, 0.52),
            "no_book": _book("no", 0.48, 0.50),
            "question": "Will BTC move higher soon?",
            "slot_expiry_ts": now + 120.0,
            "now_ts": now,
        }
    )

    assert result.prob_yes is None
    assert result.feature_status == "missing_strike"
