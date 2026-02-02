from datetime import datetime

from src.engine.trading_engine import TradingEngine
from src.api.types import MarketData, OrderBook, OrderBookEntry, Order
from src.strategies.base import Signal


class EngineClient:
    def __init__(self):
        self._balance = 1000.0
        self._markets = [
            MarketData(
                market_id="m1",
                question="Q",
                condition_id="c1",
                outcome_tokens={"YES": "t1", "NO": "t2"},
            )
        ]
        self._order_book = OrderBook(
            market_id="m1",
            token_id="t1",
            bids=[OrderBookEntry(price=0.19, size=100.0)],
            asks=[OrderBookEntry(price=0.21, size=100.0)],
            last_price=0.2,
            timestamp=datetime.utcnow(),
        )

    def get_balance(self):
        return self._balance

    def get_positions(self):
        return []

    def get_markets(self, active=True):
        return list(self._markets)

    def get_order_book(self, token_id):
        return self._order_book

    def place_order(self, market_id, token_id, outcome, side, price, size):
        return Order(
            order_id="engine_order",
            market_id=market_id,
            token_id=token_id,
            outcome=outcome,
            side=side,
            price=price,
            size=size,
            status="PENDING",
            timestamp=datetime.utcnow(),
        )

    def cancel_order(self, order_id):
        return True


def test_trading_engine_execute_signal():
    client = EngineClient()
    engine = TradingEngine(client, paper_trading=False)

    signal = Signal(
        market_id="m1",
        outcome="YES",
        action="BUY",
        confidence=0.9,
        price=0.2,
        size=5.0,
    )
    market_data = {
        "markets": client.get_markets(active=True),
        "order_books": {"t1": client.get_order_book("t1")},
        "price_history": {},
        "positions": [],
        "balance": client.get_balance(),
    }
    engine._execute_signal(signal, market_data)
    assert "engine_order" in engine.active_orders