import logging
import time

from bot import _execute_signals
from src.api.client import PolymarketClient
from src.api.types import OrderBook, OrderBookEntry, Position
from src.engine.inventory import InventoryState
from src.engine.risk_manager import RiskManager
from src.engine.state_store import BotState
from src.models.xgb_model import PredictionResult
from src.strategies.btc_updown_xgb import BTCUpDownXGBStrategy


class _FakeFeed:
    def __init__(self, healthy=True):
        now = time.time()
        self._healthy = healthy
        self._prices = [(now - 120 + idx, 100_000.0 + idx * 1.5) for idx in range(121)]

    def is_healthy(self):
        return self._healthy

    def get_recent_prices(self, window_s=300):
        cutoff = time.time() - window_s
        return [(ts, price) for ts, price in self._prices if ts >= cutoff]


class _FakeModelService:
    def __init__(self, prob_yes=None, prob_no=None, feature_status="ready"):
        self.prob_yes = prob_yes
        self.prob_no = prob_no
        self.feature_status = feature_status
        self.thresholds = {}
        self.model_version = "test_model"

    def predict(self, snapshot):
        return PredictionResult(
            prob_yes=self.prob_yes,
            prob_no=self.prob_no,
            model_version=self.model_version,
            feature_status=self.feature_status,
        )


def _book(token_id: str, bid: float, ask: float, bid_size: float = 100.0, ask_size: float = 100.0) -> OrderBook:
    return OrderBook(
        market_id="mkt",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=bid_size)],
        asks=[OrderBookEntry(price=ask, size=ask_size)],
        tick_size=0.001,
    )


def _market_data(yes_book, no_book, positions=None, seconds_to_expiry=120):
    return {
        "order_books": {"yes": yes_book, "no": no_book},
        "positions": positions or [],
        "question": "Will BTC be above $100,050 at 12:05 PM ET?",
        "strike_price": 100_050.0,
        "slot_expiry_ts": time.time() + seconds_to_expiry,
    }


def _strategy(prob_yes=None, prob_no=None, healthy=True, feature_status="ready"):
    strategy = BTCUpDownXGBStrategy(
        config={},
        btc_feed=_FakeFeed(healthy=healthy),
        model_service=_FakeModelService(prob_yes=prob_yes, prob_no=prob_no, feature_status=feature_status),
        logger=logging.getLogger("test"),
    )
    strategy.set_tokens("mkt", "yes", "no")
    return strategy


def test_buy_yes_signal():
    strategy = _strategy(prob_yes=0.62, prob_no=0.38)
    signals = strategy.analyze(_market_data(_book("yes", 0.53, 0.55), _book("no", 0.60, 0.62)))

    assert len(signals) == 1
    assert signals[0].action == "BUY"
    assert signals[0].outcome == "YES"


def test_buy_no_signal():
    strategy = _strategy(prob_yes=0.40, prob_no=0.60)
    signals = strategy.analyze(_market_data(_book("yes", 0.58, 0.60), _book("no", 0.51, 0.53)))

    assert len(signals) == 1
    assert signals[0].action == "BUY"
    assert signals[0].outcome == "NO"


def test_no_trade_on_weak_edge():
    strategy = _strategy(prob_yes=0.55, prob_no=0.45)
    signals = strategy.analyze(_market_data(_book("yes", 0.53, 0.54), _book("no", 0.45, 0.46)))

    assert signals == []


def test_no_trade_on_wide_spread():
    strategy = _strategy(prob_yes=0.70, prob_no=0.30)
    signals = strategy.analyze(_market_data(_book("yes", 0.40, 0.70), _book("no", 0.30, 0.60)))

    assert signals == []
    assert strategy.last_feature_status == "spread_too_wide"


def test_fail_closed_on_stale_feed():
    strategy = _strategy(prob_yes=0.70, prob_no=0.30, healthy=False)
    signals = strategy.analyze(_market_data(_book("yes", 0.50, 0.52), _book("no", 0.48, 0.50)))

    assert signals == []
    assert strategy.last_feature_status == "stale_btc_feed"


def test_exit_on_model_flip():
    strategy = _strategy(prob_yes=0.40, prob_no=0.60)
    strategy.active_token_id = "yes"
    strategy.entry_price = 0.55
    strategy.entry_timestamp = time.monotonic() - 10
    strategy.entry_size = 10.0
    positions = [Position(market_id="mkt", token_id="yes", outcome="YES", size=10.0, average_price=0.55)]

    signals = strategy.analyze(_market_data(_book("yes", 0.50, 0.51), _book("no", 0.51, 0.52), positions=positions))

    assert len(signals) == 1
    assert signals[0].action == "SELL"
    assert signals[0].outcome == "YES"
    assert "model_flip" in signals[0].reason or "edge_reprice" in signals[0].reason


def test_paper_trading_smoke_executes_signal():
    strategy = _strategy(prob_yes=0.62, prob_no=0.38)
    signal = strategy.analyze(_market_data(_book("yes", 0.53, 0.55), _book("no", 0.60, 0.62)))[0]

    client = PolymarketClient(paper_trading=True)
    risk_manager = RiskManager()
    state = BotState()
    inventories = {}

    _execute_signals(
        [signal],
        client,
        strategy,
        risk_manager,
        state,
        current_market_id="mkt",
        yes_token_id="yes",
        no_token_id="no",
        balance=10_000.0,
        positions=[],
        paper_trading=True,
        logger=logging.getLogger("test"),
        inventories=inventories,
    )

    assert "yes" in inventories
    assert inventories["yes"].position > 0
    assert state.strategy_status == "POSITION_OPEN"
