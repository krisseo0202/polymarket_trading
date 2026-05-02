"""Regression test for the silent-rejection bug.

Root cause: when a strategy had ``enabled: false`` in config, every BUY
signal was rejected by ``should_enter`` → ``validate_signal`` with a
DEBUG-level log that didn't appear at INFO. The decision log recorded
"TRADED" (strategy intent), but no order was placed and the session
counter stayed at 0. Symptoms: user saw 0 trades despite the bot running
for hours.

Fix: promote the rejection log from DEBUG to INFO so the failure mode is
visible. This test locks in the visibility contract.
"""

from __future__ import annotations

import logging


from src.engine.cycle_runner import execute_signals
from src.strategies.base import Signal, Strategy


class _FakeClient:
    def place_order(self, **kwargs):  # pragma: no cover - should never fire
        raise AssertionError("place_order must not be called when strategy is disabled")


class _FakeRiskManager:
    def calculate_position_size(self, sig, balance, positions, price):  # pragma: no cover
        raise AssertionError("risk manager must not run after strategy rejection")


class _DisabledStrategy(Strategy):
    """Minimum strategy that is disabled. validate_signal will short-circuit."""

    def __init__(self):
        super().__init__(name="disabled", config={"enabled": False})

    def analyze(self, *args, **kwargs):  # pragma: no cover
        return []

    def should_enter(self, signal):
        return self.validate_signal(signal)


class _FakeState:
    def __init__(self):
        self.active_order_ids = {}
        self.strategy_last_signal = ""
        self.strategy_last_signal_ts = 0.0
        self.strategy_edge_yes = None
        self.strategy_edge_no = None
        self.strategy_status = ""
        self.strategy_name = "disabled"
        self.trade_log = []


def test_disabled_strategy_logs_rejection_at_info_level(caplog):
    """A disabled strategy's rejections must be visible at INFO, not hidden at DEBUG."""
    signal = Signal(
        market_id="mkt",
        outcome="NO",
        action="BUY",
        confidence=0.7,
        price=0.6,
        size=33.33,
        reason="synthetic test signal",
    )

    logger = logging.getLogger("test_disabled_strategy")
    with caplog.at_level(logging.INFO, logger="test_disabled_strategy"):
        execute_signals(
            signals=[signal],
            client=_FakeClient(),
            strategy=_DisabledStrategy(),
            risk_manager=_FakeRiskManager(),
            state=_FakeState(),
            current_market_id="mkt",
            yes_token_id="yes",
            no_token_id="no",
            balance=10_000.0,
            positions=[],
            paper_trading=True,
            logger=logger,
        )

    rejection_records = [r for r in caplog.records if "rejected by strategy.should_enter" in r.message]
    assert rejection_records, (
        "execute_signals must log strategy rejections at INFO level so the "
        "failure mode isn't silent"
    )
    assert rejection_records[0].levelno >= logging.INFO
    # The diagnostic must include the enabled flag so the root cause of a
    # silent rejection is obvious from the log line itself.
    assert "enabled=False" in rejection_records[0].message
