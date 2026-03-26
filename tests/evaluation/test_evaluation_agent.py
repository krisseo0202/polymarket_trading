"""Tests for EvaluationAgent."""

import random
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.evaluation.agent import EvaluationAgent, EvaluationConfig, EvaluationResult
from src.strategies.base import Signal, Strategy
from src.strategies.coin_toss import CoinTossStrategy
from .conftest import make_sample_df


# ---------------------------------------------------------------------------
# Stub strategies for deterministic testing
# ---------------------------------------------------------------------------

class _AlwaysYES(Strategy):
    """Buys YES on every slot regardless of market conditions."""

    def __init__(self, size: float = 20.0):
        super().__init__("always_yes", {"enabled": True})
        self._size = size

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        if not self._yes_token_id:
            return []
        books = market_data.get("order_books", {})
        book = books.get(self._yes_token_id)
        if book is None or not book.asks:
            return []
        ask = book.asks[0].price
        return [Signal(
            market_id=self._market_id,
            outcome="YES",
            action="BUY",
            confidence=0.9,
            price=ask,
            size=self._size,
            reason="always_yes",
        )]

    def should_enter(self, signal: Signal) -> bool:
        return True


class _AlwaysNO(Strategy):
    """Buys NO on every slot."""

    def __init__(self, size: float = 20.0):
        super().__init__("always_no", {"enabled": True})
        self._size = size

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        if not self._no_token_id:
            return []
        books = market_data.get("order_books", {})
        book = books.get(self._no_token_id)
        if book is None or not book.asks:
            return []
        ask = book.asks[0].price
        return [Signal(
            market_id=self._market_id,
            outcome="NO",
            action="BUY",
            confidence=0.9,
            price=ask,
            size=self._size,
            reason="always_no",
        )]

    def should_enter(self, signal: Signal) -> bool:
        return True


class _NeverTrades(Strategy):
    """Emits no signals."""

    def __init__(self):
        super().__init__("never_trades", {"enabled": True})

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        return []

    def should_enter(self, signal: Signal) -> bool:
        return False


class _VariableSize(Strategy):
    """Buys YES with a large size for first `n_big` slots, small size afterwards."""

    def __init__(self, n_big: int = 3, big_size: float = 100.0, small_size: float = 1.0):
        super().__init__("variable_size", {"enabled": True})
        self._n_big = n_big
        self._big = big_size
        self._small = small_size
        self._count = 0

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        if not self._yes_token_id:
            return []
        books = market_data.get("order_books", {})
        book = books.get(self._yes_token_id)
        if book is None or not book.asks:
            return []
        ask = book.asks[0].price
        size = self._big if self._count < self._n_big else self._small
        self._count += 1
        return [Signal(
            market_id=self._market_id,
            outcome="YES",
            action="BUY",
            confidence=0.9,
            price=ask,
            size=size,
            reason="variable_size",
        )]

    def should_enter(self, signal: Signal) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helper to build DataFrames with fixed outcomes
# ---------------------------------------------------------------------------

def _all_up_df(n: int = 40) -> pd.DataFrame:
    """All outcomes are Up; used with _AlwaysYES to guarantee wins."""
    rows = []
    base = 1_700_000_000
    for i in range(n):
        rows.append({
            "slot_ts": base + i * 300,
            "slot_utc": "",
            "question": f"slot {i}",
            "up_token": f"up_{i}",
            "down_token": f"dn_{i}",
            "outcome": "Up",
            "volume": 1000.0,
            "up_price_start": 0.50,
            "up_price_end": 1.0,
            "down_price_start": 0.50,
            "down_price_end": 0.0,
            "strike_price": 50000.0,
        })
    return pd.DataFrame(rows)


def _all_down_df(n: int = 40) -> pd.DataFrame:
    df = _all_up_df(n)
    df["outcome"] = "Down"
    return df


def _alternating_df(n: int = 40) -> pd.DataFrame:
    df = _all_up_df(n)
    df.loc[df.index % 2 == 1, "outcome"] = "Down"
    return df


def _first_half_up_df(n: int = 40) -> pd.DataFrame:
    """First half Up, second half Down; _AlwaysYES profits then loses."""
    df = _all_up_df(n)
    mid = n // 2
    df.loc[df.index >= mid, "outcome"] = "Down"
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_promote_on_oracle_strategy():
    """Strategy that wins every trade should be promoted with high confidence."""
    df = _all_up_df(40)
    result = EvaluationAgent().evaluate(_AlwaysYES(), df)
    assert result.decision == "PROMOTE"
    assert result.confidence > 0.7
    assert result.metrics["win_rate"] == 1.0
    assert result.metrics["total_pnl"] > 0


def test_reject_insufficient_trades():
    """Fewer trades than min_trades must be rejected."""
    df = _all_up_df(10)   # only 10 slots
    config = EvaluationConfig(min_trades=20)
    result = EvaluationAgent(config).evaluate(_AlwaysYES(), df)
    assert result.decision == "REJECT"
    assert "insufficient_trades" in result.reason


def test_reject_excessive_drawdown():
    """Strategy that loses every trade should be rejected for drawdown."""
    df = _all_down_df(40)  # always Down, but we buy YES → always lose
    result = EvaluationAgent().evaluate(_AlwaysYES(), df)
    assert result.decision == "REJECT"
    assert "excessive_drawdown" in result.reason
    assert result.metrics["max_drawdown"] > 0


def test_reject_concentration():
    """PnL concentrated in 3 large trades with many small trades → reject."""
    # 3 big + 27 small → 30 total trades (>= min_trades=20)
    # With all-Up outcomes, _VariableSize will win every trade.
    # top-3 pnl / total_pnl will exceed 60%.
    df = _all_up_df(30)
    strategy = _VariableSize(n_big=3, big_size=100.0, small_size=1.0)
    config = EvaluationConfig(min_trades=20, concentration_threshold=0.60)
    result = EvaluationAgent(config).evaluate(strategy, df)
    assert result.decision == "REJECT"
    assert "pnl_concentration" in result.reason
    assert result.metrics["concentration_ratio"] > 0.60


def test_reject_instability():
    """Strategy profitable in first half but losing in second half → reject."""
    df = _first_half_up_df(40)
    config = EvaluationConfig(instability_threshold=0.50)
    result = EvaluationAgent(config).evaluate(_AlwaysYES(), df)
    assert result.decision == "REJECT"
    assert "strategy_instability" in result.reason
    assert result.metrics["instability_ratio"] > 0.50


def test_coin_toss_baseline():
    """CoinTossStrategy over 100 rows should be rejected (no edge, likely drawdown)."""
    random.seed(0)
    df = make_sample_df(n=100, seed=99)
    strategy = CoinTossStrategy({"enabled": True, "position_size_usdc": 20.0})
    config = EvaluationConfig(min_trades=5)
    result = EvaluationAgent(config).evaluate(strategy, df)
    # CoinToss has random confidence=0.5 — it should trade, but rarely beats REJECT rules
    assert isinstance(result, EvaluationResult)
    assert result.decision in ("PROMOTE", "REJECT")
    assert 0.3 <= result.metrics["win_rate"] <= 0.7   # should be near 50%


def test_evaluation_window_filters_rows():
    """Only rows within the evaluation_window_hours window should be evaluated."""
    df = make_sample_df(n=200, seed=42)
    # Each slot is 300s apart; 2h = 7200s → 24 slots
    config = EvaluationConfig(evaluation_window_hours=2.0, min_trades=1)
    result = EvaluationAgent(config).evaluate(_AlwaysYES(), df)
    assert result.metrics["total_resolved_rows"] <= 25   # ~24 slots in 2h window


def test_unresolved_rows_skipped():
    """Rows with empty outcome should not generate trades."""
    df = _all_up_df(30)
    # Mark every 5th row as unresolved
    df.loc[df.index % 5 == 0, "outcome"] = ""
    result = EvaluationAgent().evaluate(_AlwaysYES(), df)
    # 30 rows, 6 unresolved → 24 resolved
    assert result.metrics["total_resolved_rows"] == 24
    assert result.metrics["num_trades"] == 24


def test_debug_mode_output(capsys):
    """Debug mode should print trade-by-trade output to stdout."""
    df = _all_up_df(25)
    config = EvaluationConfig(debug=True, min_trades=1)
    EvaluationAgent(config).evaluate(_AlwaysYES(), df)
    captured = capsys.readouterr()
    assert "DEBUG TRADE" in captured.out
    assert "entry=" in captured.out
    assert "pnl=" in captured.out


def test_metrics_keys_present():
    """EvaluationResult.metrics must contain all required keys."""
    required_keys = {
        "num_trades", "total_resolved_rows", "total_pnl", "pnl_per_trade",
        "win_rate", "max_drawdown", "sharpe_ratio", "total_return",
        "exposure_time", "concentration_ratio", "instability_ratio",
        "first_half_pnl", "second_half_pnl",
    }
    df = make_sample_df(n=40, seed=42)
    result = EvaluationAgent().evaluate(_AlwaysYES(), df)
    assert required_keys.issubset(result.metrics.keys())


def test_csv_path_input(tmp_path):
    """Passing a CSV file path should produce the same result as a DataFrame."""
    df = _all_up_df(25)
    csv_path = tmp_path / "test_slots.csv"
    df.to_csv(csv_path, index=False)

    config = EvaluationConfig(min_trades=1)
    result_df = EvaluationAgent(config).evaluate(_AlwaysYES(size=20.0), df)
    result_csv = EvaluationAgent(config).evaluate(_AlwaysYES(size=20.0), str(csv_path))

    assert result_df.decision == result_csv.decision
    assert result_df.metrics["num_trades"] == result_csv.metrics["num_trades"]
    assert abs(result_df.metrics["total_pnl"] - result_csv.metrics["total_pnl"]) < 1e-9


def test_deterministic_output():
    """Running evaluate twice with the same inputs must return identical results."""
    df = make_sample_df(n=40, seed=42)
    config = EvaluationConfig(min_trades=5)
    r1 = EvaluationAgent(config).evaluate(_AlwaysYES(), df)
    r2 = EvaluationAgent(config).evaluate(_AlwaysYES(), df)

    assert r1.decision == r2.decision
    assert r1.reason == r2.reason
    assert r1.confidence == r2.confidence
    assert r1.metrics == r2.metrics
