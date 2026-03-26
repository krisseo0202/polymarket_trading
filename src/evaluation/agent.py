"""Minimal strategy evaluation agent for PROMOTE / REJECT decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..api.types import OrderBook, OrderBookEntry
from ..backtest.backtester import BacktestResult
from ..strategies.base import Strategy


@dataclass
class EvaluationConfig:
    spread: float = 0.02                     # full synthetic spread on mid-price
    initial_balance: float = 1000.0
    min_trades: int = 20
    max_drawdown_threshold: float = 0.25
    concentration_threshold: float = 0.60    # top-3 trades share of total PnL
    instability_threshold: float = 0.50      # second-half degradation fraction
    evaluation_window_hours: Optional[float] = None   # None = use all rows
    debug: bool = False


@dataclass
class EvaluationResult:
    decision: str        # "PROMOTE" or "REJECT"
    reason: str
    metrics: Dict[str, Any]
    confidence: float    # [0.0, 1.0]


class EvaluationAgent:
    """
    Evaluates a strategy on resolved historical market slots and returns a
    structured PROMOTE / REJECT verdict.

    Execution model:
    - Entries filled at ask = mid + spread/2  (taker cost)
    - Exits at binary resolution: 1.0 if won, 0.0 if lost
    - No lookahead: rows processed strictly chronologically
    """

    def __init__(self, config: EvaluationConfig = EvaluationConfig()):
        self.config = config

    def evaluate(
        self,
        strategy: Strategy,
        data: Union[str, pd.DataFrame],
    ) -> EvaluationResult:
        df = self._load_data(data)
        df = self._filter_window(df)

        balance = self.config.initial_balance
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = []
        total_resolved = 0

        for _, row in df.iterrows():
            up_token = str(row["up_token"])
            down_token = str(row["down_token"])
            market_id = up_token  # use up_token as a stable market proxy

            strategy.set_tokens(market_id, up_token, down_token)

            mid_yes = float(row["up_price_start"])
            mid_no = float(row["down_price_start"])
            slot_ts = float(row["slot_ts"])

            strategy.record_price(up_token, mid_yes, ts=slot_ts)
            strategy.record_price(down_token, mid_no, ts=slot_ts)

            book_yes = self._make_book(up_token, mid_yes, market_id)
            book_no = self._make_book(down_token, mid_no, market_id)

            market_data = {
                "order_books": {up_token: book_yes, down_token: book_no},
                "positions": [],
                "balance": balance,
                "slot_ts": slot_ts,
                "time_to_expiry": 300.0,
            }

            try:
                signals = strategy.analyze(market_data)
            except Exception:
                signals = []

            outcome = str(row.get("outcome", "")).strip()
            if outcome not in ("Up", "Down"):
                # Unresolved slot: advance price history but don't trade
                equity_curve.append(balance)
                continue

            total_resolved += 1

            for sig in signals:
                if strategy.validate_signal(sig) and strategy.should_enter(sig):
                    trade = self._simulate_fill(sig, row, outcome)
                    if trade:
                        # Binary contract: pay cost now, receive payout at resolution
                        balance += trade["pnl"]
                        trades.append(trade)
                        if self.config.debug:
                            self._print_debug_trade(trade)
                    break  # one trade per slot

            equity_curve.append(balance)

        metrics = self._compute_metrics(trades, equity_curve, total_resolved)
        decision, reason = self._apply_rejection_rules(metrics)
        confidence = self._compute_confidence(metrics, decision)
        return EvaluationResult(decision=decision, reason=reason, metrics=metrics, confidence=confidence)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, str):
            return pd.read_csv(data)
        return data.copy()

    def _filter_window(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("slot_ts").reset_index(drop=True)
        hours = self.config.evaluation_window_hours
        if hours is not None and len(df) > 0:
            cutoff = df["slot_ts"].max() - hours * 3600
            df = df[df["slot_ts"] >= cutoff].reset_index(drop=True)
        return df

    def _make_book(self, token_id: str, mid: float, market_id: str) -> OrderBook:
        half = self.config.spread / 2.0
        bid = max(0.01, min(0.99, mid - half))
        ask = max(0.01, min(0.99, mid + half))
        return OrderBook(
            market_id=market_id,
            token_id=token_id,
            bids=[OrderBookEntry(price=bid, size=500.0)],
            asks=[OrderBookEntry(price=ask, size=500.0)],
            last_price=mid,
            tick_size=0.01,
        )

    def _simulate_fill(
        self, signal, row: pd.Series, outcome: str
    ) -> Optional[Dict[str, Any]]:
        half = self.config.spread / 2.0
        if signal.outcome == "YES":
            entry_price = min(0.99, float(row["up_price_start"]) + half)
        else:
            entry_price = min(0.99, float(row["down_price_start"]) + half)

        entry_price = max(0.01, entry_price)
        size = signal.size
        cost = entry_price * size

        won = (signal.outcome == "YES" and outcome == "Up") or \
              (signal.outcome == "NO" and outcome == "Down")
        exit_price = 1.0 if won else 0.0
        pnl = (exit_price - entry_price) * size

        return {
            "slot_ts": float(row["slot_ts"]),
            "outcome_traded": signal.outcome,
            "market_outcome": outcome,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "cost": cost,
            "pnl": pnl,
            "won": won,
            "confidence": signal.confidence,
            "signal_reason": signal.reason,
        }

    def _compute_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        total_resolved: int,
    ) -> Dict[str, Any]:
        num_trades = len(trades)
        total_pnl = sum(t["pnl"] for t in trades)
        pnl_per_trade = total_pnl / num_trades if num_trades else 0.0
        exposure_time = num_trades / total_resolved if total_resolved else 0.0

        # Reuse BacktestResult for equity-curve-based metrics
        br = BacktestResult()
        br.initial_balance = self.config.initial_balance
        br.final_balance = (equity_curve[-1] if equity_curve else self.config.initial_balance)
        br.trades = trades
        br.equity_curve = equity_curve
        br.calculate_metrics()

        # Concentration: top-3 winning trades / total_pnl
        concentration_ratio = 0.0
        if total_pnl > 0 and num_trades >= 3:
            sorted_pnl = sorted((t["pnl"] for t in trades), reverse=True)
            concentration_ratio = sum(sorted_pnl[:3]) / total_pnl

        # Stability: first-half vs second-half PnL
        first_half_pnl = 0.0
        second_half_pnl = 0.0
        instability_ratio = 0.0
        if num_trades >= 2:
            mid = num_trades // 2
            first_half_pnl = sum(t["pnl"] for t in trades[:mid])
            second_half_pnl = sum(t["pnl"] for t in trades[mid:])
            if first_half_pnl > 0:
                instability_ratio = (first_half_pnl - second_half_pnl) / abs(first_half_pnl)

        return {
            "num_trades": num_trades,
            "total_resolved_rows": total_resolved,
            "total_pnl": round(total_pnl, 4),
            "pnl_per_trade": round(pnl_per_trade, 4),
            "exposure_time": round(exposure_time, 4),
            "win_rate": round(br.win_rate, 4),
            "max_drawdown": round(br.max_drawdown, 4),
            "sharpe_ratio": round(br.sharpe_ratio, 4),
            "total_return": round(br.total_return, 4),
            "winning_trades": br.winning_trades,
            "losing_trades": br.losing_trades,
            "concentration_ratio": round(concentration_ratio, 4),
            "first_half_pnl": round(first_half_pnl, 4),
            "second_half_pnl": round(second_half_pnl, 4),
            "instability_ratio": round(instability_ratio, 4),
        }

    def _apply_rejection_rules(
        self, metrics: Dict[str, Any]
    ) -> Tuple[str, str]:
        cfg = self.config

        if metrics["num_trades"] < cfg.min_trades:
            return "REJECT", (
                f"insufficient_trades: {metrics['num_trades']} < {cfg.min_trades}"
            )

        if metrics["max_drawdown"] > cfg.max_drawdown_threshold:
            return "REJECT", (
                f"excessive_drawdown: {metrics['max_drawdown']:.2%} > "
                f"{cfg.max_drawdown_threshold:.2%}"
            )

        if metrics["concentration_ratio"] > cfg.concentration_threshold:
            return "REJECT", (
                f"pnl_concentration: top-3 trades account for "
                f"{metrics['concentration_ratio']:.1%} of total PnL "
                f"(threshold {cfg.concentration_threshold:.0%})"
            )

        if (
            metrics["first_half_pnl"] > 0
            and metrics["instability_ratio"] > cfg.instability_threshold
        ):
            return "REJECT", (
                f"strategy_instability: second-half PnL degraded by "
                f"{metrics['instability_ratio']:.1%} vs first half "
                f"(threshold {cfg.instability_threshold:.0%})"
            )

        return "PROMOTE", "all_checks_passed"

    def _compute_confidence(
        self, metrics: Dict[str, Any], decision: str
    ) -> float:
        cfg = self.config
        count_score = min(1.0, metrics["num_trades"] / (5 * cfg.min_trades))
        wr_score = max(0.0, (metrics["win_rate"] - 0.50) / 0.50)
        dd_score = max(
            0.0,
            1.0 - metrics["max_drawdown"] / cfg.max_drawdown_threshold
            if cfg.max_drawdown_threshold > 0 else 0.0,
        )
        consistency_score = max(0.0, 1.0 - max(0.0, metrics["instability_ratio"]))

        raw = (
            0.30 * count_score
            + 0.35 * wr_score
            + 0.20 * dd_score
            + 0.15 * consistency_score
        )
        result = (1.0 - raw) if decision == "REJECT" else raw
        return round(min(1.0, max(0.0, result)), 3)

    def _print_debug_trade(self, trade: Dict[str, Any]) -> None:
        pnl_sign = "+" if trade["pnl"] >= 0 else ""
        print(
            f"[DEBUG TRADE] slot={int(trade['slot_ts'])} "
            f"side={trade['outcome_traded']} "
            f"entry={trade['entry_price']:.3f} "
            f"exit={trade['exit_price']:.3f} "
            f"size={trade['size']:.2f} "
            f"pnl={pnl_sign}{trade['pnl']:.4f} "
            f"won={trade['won']} "
            f"conf={trade['confidence']:.2f} "
            f"reason=\"{trade['signal_reason']}\""
        )
