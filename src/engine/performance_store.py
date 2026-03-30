"""
SQLite-backed persistence for live session and backtest performance summaries.

Each live bot run writes one row to `sessions` at shutdown.
Each backtester.run() call writes one row to `backtests`.

Usage:
    store = PerformanceStore("perf.db")
    store.record_session(state, "prob_edge", start_ts, end_ts, paper_trading=True)
    store.record_backtest(result, "prob_edge", params, "2026-01-01", "2026-03-01")
    rows = store.compare("prob_edge")
"""

import json
import sqlite3
import time
from datetime import date
from typing import Any, Dict, List, Optional

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts          REAL    NOT NULL,
    strategy_name   TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    start_ts        REAL    NOT NULL,
    end_ts          REAL    NOT NULL,
    paper_trading   INTEGER NOT NULL DEFAULT 1,
    n_trades        INTEGER NOT NULL DEFAULT 0,
    n_wins          INTEGER NOT NULL DEFAULT 0,
    n_losses        INTEGER NOT NULL DEFAULT 0,
    win_rate        REAL    NOT NULL DEFAULT 0.0,
    realized_pnl    REAL    NOT NULL DEFAULT 0.0,
    notes           TEXT
)
"""

_CREATE_BACKTESTS = """
CREATE TABLE IF NOT EXISTS backtests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts          REAL    NOT NULL,
    strategy_name   TEXT    NOT NULL,
    data_start      TEXT    NOT NULL,
    data_end        TEXT    NOT NULL,
    initial_balance REAL    NOT NULL DEFAULT 0.0,
    final_balance   REAL    NOT NULL DEFAULT 0.0,
    total_return    REAL    NOT NULL DEFAULT 0.0,
    sharpe_ratio    REAL    NOT NULL DEFAULT 0.0,
    max_drawdown    REAL    NOT NULL DEFAULT 0.0,
    win_rate        REAL    NOT NULL DEFAULT 0.0,
    n_trades        INTEGER NOT NULL DEFAULT 0,
    n_wins          INTEGER NOT NULL DEFAULT 0,
    n_losses        INTEGER NOT NULL DEFAULT 0,
    params_json     TEXT
)
"""


class PerformanceStore:
    def __init__(self, path: str = "perf.db") -> None:
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(_CREATE_SESSIONS)
            self._conn.execute(_CREATE_BACKTESTS)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_session(
        self,
        state: Any,           # BotState — imported lazily to avoid circular deps
        strategy_name: str,
        start_ts: float,
        end_ts: float,
        paper_trading: bool = True,
        notes: Optional[str] = None,
    ) -> None:
        """Record one live bot session at shutdown."""
        n_wins: int = state.session_wins
        n_losses: int = state.session_losses
        n_trades = n_wins + n_losses
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        with self._conn:
            self._conn.execute(
                """INSERT INTO sessions
                   (run_ts, strategy_name, date, start_ts, end_ts,
                    paper_trading, n_trades, n_wins, n_losses, win_rate,
                    realized_pnl, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    strategy_name,
                    str(date.today()),
                    start_ts,
                    end_ts,
                    int(paper_trading),
                    n_trades,
                    n_wins,
                    n_losses,
                    win_rate,
                    state.daily_realized_pnl,
                    notes,
                ),
            )

    def record_backtest(
        self,
        result: Any,          # BacktestResult
        strategy_name: str,
        params: Optional[Dict[str, Any]] = None,
        data_start: str = "",
        data_end: str = "",
    ) -> None:
        """Record one backtest run."""
        with self._conn:
            self._conn.execute(
                """INSERT INTO backtests
                   (run_ts, strategy_name, data_start, data_end,
                    initial_balance, final_balance, total_return,
                    sharpe_ratio, max_drawdown, win_rate,
                    n_trades, n_wins, n_losses, params_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    strategy_name,
                    data_start,
                    data_end,
                    result.initial_balance,
                    result.final_balance,
                    result.total_return,
                    result.sharpe_ratio,
                    result.max_drawdown,
                    result.win_rate,
                    result.total_trades,
                    result.winning_trades,
                    result.losing_trades,
                    json.dumps(params) if params else None,
                ),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def sessions(
        self,
        strategy_name: Optional[str] = None,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return last n live session rows, newest first."""
        if strategy_name:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE strategy_name=? ORDER BY run_ts DESC LIMIT ?",
                (strategy_name, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY run_ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def backtests(
        self,
        strategy_name: Optional[str] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return last n backtest rows, newest first."""
        if strategy_name:
            rows = self._conn.execute(
                "SELECT * FROM backtests WHERE strategy_name=? ORDER BY run_ts DESC LIMIT ?",
                (strategy_name, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM backtests ORDER BY run_ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def compare(self, strategy_name: str) -> Dict[str, Any]:
        """Return the most recent live session and backtest for a strategy side-by-side."""
        live = self.sessions(strategy_name, n=1)
        bt = self.backtests(strategy_name, n=1)
        return {
            "strategy": strategy_name,
            "live": live[0] if live else None,
            "backtest": bt[0] if bt else None,
        }

    def close(self) -> None:
        self._conn.close()
