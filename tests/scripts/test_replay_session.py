"""Tests for scripts/replay_session.py — the backtest parity tool.

The replay script's contract: given the same Settlement records the
live bot produced, it must compute identical PnL via
``(exit - entry) * size_shares``. If this drifts, the backtester is
unreliable and offline experiments lose meaning.

These tests build synthetic bot.log fixtures with hand-known PnLs and
confirm the script's per-trade and total math matches.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pytest

from scripts.replay_session import (
    _replay_pnl,
    replay_session,
    _flag_outliers,
    ReplayTrade,
)


def _write_botlog(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "btc_updown_bot.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_replay_pnl_winning_yes_trade():
    """A 60-share YES trade entered at 0.50, settled at 0.99 (winner).
    Live formula: (0.99 - 0.50) * 60 = +29.40. Replay must match."""
    s = {"entry": 0.50, "exit": 0.99, "size": 60.0}
    assert _replay_pnl(s) == pytest.approx(29.40)


def test_replay_pnl_losing_no_trade():
    """A 50-share NO trade entered at 0.45, settled at 0.01 (loser).
    Live formula: (0.01 - 0.45) * 50 = -22.00."""
    s = {"entry": 0.45, "exit": 0.01, "size": 50.0}
    assert _replay_pnl(s) == pytest.approx(-22.00)


def test_replay_session_matches_three_synthetic_trades(tmp_path):
    """End-to-end: a fixture bot.log with three Settlement records.
    Total PnL must match the sum of per-trade math."""
    lines = [
        # YES winner: 60sh @ 0.50 → 0.99 → +29.40
        "2026-04-26 10:00:00 - polymarket_trading - INFO - slot_state.py:346 - "
        "[paper] Settlement SELL: YES 12345 60.00sh @ 0.99 (entry 0.5000) "
        "→ realized +29.4000 (outcome: Up)",
        # YES loser: 50sh @ 0.55 → 0.01 → -27.00
        "2026-04-26 10:05:00 - polymarket_trading - INFO - slot_state.py:346 - "
        "[paper] Settlement SELL: YES 67890 50.00sh @ 0.01 (entry 0.5500) "
        "→ realized -27.0000 (outcome: Down)",
        # NO winner: 100sh @ 0.40 → 0.99 → +59.00
        "2026-04-26 10:10:00 - polymarket_trading - INFO - slot_state.py:346 - "
        "[paper] Settlement SELL: NO 11111 100.00sh @ 0.99 (entry 0.4000) "
        "→ realized +59.0000 (outcome: Down)",
    ]
    botlog = _write_botlog(tmp_path, lines)
    decision_log = tmp_path / "decision_log.jsonl"
    decision_log.write_text("")  # not consumed by replay_session today

    since = datetime(2026, 4, 26, 0, 0, 0).timestamp()
    trades = replay_session(decision_log, botlog, since_ts=since)

    assert len(trades) == 3
    by_side = {(t.side, t.size_shares): t for t in trades}
    yes_win = by_side[("YES", 60.0)]
    yes_loss = by_side[("YES", 50.0)]
    no_win = by_side[("NO", 100.0)]

    assert yes_win.replay_pnl == pytest.approx(29.40)
    assert yes_win.live_pnl == pytest.approx(29.40)
    assert yes_win.diff == pytest.approx(0.0)

    assert yes_loss.replay_pnl == pytest.approx(-27.00)
    assert yes_loss.live_pnl == pytest.approx(-27.00)

    assert no_win.replay_pnl == pytest.approx(59.00)
    assert no_win.live_pnl == pytest.approx(59.00)

    total_replay = sum(t.replay_pnl for t in trades)
    total_live = sum(t.live_pnl for t in trades)
    assert total_replay == pytest.approx(61.40)
    assert total_live == pytest.approx(61.40)


def test_flag_outliers_returns_only_disagreements():
    trades = [
        ReplayTrade(ts=0, side="YES", entry=0.5, exit=0.99, size_shares=60,
                    outcome="Up", live_pnl=29.40, replay_pnl=29.40),
        ReplayTrade(ts=0, side="NO", entry=0.4, exit=0.01, size_shares=50,
                    outcome="Up", live_pnl=-19.50, replay_pnl=-19.50),
        # Synthetic disagreement (would only happen if a fee/rounding
        # divergence existed):
        ReplayTrade(ts=0, side="YES", entry=0.6, exit=0.99, size_shares=40,
                    outcome="Up", live_pnl=15.00, replay_pnl=15.60),
    ]
    out = _flag_outliers(trades, threshold=0.01)
    assert len(out) == 1
    assert out[0].side == "YES"
    assert out[0].size_shares == 40


def test_replay_handles_empty_window(tmp_path):
    """No Settlement records → empty list, no crash."""
    botlog = _write_botlog(tmp_path, [
        "2026-04-26 10:00:00 - polymarket_trading - INFO - cycle_runner.py:1 - some unrelated line",
    ])
    decision_log = tmp_path / "decision_log.jsonl"
    decision_log.write_text("")
    since = datetime(2026, 4, 26, 0, 0, 0).timestamp()
    trades = replay_session(decision_log, botlog, since_ts=since)
    assert trades == []


def test_replay_filters_by_since_ts(tmp_path):
    """Settlement records before since_ts must be excluded."""
    lines = [
        # Before window
        "2026-04-25 23:59:59 - polymarket_trading - INFO - slot_state.py:346 - "
        "[paper] Settlement SELL: YES 11111 10.00sh @ 0.99 (entry 0.5000) "
        "→ realized +4.9000 (outcome: Up)",
        # Inside window
        "2026-04-26 00:00:01 - polymarket_trading - INFO - slot_state.py:346 - "
        "[paper] Settlement SELL: NO 22222 20.00sh @ 0.99 (entry 0.4000) "
        "→ realized +11.8000 (outcome: Down)",
    ]
    botlog = _write_botlog(tmp_path, lines)
    decision_log = tmp_path / "decision_log.jsonl"
    decision_log.write_text("")
    since = datetime(2026, 4, 26, 0, 0, 0).timestamp()
    trades = replay_session(decision_log, botlog, since_ts=since)
    assert len(trades) == 1
    assert trades[0].side == "NO"
