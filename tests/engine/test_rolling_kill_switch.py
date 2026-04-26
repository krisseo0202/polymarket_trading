"""Tests for the rolling-window kill switches.

These guard against bad streaks before they eat the day's session-loss
allowance. They're informed by the diagnose_run analysis on the 131-trade
session that lost $414: a single 20-trade window during that run
sustained ~30% win rate and ~$200 of cumulative loss before the session
cap fired. Catching that earlier would have saved most of the bleed.
"""

from __future__ import annotations

from src.engine.risk_manager import RiskLimits, RiskManager


def _mk_rm(window_n=5, pnl_floor=-50.0, winrate_floor=0.30) -> RiskManager:
    return RiskManager(RiskLimits(
        rolling_window_n=window_n,
        rolling_pnl_floor_usdc=pnl_floor,
        rolling_winrate_floor=winrate_floor,
    ))


def test_kill_switch_disabled_by_default():
    rm = RiskManager()
    halted, reason = rm.rolling_kill_switch()
    assert not halted
    assert reason == "OK"


def test_kill_switch_inactive_until_window_full():
    rm = _mk_rm(window_n=5, pnl_floor=-1.0)
    for _ in range(4):
        rm.record_trade(-100.0)  # massive losses
    halted, _ = rm.rolling_kill_switch()
    assert not halted, "must not halt before the rolling window is full"


def test_kill_switch_pnl_floor_trips():
    rm = _mk_rm(window_n=5, pnl_floor=-50.0)
    for pnl in (-15.0, -10.0, -8.0, -12.0, -10.0):  # sum = -55 < -50
        rm.record_trade(pnl)
    halted, reason = rm.rolling_kill_switch()
    assert halted
    assert "Rolling PnL floor" in reason
    assert "$-55" in reason


def test_kill_switch_winrate_floor_trips():
    rm = _mk_rm(window_n=5, pnl_floor=-1000.0, winrate_floor=0.40)
    # 1 win, 4 losses → 20% win rate < 40%
    for pnl in (-5.0, -5.0, -5.0, +10.0, -5.0):
        rm.record_trade(pnl)
    halted, reason = rm.rolling_kill_switch()
    assert halted
    assert "win-rate floor" in reason


def test_kill_switch_does_not_trip_on_break_even_window():
    rm = _mk_rm(window_n=5, pnl_floor=-50.0, winrate_floor=0.30)
    # 3 wins, 2 losses, sum positive
    for pnl in (+10.0, -5.0, +8.0, -7.0, +5.0):
        rm.record_trade(pnl)
    halted, _ = rm.rolling_kill_switch()
    assert not halted


def test_window_slides_old_losses_out():
    """A bad streak followed by a good streak must lift the halt."""
    rm = _mk_rm(window_n=5, pnl_floor=-50.0, winrate_floor=0.0)
    for pnl in (-15.0, -10.0, -8.0, -12.0, -10.0):
        rm.record_trade(pnl)
    halted, _ = rm.rolling_kill_switch()
    assert halted

    # Recover with 5 wins (push the losses out of the rolling window)
    for _ in range(5):
        rm.record_trade(+5.0)
    halted, _ = rm.rolling_kill_switch()
    assert not halted, "halt must lift once the bad streak slides out"


def test_kill_switch_blocks_position_sizing():
    """The kill switch must actually clip new entries to size 0."""
    from src.strategies.base import Signal

    rm = _mk_rm(window_n=5, pnl_floor=-50.0)
    for pnl in (-15.0, -10.0, -8.0, -12.0, -10.0):
        rm.record_trade(pnl)

    sig = Signal(
        market_id="m", outcome="YES", action="BUY",
        confidence=0.7, price=0.5, size=100.0, reason="t",
    )
    size = rm.calculate_position_size(sig, balance=10_000.0, positions=[], current_price=0.5)
    assert size == 0.0, "rolling kill switch must clip new entries to 0"
