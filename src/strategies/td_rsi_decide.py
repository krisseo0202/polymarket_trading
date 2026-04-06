"""Minimal TD Sequential + RSI decision rule for BTC 5-minute Polymarket markets.

Convention
----------
td_{tf} is a signed setup count:
  negative  →  TD Down (buy setup): consecutive closes < close[4]  →  downtrend exhaustion
  positive  →  TD Up  (sell setup): consecutive closes > close[4]  →  uptrend exhaustion

Decision
--------
  BUY_YES   TD Down near/complete (td <= -8) + oversold RSI (< 35)
  BUY_NO    TD Up   near/complete (td >= +8) + overbought RSI (> 65)
  NO_TRADE  signals absent, mixed, or too close to expiry
"""

from typing import Literal

Decision = Literal["BUY_YES", "BUY_NO", "NO_TRADE"]

# ── Thresholds ────────────────────────────────────────────────────────────────
_MIN_TTE        = 45    # seconds — no new entries this close to expiry
_RSI_OVERSOLD   = 35
_RSI_OVERBOUGHT = 65
_TD_TRIGGER     = 8     # fire on bar 8 or 9 of a setup

# ── Timeframe weights (shorter = higher weight) ───────────────────────────────
# 30s=4, 1m=3, 3m=2, 5m=1   →   minimum score to trade = 4
_TIMEFRAMES = [
    ("30s", 4),
    ("1m",  3),
    ("3m",  2),
    ("5m",  1),
]
_MIN_SCORE = 4


def decide(snapshot: dict) -> Decision:
    """Return BUY_YES, BUY_NO, or NO_TRADE based on TD + RSI signals.

    snapshot keys
    -------------
    td_30s, rsi_30s   — 30-second bar values
    td_1m,  rsi_1m    — 1-minute bar values
    td_3m,  rsi_3m    — 3-minute bar values
    td_5m,  rsi_5m    — 5-minute bar values
    time_to_expiry_seconds
    """
    if snapshot.get("time_to_expiry_seconds", 0) < _MIN_TTE:
        return "NO_TRADE"

    yes_score = 0
    no_score  = 0

    for tf, weight in _TIMEFRAMES:
        td  = snapshot.get(f"td_{tf}")
        rsi = snapshot.get(f"rsi_{tf}")
        if td is None or rsi is None:
            continue

        if td <= -_TD_TRIGGER and rsi < _RSI_OVERSOLD:
            yes_score += weight
        elif td >= _TD_TRIGGER and rsi > _RSI_OVERBOUGHT:
            no_score  += weight

    if yes_score >= _MIN_SCORE and yes_score > no_score:
        return "BUY_YES"
    if no_score >= _MIN_SCORE and no_score > yes_score:
        return "BUY_NO"
    return "NO_TRADE"
