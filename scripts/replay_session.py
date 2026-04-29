"""Stage 1 backtest-parity tool.

Replays a live trading session by reading the actual decisions the bot
made (decision-log JSONL) and the actual settlement outcomes (bot.log
Settlement records), then computes PnL using the same dollar-math the
live bot uses (`(exit - entry) * size_shares`).

If the replay PnL doesn't match the live session PnL within a small
tolerance, our backtester is unreliable and any subsequent feature
experiment can't be trusted.

Usage::

    python scripts/replay_session.py \\
        --decision-log data/2026-04-26/decision_log_*.jsonl \\
        --bot-log logs/btc_updown_bot.log \\
        --since "2026-04-26 00:00:00"

Output: per-side PnL totals, replay-vs-live comparison, exit code 0
when the diff is within tolerance, 1 when it isn't.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the already-tested parsers from diagnose_run instead of forking them.
from scripts.diagnose_run import _parse_settlements


_DEFAULT_TOLERANCE_USDC = 0.50


@dataclass
class ReplayTrade:
    """One settled trade reconstructed from the bot's own logs."""
    ts: float
    side: str
    entry: float
    exit: float
    size_shares: float
    outcome: str
    live_pnl: float        # what the bot reported (from Settlement realized)
    replay_pnl: float      # what we recompute via (exit - entry) * shares

    @property
    def diff(self) -> float:
        return self.replay_pnl - self.live_pnl


def _replay_pnl(s: dict) -> float:
    """Recompute PnL using the live bot's math.

    The bot tracks shares (s["size"]) and entry price (s["entry"]). At
    settlement, the paper-trading branch closes the position at the
    recorded exit (typically 0.99 for a winner, 0.01 for a loser — the
    market quote at slot end, not 1.0/0.0).

    Live formula (src/engine/inventory.py:apply_fill):
        pnl = (exit_price - entry_price) * size_shares

    For NO positions the math is symmetric because the bot stores NO
    shares + NO entry directly (it doesn't invert).
    """
    return (s["exit"] - s["entry"]) * s["size"]


def replay_session(
    decision_log: Path,
    bot_log: Path,
    since_ts: float,
) -> List[ReplayTrade]:
    """Reconstruct each settled trade and compute PnL two ways."""
    settlements = _parse_settlements(bot_log, since_ts=since_ts)
    out: List[ReplayTrade] = []
    for s in settlements:
        out.append(ReplayTrade(
            ts=s["ts"],
            side=s["side"],
            entry=s["entry"],
            exit=s["exit"],
            size_shares=s["size"],
            outcome=s["outcome"],
            live_pnl=s["pnl"],
            replay_pnl=_replay_pnl(s),
        ))
    return out


def _format_table(trades: Sequence[ReplayTrade]) -> str:
    """Per-side rollup + grand total."""
    by_side = {"YES": [], "NO": []}
    for t in trades:
        by_side.setdefault(t.side, []).append(t)

    lines = []
    lines.append(f"{'side':<6} {'n':>5} {'live_pnl':>12} {'replay_pnl':>12} {'diff':>10}")
    lines.append("-" * 55)
    for side in ("YES", "NO"):
        rows = by_side.get(side, [])
        if not rows:
            continue
        live = sum(r.live_pnl for r in rows)
        rep = sum(r.replay_pnl for r in rows)
        lines.append(
            f"{side:<6} {len(rows):>5d} {live:>+12.4f} {rep:>+12.4f} {rep - live:>+10.4f}"
        )
    live_tot = sum(t.live_pnl for t in trades)
    rep_tot = sum(t.replay_pnl for t in trades)
    lines.append("-" * 55)
    lines.append(
        f"{'TOTAL':<6} {len(trades):>5d} {live_tot:>+12.4f} {rep_tot:>+12.4f} {rep_tot - live_tot:>+10.4f}"
    )
    return "\n".join(lines)


def _flag_outliers(trades: Sequence[ReplayTrade], threshold: float = 0.01) -> List[ReplayTrade]:
    """Return per-trade rows where replay disagrees with live by more than `threshold`."""
    return [t for t in trades if abs(t.diff) > threshold]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--bot-log", type=Path, required=True)
    parser.add_argument("--since", required=True,
                        help="Start of replay window (UTC), 'YYYY-MM-DD HH:MM:SS'.")
    parser.add_argument("--tolerance-usdc", type=float, default=_DEFAULT_TOLERANCE_USDC,
                        help="Pass when |replay_total - live_total| ≤ this. Default 0.50.")
    parser.add_argument("--show-outliers", action="store_true",
                        help="Print every trade where replay differs from live by > $0.01.")
    args = parser.parse_args()

    since_ts = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S").timestamp()

    trades = replay_session(
        decision_log=args.decision_log,
        bot_log=args.bot_log,
        since_ts=since_ts,
    )

    if not trades:
        print("No settled trades found in the window. Nothing to compare.")
        return 0

    print(f"=== Stage 1: backtest parity replay ===")
    print(f"window since: {args.since} UTC")
    print(f"settled trades: {len(trades)}")
    print()
    print(_format_table(trades))
    print()

    live_total = sum(t.live_pnl for t in trades)
    rep_total = sum(t.replay_pnl for t in trades)
    diff = rep_total - live_total
    print(f"replay_pnl: {rep_total:+.4f}")
    print(f"live_pnl:   {live_total:+.4f}")
    print(f"diff:       {diff:+.4f}  (tolerance ±{args.tolerance_usdc:.2f})")

    outliers = _flag_outliers(trades)
    if outliers:
        print()
        print(f"per-trade outliers (|diff| > $0.01): {len(outliers)} of {len(trades)}")
        if args.show_outliers:
            for t in outliers[:30]:
                print(f"  {datetime.utcfromtimestamp(t.ts).isoformat()}  "
                      f"{t.side} {t.size_shares:.2f}sh  entry={t.entry:.4f}  "
                      f"exit={t.exit:.4f}  live={t.live_pnl:+.4f}  "
                      f"replay={t.replay_pnl:+.4f}  diff={t.diff:+.4f}")
            if len(outliers) > 30:
                print(f"  ... and {len(outliers) - 30} more")

    if abs(diff) <= args.tolerance_usdc:
        print()
        print(f"PASS — replay matches live within ${args.tolerance_usdc:.2f}.")
        return 0
    else:
        print()
        print(f"FAIL — diff ${diff:+.4f} exceeds tolerance ±${args.tolerance_usdc:.2f}.")
        print("Stop here. Investigate the divergence before proceeding to Stage 2/3/4.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
