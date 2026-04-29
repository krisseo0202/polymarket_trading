"""Count trades where BTC was on the bet's winning side in the final seconds
of the slot but the trade still lost.

Run:
    python -m scripts.analyze_last_seconds \
        --decision-log data/2026-04-26/decision_log_20260426T055228Z.jsonl \
        --bot-log logs/btc_updown_bot.log \
        --since 2026-04-26T00:00:00Z
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3

from scripts.diagnose_run import _parse_settlements
from scripts.replay_visual import (
    _attach_settlements_by_match,
    _btc_trajectory_for_slot,
    _load_decisions_by_slot,
    _trade_records,
)

CHECK_OFFSETS = (1, 3, 5, 10, 30)  # seconds before slot close


def _parse_since(s: Optional[str]) -> float:
    if not s:
        return 0.0
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def _winning_side_btc(side: str, btc: float, strike: float) -> bool:
    """True iff BTC at this instant would settle the bet as a win.

    YES bets win when BTC > strike (Up). NO bets win when BTC < strike (Down).
    Equality goes to Up per market spec, so YES wins on equality.
    """
    if side == "YES":
        return btc >= strike
    return btc < strike


def _btc_at_offset(xs: List[float], ys: List[float], offset_from_end: float) -> Optional[float]:
    """BTC value at (slot_close - offset). xs are seconds from slot_open;
    slot length is 300s, so we want the last sample with xs <= 300 - offset.
    """
    target = 300.0 - offset_from_end
    last = None
    for x, y in zip(xs, ys):
        if x <= target:
            last = y
        else:
            break
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-log", required=True, type=Path)
    ap.add_argument("--bot-log", required=True, type=Path)
    ap.add_argument("--since", type=str, default=None,
                    help="ISO8601 UTC, e.g. 2026-04-26T00:00:00Z")
    ap.add_argument("--list-flips", action="store_true",
                    help="Print slot_ts of trades flipped within --flip-window seconds")
    ap.add_argument("--flip-window", type=int, default=5)
    args = ap.parse_args()

    since_ts = _parse_since(args.since)
    settlements = _parse_settlements(args.bot_log, since_ts=since_ts)
    by_slot = _load_decisions_by_slot(args.decision_log, since_ts=since_ts)
    trades_by_slot = {st: _trade_records(recs) for st, recs in by_slot.items()
                      if _trade_records(recs)}
    settle_by_slot = _attach_settlements_by_match(settlements, trades_by_slot)

    s3 = boto3.client("s3")

    n_total = 0
    n_lost = 0
    n_won = 0
    n_no_btc = 0
    # For each offset: trades that LOST overall but were on the winning side at T-offset.
    flipped_at: Counter = Counter()
    flipped_examples: Dict[int, List[int]] = {n: [] for n in CHECK_OFFSETS}
    # For sanity: trades that WON but were on losing side at T-offset (rescue flips).
    rescued_at: Counter = Counter()

    for slot_ts, recs in trades_by_slot.items():
        settlement = settle_by_slot.get(slot_ts)
        if settlement is None:
            continue
        first_trade = next((r for r in recs if r.get("action")), None)
        if first_trade is None:
            continue
        side = first_trade["action"].get("side")
        strike = recs[0].get("strike") or recs[0].get("f_strike_price")
        if not strike:
            continue
        strike = float(strike)
        slot_open = float(slot_ts)
        slot_close = slot_open + 300.0

        xs, ys = _btc_trajectory_for_slot(slot_open, slot_close, s3)
        if not xs:
            n_no_btc += 1
            continue

        n_total += 1
        won = settlement["pnl"] > 0
        if won:
            n_won += 1
        else:
            n_lost += 1

        for off in CHECK_OFFSETS:
            btc_at = _btc_at_offset(xs, ys, off)
            if btc_at is None:
                continue
            on_winning_side = _winning_side_btc(side, btc_at, strike)
            if not won and on_winning_side:
                flipped_at[off] += 1
                flipped_examples[off].append(slot_ts)
            elif won and not on_winning_side:
                rescued_at[off] += 1

    print(f"Trades analyzed:       {n_total}")
    print(f"  won:                 {n_won}")
    print(f"  lost:                {n_lost}")
    print(f"Trades skipped (no S3 BTC data): {n_no_btc}")
    print()
    print(f"Of the {n_lost} losing trades, how many were on the WINNING side at T-N seconds?")
    print(f"  (i.e. exchange-feed BTC said 'win' that close to settlement, but they still lost)")
    print()
    print(f"  {'T-N':>5}  {'count':>6}  {'pct of losses':>14}")
    for off in CHECK_OFFSETS:
        c = flipped_at[off]
        pct = (100.0 * c / n_lost) if n_lost else 0.0
        print(f"  T-{off:>2}s  {c:>6}  {pct:>13.1f}%")

    print()
    print(f"Mirror sanity check — winning trades that were on the LOSING side at T-N:")
    print(f"  {'T-N':>5}  {'count':>6}")
    for off in CHECK_OFFSETS:
        print(f"  T-{off:>2}s  {rescued_at[off]:>6}")

    if args.list_flips:
        print()
        ex = sorted(set(flipped_examples.get(args.flip_window, [])))
        print(f"Slot timestamps with loss-but-winning-at-T-{args.flip_window}s ({len(ex)}):")
        for st in ex:
            iso = datetime.fromtimestamp(st, tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")
            print(f"  {st}  {iso}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
