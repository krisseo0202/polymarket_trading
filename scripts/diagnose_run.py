"""Post-run diagnostics for the BTC 5-minute paper-trading bot.

Joins the JSONL decision log with bot.log Settlement records and breaks
down realized PnL by edge / confidence / spread / TTE / moneyness /
side. Also produces a calibration table (predicted prob vs actual win
rate) and a concise summary.

Usage::

    python scripts/diagnose_run.py \\
        --decision-log data/2026-04-25/decision_log_20260425T135417Z.jsonl \\
        --bot-log logs/btc_updown_bot.log \\
        --since "2026-04-25 06:54:00"

Output: prints the report to stdout. Designed to be run after a session
to catch problems before adding more capital or features.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


_SETTLEMENT_RE = re.compile(
    r"SELL: (YES|NO) \w+ ([\d.]+)sh @ ([\d.]+) "
    r"\(entry ([\d.]+)\) → realized ([+-]?[\d.]+) "
    r"\(outcome: (Up|Down)\)"
)
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_PHAT_RE = re.compile(r"p_hat=(-?\d+\.\d+)")


@dataclass
class Trade:
    """Joined view: decision-log entry intent + bot-log settlement."""
    slot_ts: int
    side: str
    entry_price: float
    exit_price: float
    size: float
    realized_pnl: float
    outcome: str  # "Up" / "Down"
    p_hat: float
    edge: float  # raw edge for the chosen side at decision time
    spread_pct: float
    tte: float
    moneyness: float
    confidence: float

    @property
    def p_chosen(self) -> float:
        return self.p_hat if self.side == "YES" else 1.0 - self.p_hat

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > 0


def _parse_settlements(bot_log: Path, since_ts: float) -> List[dict]:
    out: List[dict] = []
    with bot_log.open() as f:
        for line in f:
            if "Settlement SELL" not in line:
                continue
            m_ts = _TS_RE.match(line)
            if not m_ts:
                continue
            ts = datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            if ts < since_ts:
                continue
            m = _SETTLEMENT_RE.search(line)
            if not m:
                continue
            side, sz, exit_p, entry_p, pnl, outcome = m.groups()
            out.append({
                "ts": ts, "side": side, "size": float(sz),
                "entry": float(entry_p), "exit": float(exit_p),
                "pnl": float(pnl), "outcome": outcome,
            })
    return out


def _parse_decisions(decision_log: Path) -> List[dict]:
    out: List[dict] = []
    with decision_log.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("action"):
                continue
            m = _PHAT_RE.search(r["action"]["reason"])
            if not m:
                continue
            out.append({
                "slot_ts": r["slot_ts"],
                "side": r["action"]["side"],
                "price": r["action"]["price"],
                "size": r["action"]["size"],
                "p_hat": float(m.group(1)),
                "edge_yes": r.get("edge_yes", 0.0) or 0.0,
                "edge_no": r.get("edge_no", 0.0) or 0.0,
                "tte": r.get("tte", 0.0) or 0.0,
                "spread_pct": r.get("f_yes_spread_pct", 0.0) or 0.0,
                "moneyness": r.get("f_moneyness", 0.0) or 0.0,
                "confidence": r["action"].get("confidence", 0.0) or 0.0,
            })
    return out


def _join(settlements: Sequence[dict], decisions: Sequence[dict]) -> List[Trade]:
    """Match each settlement to the closest preceding decision on the same side."""
    by_side_slot: Dict[tuple, List[dict]] = defaultdict(list)
    for d in decisions:
        by_side_slot[(d["side"], d["slot_ts"])].append(d)

    trades: List[Trade] = []
    for s in settlements:
        # Settlement fires at slot_end ≈ slot_open + 300s. Look back up
        # to 700s for the matching decision (covers the whole slot plus
        # margin for clock skew).
        cands = [
            d for d in decisions
            if d["side"] == s["side"]
            and d["slot_ts"] <= s["ts"]
            and s["ts"] - d["slot_ts"] < 700
        ]
        if not cands:
            continue
        # Prefer same entry price; fallback to most recent decision.
        same_price = [c for c in cands if abs(c["price"] - s["entry"]) < 0.001]
        d = (same_price or cands)[-1]
        edge = d["edge_yes"] if d["side"] == "YES" else d["edge_no"]
        trades.append(Trade(
            slot_ts=int(d["slot_ts"]), side=s["side"],
            entry_price=s["entry"], exit_price=s["exit"], size=s["size"],
            realized_pnl=s["pnl"], outcome=s["outcome"],
            p_hat=d["p_hat"], edge=edge, spread_pct=d["spread_pct"],
            tte=d["tte"], moneyness=d["moneyness"], confidence=d["confidence"],
        ))
    return trades


def _print_bucket_table(
    title: str, trades: Sequence[Trade], bucket_fn, buckets,
    extra_cols: Optional[Sequence[str]] = None,
) -> None:
    print(f"=== {title} ===")
    rows = []
    for label, lo, hi in buckets:
        sel = [t for t in trades if lo <= bucket_fn(t) < hi]
        if not sel:
            continue
        n = len(sel)
        win_rate = sum(1 for t in sel if t.is_win) / n
        avg_pnl = sum(t.realized_pnl for t in sel) / n
        sum_pnl = sum(t.realized_pnl for t in sel)
        rows.append((label, n, win_rate, avg_pnl, sum_pnl))
    if not rows:
        print("  (no trades)")
        return
    print(f"  {'bucket':<14} {'n':>4} {'win%':>6} {'avg_pnl':>9} {'sum_pnl':>10}")
    for label, n, wr, ap, sp in rows:
        print(f"  {label:<14} {n:>4} {wr*100:>5.1f}% {ap:>+8.2f} {sp:>+9.2f}")
    print()


def _print_split(title: str, trades: Sequence[Trade], key_fn, key_label="key") -> None:
    print(f"=== {title} ===")
    by_key: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        by_key[str(key_fn(t))].append(t)
    print(f"  {key_label:<10} {'n':>4} {'win%':>6} {'avg_pnl':>9} {'sum_pnl':>10}")
    for k in sorted(by_key):
        sel = by_key[k]
        n = len(sel)
        wr = sum(1 for t in sel if t.is_win) / n
        ap = sum(t.realized_pnl for t in sel) / n
        sp = sum(t.realized_pnl for t in sel)
        print(f"  {k:<10} {n:>4} {wr*100:>5.1f}% {ap:>+8.2f} {sp:>+9.2f}")
    print()


def _print_calibration(trades: Sequence[Trade]) -> None:
    """Calibration: predicted probability bucket → actual win rate.

    For YES trades the prediction is p_hat; for NO trades it's 1-p_hat
    (the probability of the side actually taken). A well-calibrated model
    has avg_p ≈ win%.
    """
    print("=== CALIBRATION (predicted vs actual) ===")
    print(f"  {'p_chosen':<14} {'n':>4} {'avg_p':>7} {'win%':>6} {'avg_pnl':>9} {'gap':>7}")
    buckets = [
        ("[0.50,0.54)", 0.50, 0.54),
        ("[0.54,0.58)", 0.54, 0.58),
        ("[0.58,0.62)", 0.58, 0.62),
        ("[0.62,0.68)", 0.62, 0.68),
        ("[0.68,0.75)", 0.68, 0.75),
        ("[0.75,1.00)", 0.75, 1.00),
    ]
    for label, lo, hi in buckets:
        sel = [t for t in trades if lo <= t.p_chosen < hi]
        if not sel:
            continue
        n = len(sel)
        avg_p = statistics.mean(t.p_chosen for t in sel)
        win_rate = sum(1 for t in sel if t.is_win) / n
        avg_pnl = sum(t.realized_pnl for t in sel) / n
        gap = win_rate - avg_p  # negative = overconfident
        print(f"  {label:<14} {n:>4} {avg_p:>7.3f} {win_rate*100:>5.1f}% {avg_pnl:>+8.2f} {gap:>+6.3f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decision-log", required=True, type=Path)
    parser.add_argument("--bot-log", required=True, type=Path)
    parser.add_argument(
        "--since", default="1970-01-01 00:00:00",
        help="Only consider settlements at or after this timestamp (UTC).",
    )
    args = parser.parse_args()

    since_ts = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S").timestamp()
    settlements = _parse_settlements(args.bot_log, since_ts=since_ts)
    decisions = _parse_decisions(args.decision_log)
    trades = _join(settlements, decisions)

    print()
    print(f"Decision-log entries: {len(decisions)}")
    print(f"Settlements:          {len(settlements)} (since {args.since})")
    print(f"Joined trades:        {len(trades)}  ({len(trades)/max(1,len(settlements))*100:.0f}% match rate)")
    print()
    if not trades:
        print("No trades to analyze.")
        return

    total_pnl = sum(t.realized_pnl for t in trades)
    wins = sum(1 for t in trades if t.is_win)
    print(f"Total PnL: {total_pnl:+.2f} | Wins: {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%)")
    print()

    _print_calibration(trades)

    _print_split("BY SIDE", trades, lambda t: t.side, "side")

    _print_bucket_table(
        "BY EDGE (raw, chosen side)", trades, lambda t: t.edge,
        [
            ("[0.00,0.02)", 0.0, 0.02),
            ("[0.02,0.04)", 0.02, 0.04),
            ("[0.04,0.06)", 0.04, 0.06),
            ("[0.06,0.10)", 0.06, 0.10),
            ("[0.10,0.20)", 0.10, 0.20),
            ("[0.20,1.00)", 0.20, 1.0),
        ],
    )

    _print_bucket_table(
        "BY SPREAD %", trades, lambda t: t.spread_pct,
        [
            ("[0.00,0.01)", 0.0, 0.01),
            ("[0.01,0.02)", 0.01, 0.02),
            ("[0.02,0.04)", 0.02, 0.04),
            ("[0.04,0.10)", 0.04, 0.10),
            ("[0.10,1.00)", 0.10, 1.0),
        ],
    )

    _print_bucket_table(
        "BY TTE (seconds)", trades, lambda t: t.tte,
        [
            ("[0,30)",   0.0,  30.0),
            ("[30,60)",  30.0, 60.0),
            ("[60,120)", 60.0, 120.0),
            ("[120,200)",120.0,200.0),
            ("[200,300)",200.0,300.0),
        ],
    )

    _print_bucket_table(
        "BY |MONEYNESS|", trades, lambda t: abs(t.moneyness),
        [
            ("[0,1e-4)",   0.0,    1e-4),
            ("[1e-4,1e-3)",1e-4,   1e-3),
            ("[1e-3,5e-3)",1e-3,   5e-3),
            ("[5e-3,inf)", 5e-3,   1.0),
        ],
    )


if __name__ == "__main__":
    main()
