#!/usr/bin/env python3
"""
Analyze trades.jsonl and compute per-trade PnL, win rate, calibration, and breakdowns.

Usage:
    python scripts/analyze_trades.py
    python scripts/analyze_trades.py --trades logs/trades.jsonl --state bot_state.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import requests


def load_trades(path: str) -> list:
    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    return trades


def load_slot_outcomes(state_path: str) -> dict:
    """Load slot outcomes from bot_state.json."""
    try:
        with open(state_path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("slot_outcomes", {}).items()}
    except (OSError, json.JSONDecodeError):
        return {}


def fetch_slot_outcome(slot_ts: int) -> str | None:
    """Fetch outcome from Gamma API."""
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
        if not events:
            return None
        market = (events[0].get("markets") or [{}])[0]
        if not market.get("closed"):
            return None
        raw = market.get("outcomePrices", "")
        prices = json.loads(raw) if isinstance(raw, str) else (raw or [])
        if len(prices) < 2:
            return None
        if float(prices[0]) > 0.9:
            return "Up"
        if float(prices[1]) > 0.9:
            return "Down"
    except Exception:
        pass
    return None


def resolve_outcomes(trades: list, cached: dict) -> dict:
    """Ensure all trade slots have outcomes. Fetches missing from API."""
    needed = set()
    for t in trades:
        slot_ts = int(math.floor(t["ts"] / 300) * 300)
        if slot_ts not in cached:
            needed.add(slot_ts)

    for slot_ts in sorted(needed):
        outcome = fetch_slot_outcome(slot_ts)
        if outcome:
            cached[slot_ts] = outcome

    return cached


def compute_pnl(trade: dict, outcome: str | None) -> float | None:
    """Compute PnL for a single trade given market outcome."""
    if outcome is None:
        return None
    action = trade.get("outcome", "")  # YES or NO
    price = trade.get("price", 0)
    size = trade.get("size", 0)

    # Settlement: YES wins if Up, NO wins if Down
    if action == "YES":
        settlement = 0.99 if outcome == "Up" else 0.01
    elif action == "NO":
        settlement = 0.99 if outcome == "Down" else 0.01
    else:
        return None

    return (settlement - price) * size


def bucket(val: float, edges: list) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= val < edges[i + 1]:
            return f"{edges[i]:.2f}-{edges[i+1]:.2f}"
    return f">={edges[-1]:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze trading performance")
    parser.add_argument("--trades", default="logs/trades.jsonl")
    parser.add_argument("--state", default="bot_state.json")
    args = parser.parse_args()

    trades = load_trades(args.trades)
    if not trades:
        print("No trades found.")
        return

    outcomes = load_slot_outcomes(args.state)
    outcomes = resolve_outcomes(trades, outcomes)

    # Compute per-trade PnL
    resolved = []
    unresolved = 0
    for t in trades:
        slot_ts = int(math.floor(t["ts"] / 300) * 300)
        outcome = outcomes.get(slot_ts)
        pnl = compute_pnl(t, outcome)
        if pnl is not None:
            t["_pnl"] = pnl
            t["_outcome"] = outcome
            t["_slot_ts"] = slot_ts
            t["_won"] = pnl > 0
            resolved.append(t)
        else:
            unresolved += 1

    print(f"\n{'='*60}")
    print(f"  TRADE ANALYSIS — {len(trades)} trades, {len(resolved)} resolved, {unresolved} unresolved")
    print(f"{'='*60}\n")

    if not resolved:
        print("No resolved trades to analyze.")
        return

    # Overall stats
    total_pnl = sum(t["_pnl"] for t in resolved)
    wins = [t for t in resolved if t["_won"]]
    losses = [t for t in resolved if not t["_won"]]
    avg_win = sum(t["_pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["_pnl"] for t in losses) / len(losses) if losses else 0
    win_rate = len(wins) / len(resolved)

    print(f"  Total PnL:      ${total_pnl:+.2f}")
    print(f"  Avg PnL/trade:  ${total_pnl / len(resolved):+.2f}")
    print(f"  Win rate:       {win_rate:.1%} ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win:        ${avg_win:+.2f}")
    print(f"  Avg loss:       ${avg_loss:+.2f}")
    print(f"  Win/Loss ratio: {abs(avg_win / avg_loss):.2f}x" if avg_loss != 0 else "")
    print(f"  Max win:        ${max(t['_pnl'] for t in resolved):+.2f}")
    print(f"  Max loss:       ${min(t['_pnl'] for t in resolved):+.2f}")

    # By outcome side (YES vs NO)
    print(f"\n--- By Side ---")
    for side in ["YES", "NO"]:
        side_trades = [t for t in resolved if t.get("outcome") == side]
        if not side_trades:
            continue
        side_pnl = sum(t["_pnl"] for t in side_trades)
        side_wr = sum(1 for t in side_trades if t["_won"]) / len(side_trades)
        print(f"  {side:3s}: {len(side_trades):3d} trades | PnL: ${side_pnl:+8.2f} | Avg: ${side_pnl/len(side_trades):+.2f} | WR: {side_wr:.1%}")

    # By price bucket
    print(f"\n--- By Entry Price ---")
    price_edges = [0.0, 0.30, 0.40, 0.50, 0.60, 1.01]
    price_buckets = defaultdict(list)
    for t in resolved:
        b = bucket(t["price"], price_edges)
        price_buckets[b].append(t)
    for b in sorted(price_buckets.keys()):
        bt = price_buckets[b]
        pnl = sum(t["_pnl"] for t in bt)
        wr = sum(1 for t in bt if t["_won"]) / len(bt)
        print(f"  ${b}: {len(bt):3d} trades | PnL: ${pnl:+8.2f} | Avg: ${pnl/len(bt):+.2f} | WR: {wr:.1%}")

    # By confidence
    print(f"\n--- By Confidence ---")
    conf_buckets = defaultdict(list)
    for t in resolved:
        c = t.get("confidence", 0.5)
        b = f"{c:.1f}"
        conf_buckets[b].append(t)
    for b in sorted(conf_buckets.keys()):
        bt = conf_buckets[b]
        pnl = sum(t["_pnl"] for t in bt)
        wr = sum(1 for t in bt if t["_won"]) / len(bt)
        print(f"  {b}: {len(bt):3d} trades | PnL: ${pnl:+8.2f} | WR: {wr:.1%}")

    # By hour (UTC)
    print(f"\n--- By Hour (UTC) ---")
    hour_buckets = defaultdict(list)
    for t in resolved:
        h = datetime.fromtimestamp(t["ts"], tz=timezone.utc).hour
        hour_buckets[h].append(t)
    for h in sorted(hour_buckets.keys()):
        bt = hour_buckets[h]
        pnl = sum(t["_pnl"] for t in bt)
        wr = sum(1 for t in bt if t["_won"]) / len(bt)
        print(f"  {h:02d}:00: {len(bt):3d} trades | PnL: ${pnl:+8.2f} | WR: {wr:.1%}")

    # Model calibration
    print(f"\n--- Calibration (prob_yes vs actual Up rate) ---")
    cal_buckets = defaultdict(lambda: {"count": 0, "up": 0, "pred_sum": 0.0})
    for t in resolved:
        p = t.get("prob_yes") or t.get("confidence", 0.5)
        b = round(p, 1)
        cal_buckets[b]["count"] += 1
        cal_buckets[b]["pred_sum"] += p
        if t["_outcome"] == "Up":
            cal_buckets[b]["up"] += 1
    print(f"  {'Predicted':>10s}  {'Actual':>8s}  {'Count':>6s}  {'Error':>8s}")
    for b in sorted(cal_buckets.keys()):
        d = cal_buckets[b]
        actual = d["up"] / d["count"]
        pred = d["pred_sum"] / d["count"]
        err = actual - pred
        print(f"  {pred:10.1%}  {actual:8.1%}  {d['count']:6d}  {err:+8.1%}")

    # Feature status impact
    print(f"\n--- By Feature Status ---")
    fs_buckets = defaultdict(list)
    for t in resolved:
        fs = t.get("feature_status", "unknown")
        fs_buckets[fs].append(t)
    for fs in sorted(fs_buckets.keys()):
        bt = fs_buckets[fs]
        pnl = sum(t["_pnl"] for t in bt)
        wr = sum(1 for t in bt if t["_won"]) / len(bt)
        print(f"  {fs:30s}: {len(bt):3d} trades | PnL: ${pnl:+8.2f} | WR: {wr:.1%}")

    print()


if __name__ == "__main__":
    main()
