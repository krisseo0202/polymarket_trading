#!/usr/bin/env python3
"""
Signal diagnostic: show cycle timing, model vs market probability,
and whether a trade would trigger — all in one deterministic snapshot.

Usage:
    # Live mode (fetches real market data):
    python signal_diagnostic.py

    # Mock mode (fully deterministic, no network):
    python signal_diagnostic.py --mock --start-price 84000 --current-price 84100 \
        --market-yes-mid 0.52 --time-left 180 --vol 0.65 --seed 42

    # Override vol/seed in live mode:
    python signal_diagnostic.py --vol 0.65 --seed 42
"""

import argparse
import math
import sys
import time
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from quant_desk import simulate_up_prob


# ---------------------------------------------------------------------------
# Cycle timing
# ---------------------------------------------------------------------------

def get_cycle_info(interval: int = 300) -> dict:
    """Return wall-clock-aligned cycle timing info."""
    now = time.time()
    slot_start = int(math.floor(now / interval) * interval)
    slot_end = slot_start + interval
    elapsed = now - slot_start
    remaining = slot_end - now
    return {
        "slot_start": slot_start,
        "slot_end": slot_end,
        "elapsed_sec": elapsed,
        "remaining_sec": remaining,
        "interval": interval,
        "pct_elapsed": elapsed / interval * 100,
    }


# ---------------------------------------------------------------------------
# Live market fetch (thin wrapper around bot.py helpers)
# ---------------------------------------------------------------------------

def fetch_live_context(keywords=None, min_volume=0):
    """Fetch current BTC Up/Down market prices.

    Returns dict with yes_mid, no_mid, yes_bid, yes_ask, no_bid, no_ask,
    market_id, yes_token_id, no_token_id — or None if market not found.
    """
    import requests
    import json
    import logging

    log = logging.getLogger("diagnostic")
    keywords = keywords or ["Bitcoin", "Up or Down"]
    lower_kws = [k.lower() for k in keywords]

    # Slug-based lookup
    slot = int(math.floor(time.time() / 300) * 300)
    slug = f"btc-updown-5m-{slot}"
    market_info = None

    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug}, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        events = data if isinstance(data, list) else [data]
        for event in events:
            for m in event.get("markets", []):
                tids = json.loads(m.get("clobTokenIds", "[]"))
                if len(tids) >= 2:
                    market_info = {
                        "market_id": m.get("id", ""),
                        "yes_token_id": tids[0],
                        "no_token_id": tids[1],
                    }
                    break
            if market_info:
                break
    except Exception as e:
        log.warning(f"Slug lookup failed: {e}")

    if not market_info:
        # Keyword fallback
        try:
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"active": "true", "closed": "false", "limit": 100},
                timeout=10,
            )
            resp.raise_for_status()
            for event in resp.json():
                title = event.get("title", "").lower()
                if not all(kw in title for kw in lower_kws):
                    continue
                for m_item in event.get("markets", []):
                    tids = json.loads(m_item.get("clobTokenIds", "[]"))
                    if len(tids) >= 2:
                        market_info = {
                            "market_id": m_item.get("id", ""),
                            "yes_token_id": tids[0],
                            "no_token_id": tids[1],
                        }
                        break
                if market_info:
                    break
        except Exception as e:
            log.warning(f"Keyword search failed: {e}")

    if not market_info:
        return None

    # Fetch order books
    def _top(token_id):
        try:
            r = requests.get(
                "https://clob.polymarket.com/book",
                params={"token_id": token_id}, timeout=5,
            )
            r.raise_for_status()
            d = r.json()
            bids = d.get("bids") or []
            asks = d.get("asks") or []
            bid = float(bids[0]["p"]) if bids else None
            ask = float(asks[0]["p"]) if asks else None
            return bid, ask
        except Exception:
            return None, None

    yb, ya = _top(market_info["yes_token_id"])
    nb, na = _top(market_info["no_token_id"])

    def _mid(b, a):
        if b is not None and a is not None:
            return (b + a) / 2
        return b or a

    return {
        **market_info,
        "yes_bid": yb, "yes_ask": ya, "yes_mid": _mid(yb, ya),
        "no_bid": nb, "no_ask": na, "no_mid": _mid(nb, na),
    }


# ---------------------------------------------------------------------------
# Trade decision logic
# ---------------------------------------------------------------------------

def evaluate_trade(
    model_up_prob: float,
    market_yes_mid: float,
    edge_threshold: float = 0.03,
    min_confidence: float = 0.7,
    position_size_usdc: float = 20.0,
    time_remaining_sec: float = 300.0,
    max_hold_seconds: float = 240.0,
) -> dict:
    """Decide whether a trade triggers and on which side.

    Returns a dict describing the decision:
      - would_trade: bool
      - side: "YES" | "NO" | None
      - edge: float (model - market on the chosen side)
      - reason: str
    """
    market_no_mid = 1.0 - market_yes_mid
    model_down_prob = 1.0 - model_up_prob

    yes_edge = model_up_prob - market_yes_mid
    no_edge = model_down_prob - market_no_mid  # equivalent to market_yes_mid - model_up_prob

    # Pick the side with the larger positive edge
    if yes_edge >= no_edge:
        best_side, best_edge = "YES", yes_edge
    else:
        best_side, best_edge = "NO", no_edge

    reasons = []

    # Check edge threshold
    if best_edge < edge_threshold:
        reasons.append(f"edge {best_edge:.4f} < threshold {edge_threshold:.4f}")

    # Check time remaining
    if time_remaining_sec < max_hold_seconds:
        reasons.append(
            f"time_remaining {time_remaining_sec:.0f}s < max_hold {max_hold_seconds:.0f}s"
        )

    # Confidence proxy: how far is model_up_prob from 0.5?
    confidence = 0.5 + abs(model_up_prob - 0.5)
    if confidence < min_confidence:
        reasons.append(f"confidence {confidence:.4f} < min {min_confidence:.4f}")

    would_trade = len(reasons) == 0

    return {
        "would_trade": would_trade,
        "side": best_side if would_trade else None,
        "edge": best_edge,
        "confidence": confidence,
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "rejection_reasons": reasons,
        "position_size_usdc": position_size_usdc if would_trade else 0.0,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_report(cycle: dict, model_prob: float, market_yes_mid: float,
                  trade: dict, n_paths: int, seed, live: bool) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  SIGNAL DIAGNOSTIC")
    lines.append("=" * 60)

    # Cycle timing
    e_m, e_s = divmod(int(cycle["elapsed_sec"]), 60)
    r_m, r_s = divmod(int(cycle["remaining_sec"]), 60)
    lines.append(f"\n  Cycle timing ({cycle['interval']}s interval)")
    lines.append(f"    Elapsed:    {e_m:02d}:{e_s:02d}  ({cycle['pct_elapsed']:.0f}%)")
    lines.append(f"    Remaining:  {r_m:02d}:{r_s:02d}")

    # Probabilities
    lines.append(f"\n  Probabilities")
    lines.append(f"    Model  P(up):  {model_prob:.4f}  ({model_prob:.1%})")
    lines.append(f"    Market YES:    {market_yes_mid:.4f}  ({market_yes_mid:.1%})")
    lines.append(f"    Market NO:     {1-market_yes_mid:.4f}  ({1-market_yes_mid:.1%})")
    lines.append(f"    YES edge:     {trade['yes_edge']:+.4f}")
    lines.append(f"    NO  edge:     {trade['no_edge']:+.4f}")

    # Trade decision
    lines.append(f"\n  Trade decision")
    if trade["would_trade"]:
        lines.append(f"    >>> TRADE TRIGGERED: BUY {trade['side']}")
        lines.append(f"    Edge:       {trade['edge']:+.4f}")
        lines.append(f"    Confidence: {trade['confidence']:.4f}")
        lines.append(f"    Size:       ${trade['position_size_usdc']:.2f} USDC")
    else:
        lines.append(f"    >>> NO TRADE")
        for r in trade["rejection_reasons"]:
            lines.append(f"    - {r}")

    # Sim params
    lines.append(f"\n  Simulation")
    lines.append(f"    Paths: {n_paths:,}")
    lines.append(f"    Seed:  {seed}")
    lines.append(f"    Mode:  {'LIVE' if live else 'MOCK'}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Signal diagnostic snapshot")

    # Mock mode inputs
    parser.add_argument("--mock", action="store_true",
                        help="Use mock inputs instead of live market data")
    parser.add_argument("--start-price", type=float, default=84000.0,
                        help="BTC start price for the 5-min window (mock mode)")
    parser.add_argument("--current-price", type=float, default=84050.0,
                        help="BTC current price (mock mode)")
    parser.add_argument("--market-yes-mid", type=float, default=0.52,
                        help="YES token mid-price from market (mock mode)")
    parser.add_argument("--time-left", type=float, default=None,
                        help="Override time remaining in seconds")

    # Simulation params
    parser.add_argument("--vol", type=float, default=0.65,
                        help="Annualised volatility (default: 0.65)")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="Annualised drift (default: 0)")
    parser.add_argument("--n-paths", type=int, default=10000,
                        help="Monte Carlo paths (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")

    # Trade thresholds
    parser.add_argument("--edge-threshold", type=float, default=0.03,
                        help="Minimum edge to trigger trade (default: 0.03)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum confidence (default: 0.7)")
    parser.add_argument("--position-size", type=float, default=20.0,
                        help="Position size in USDC (default: 20)")
    parser.add_argument("--max-hold", type=float, default=240.0,
                        help="Max hold time in seconds (default: 240)")

    args = parser.parse_args()

    # Seed RNG
    if args.seed is not None:
        np.random.seed(args.seed)

    # Cycle timing
    cycle = get_cycle_info()
    time_left = args.time_left if args.time_left is not None else cycle["remaining_sec"]

    if args.mock:
        # ---- MOCK MODE ----
        start_price = args.start_price
        current_price = args.current_price
        market_yes_mid = args.market_yes_mid
    else:
        # ---- LIVE MODE ----
        ctx = fetch_live_context()
        if ctx is None:
            print("ERROR: Could not find BTC Up/Down market. Use --mock for offline testing.")
            sys.exit(1)

        market_yes_mid = ctx["yes_mid"]
        if market_yes_mid is None:
            print("ERROR: YES order book is empty. Use --mock for offline testing.")
            sys.exit(1)

        # For BTC price we'd need an external feed; approximate from the
        # market probability (the market IS the BTC price derivative).
        # In live mode, start_price and current_price are synthetic —
        # the model prob is what matters.
        # Use a nominal BTC price and shift by market-implied direction.
        start_price = 84000.0
        current_price = start_price * (1 + (market_yes_mid - 0.5) * 0.01)

        print(f"  Live market found: YES={ctx['yes_mid']:.4f} NO={ctx['no_mid']:.4f}")
        print(f"  Market: {ctx['market_id']}")
        print(f"  YES token: {ctx['yes_token_id'][:16]}...")
        print()

    # Run MC simulation
    model_up_prob = simulate_up_prob(
        start_price=start_price,
        current_price=current_price,
        time_left_sec=time_left,
        vol=args.vol,
        mu=args.mu,
        n_paths=args.n_paths,
    )

    # Evaluate trade
    trade = evaluate_trade(
        model_up_prob=model_up_prob,
        market_yes_mid=market_yes_mid,
        edge_threshold=args.edge_threshold,
        min_confidence=args.min_confidence,
        position_size_usdc=args.position_size,
        time_remaining_sec=time_left,
        max_hold_seconds=args.max_hold,
    )

    # Display
    report = format_report(
        cycle=cycle,
        model_prob=model_up_prob,
        market_yes_mid=market_yes_mid,
        trade=trade,
        n_paths=args.n_paths,
        seed=args.seed,
        live=not args.mock,
    )
    print(report)

    # Machine-readable exit code: 0 = trade triggered, 1 = no trade
    sys.exit(0 if trade["would_trade"] else 1)


if __name__ == "__main__":
    main()
