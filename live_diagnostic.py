#!/usr/bin/env python3
"""
Live Signal Diagnostic — continuously updating BTC price (Binance WS) vs
Polymarket market probability, with MC model comparison and trade trigger.

Usage:
    python live_diagnostic.py
    python live_diagnostic.py --vol 0.65 --n-paths 10000 --refresh 5
    python live_diagnostic.py --edge-threshold 0.02 --min-confidence 0.6
"""

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))

from quant_desk import simulate_up_prob
from signal_diagnostic import evaluate_trade, get_cycle_info
from bot import find_btc_updown_market, _fetch_book_top
from src.utils.btc_feed import BtcPriceFeed


# ---------------------------------------------------------------------------
# Binance REST: fetch current 5m kline open price
# ---------------------------------------------------------------------------

def fetch_slot_open_price() -> float | None:
    """Fetch the open price of the current 5-minute kline from Binance REST."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "5m", "limit": 1},
            timeout=5,
        )
        r.raise_for_status()
        kline = r.json()[0]
        return float(kline[1])  # index 1 = open price
    except Exception as e:
        logging.getLogger("live_diag").warning(f"Kline fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Polymarket polling (uses bot.py's 3-tier market discovery)
# ---------------------------------------------------------------------------

# Cache: token IDs are stable within a 5-min slot, so avoid re-discovering every poll
_cached_market: dict | None = None
_cached_slot: int = 0


def poll_polymarket() -> dict | None:
    """Fetch current market YES/NO mid-prices via bot.py's 3-tier discovery.

    Returns dict with yes_mid, no_mid, yes_bid, yes_ask, no_bid, no_ask,
    market_id, yes_token_id, no_token_id — or None if market not found.
    """
    global _cached_market, _cached_slot
    log = logging.getLogger("live_diag")

    current_slot = int(math.floor(time.time() / 300) * 300)

    # Re-discover market if slot changed or no cache
    if _cached_market is None or current_slot != _cached_slot:
        log.info(f"Discovering market for slot {current_slot}...")
        _cached_market = find_btc_updown_market(
            keywords=["Bitcoin", "Up or Down"],
            min_volume=0,
            logger=log,
        )
        _cached_slot = current_slot
        if _cached_market is None:
            log.warning("All 3 discovery tiers failed — no market found")
            return None

    mkt = _cached_market
    try:
        yes_bid, yes_ask, no_bid, no_ask = _fetch_book_top(
            mkt["yes_token_id"], mkt["no_token_id"], logger=log,
        )
    except Exception as e:
        log.warning(f"Book fetch error: {e}")
        return None

    yes_mid = (yes_bid + yes_ask) / 2 if yes_bid is not None and yes_ask is not None else None
    no_mid = (no_bid + no_ask) / 2 if no_bid is not None and no_ask is not None else None

    return {
        "yes_mid": yes_mid,
        "no_mid": no_mid,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "market_id": mkt["market_id"],
        "yes_token_id": mkt["yes_token_id"],
        "no_token_id": mkt["no_token_id"],
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def render_display(
    cycle: dict,
    slot_open: float | None,
    btc_mid: float | None,
    market_ctx: dict | None,
    model_prob: float | None,
    trade: dict | None,
    n_paths: int,
    vol: float,
    feed_healthy: bool,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "[LIVE]" if feed_healthy else "[WAIT]"

    e_m, e_s = divmod(int(cycle["elapsed_sec"]), 60)
    r_m, r_s = divmod(int(cycle["remaining_sec"]), 60)
    pct = cycle["pct_elapsed"]

    lines = []
    lines.append("")
    lines.append("\033[2J\033[H")  # clear screen + cursor home
    lines.append("═" * 62)
    lines.append(f"  LIVE SIGNAL DIAGNOSTIC              {status} {now_str}")
    lines.append("═" * 62)

    # Cycle timing
    lines.append(f"  Cycle:  {e_m:02d}:{e_s:02d} elapsed / {r_m:02d}:{r_s:02d} remaining  ({pct:.0f}%)")
    lines.append(f"  Slot:   {cycle['slot_start']} → {cycle['slot_end']}")

    # BTC Price
    lines.append("")
    lines.append("  BTC Price")
    if slot_open is not None:
        lines.append(f"    Slot open:    ${slot_open:,.2f}")
    else:
        lines.append(f"    Slot open:    --")
    if btc_mid is not None:
        lines.append(f"    Current:      ${btc_mid:,.2f}", )
        if slot_open is not None and slot_open > 0:
            chg = (btc_mid - slot_open) / slot_open * 100
            lines[-1] += f"  ({chg:+.3f}%)"
    else:
        lines.append(f"    Current:      -- (waiting for Binance WS)")

    # Probabilities
    lines.append("")
    lines.append("  Probabilities")
    if model_prob is not None:
        lines.append(f"    Model  P(up):   {model_prob:.4f}  ({model_prob:.1%})    [{n_paths:,} paths, σ={vol}]")
    else:
        lines.append(f"    Model  P(up):   --  (need BTC price)")

    if market_ctx and market_ctx.get("yes_mid") is not None:
        ym = market_ctx["yes_mid"]
        nm = market_ctx.get("no_mid") or (1.0 - ym)
        lines.append(f"    Market YES:     {ym:.4f}  ({ym:.1%})")
        lines.append(f"    Market NO:      {nm:.4f}  ({nm:.1%})")
    else:
        lines.append(f"    Market YES:     -- (market not found)")
        lines.append(f"    Market NO:      --")

    # Edge analysis
    lines.append("")
    lines.append("  Edge Analysis")
    if trade is not None:
        yes_e = trade["yes_edge"]
        no_e = trade["no_edge"]
        best_yes = " ← best" if yes_e >= no_e else ""
        best_no = " ← best" if no_e > yes_e else ""
        lines.append(f"    YES edge:      {yes_e:+.4f}{best_yes}")
        lines.append(f"    NO  edge:      {no_e:+.4f}{best_no}")
    else:
        lines.append(f"    --  (insufficient data)")

    # Trade decision
    lines.append("")
    lines.append("  Trade Decision")
    if trade is not None:
        if trade["would_trade"]:
            lines.append(f"    >>> TRADE TRIGGERED: BUY {trade['side']}")
            lines.append(f"    Edge: {trade['edge']:+.4f} | Confidence: {trade['confidence']:.4f} | Size: ${trade['position_size_usdc']:.2f}")
        else:
            lines.append(f"    >>> NO TRADE")
            for r in trade["rejection_reasons"]:
                lines.append(f"    - {r}")
    else:
        lines.append(f"    >>> WAITING FOR DATA")

    lines.append("═" * 62)
    lines.append("  Ctrl-C to exit")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main_loop(args):
    log = logging.getLogger("live_diag")

    # Start BTC price feed
    feed = BtcPriceFeed(logger=logging.getLogger("btc_feed"))
    feed.start()
    log.info("BTC price feed starting...")

    # Initial slot open price from Binance REST
    slot_open = fetch_slot_open_price()
    current_slot = int(math.floor(time.time() / 300) * 300)

    # Cache for Polymarket (re-fetch every cycle)
    market_ctx = None
    last_market_fetch = 0.0

    if args.seed is not None:
        np.random.seed(args.seed)

    try:
        while True:
            now = time.time()
            cycle = get_cycle_info()

            # Detect slot rollover → refresh slot open price
            new_slot = cycle["slot_start"]
            if new_slot != current_slot:
                current_slot = new_slot
                slot_open = fetch_slot_open_price()
                market_ctx = None  # force market re-fetch on new slot
                log.info(f"Slot rollover → {new_slot}, new open: {slot_open}")

            # Refresh Polymarket data every 10s (less aggressive than BTC)
            if now - last_market_fetch > 10.0:
                market_ctx = poll_polymarket()
                last_market_fetch = now

            # Get current BTC mid from WebSocket feed
            btc_mid = feed.get_latest_mid()

            # Run MC simulation if we have the data
            model_prob = None
            trade = None
            if btc_mid is not None and slot_open is not None:
                model_prob = simulate_up_prob(
                    start_price=slot_open,
                    current_price=btc_mid,
                    time_left_sec=max(cycle["remaining_sec"], 0.1),
                    vol=args.vol,
                    mu=args.mu,
                    n_paths=args.n_paths,
                )

                # Evaluate trade if we have market data
                if market_ctx and market_ctx.get("yes_mid") is not None:
                    trade = evaluate_trade(
                        model_up_prob=model_prob,
                        market_yes_mid=market_ctx["yes_mid"],
                        edge_threshold=args.edge_threshold,
                        min_confidence=args.min_confidence,
                        position_size_usdc=args.position_size,
                        time_remaining_sec=cycle["remaining_sec"],
                        max_hold_seconds=args.max_hold,
                    )

            # Render
            display = render_display(
                cycle=cycle,
                slot_open=slot_open,
                btc_mid=btc_mid,
                market_ctx=market_ctx,
                model_prob=model_prob,
                trade=trade,
                n_paths=args.n_paths,
                vol=args.vol,
                feed_healthy=feed.is_healthy(),
            )
            print(display, end="", flush=True)

            time.sleep(args.refresh)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        feed.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live signal diagnostic with Binance WS + Polymarket"
    )
    parser.add_argument("--vol", type=float, default=0.65,
                        help="Annualised volatility (default: 0.65)")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="Annualised drift (default: 0.0)")
    parser.add_argument("--n-paths", type=int, default=10000,
                        help="MC simulation paths (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--refresh", type=float, default=5.0,
                        help="Display refresh interval in seconds (default: 5.0)")
    parser.add_argument("--edge-threshold", type=float, default=0.03,
                        help="Minimum edge to trigger (default: 0.03)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum confidence (default: 0.7)")
    parser.add_argument("--position-size", type=float, default=20.0,
                        help="USDC per trade (default: 20.0)")
    parser.add_argument("--max-hold", type=float, default=240.0,
                        help="Max hold seconds (default: 240)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    main_loop(args)


if __name__ == "__main__":
    main()
