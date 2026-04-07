#!/usr/bin/env python3
"""
Record live BTC 1-second OHLCV bars via Coinbase WebSocket.

Saves rows with: timestamp, open, high, low, close, volume
Volume is tick count per second (Coinbase ticker doesn't provide true volume).

Usage:
    python scripts/record_btc_1s.py
    python scripts/record_btc_1s.py --output data/btc_live_1s.csv
    python scripts/record_btc_1s.py --exchange binance_us --symbol btcusd
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.btc_feed import BtcPriceFeed

_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    print("\nShutting down...")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def run(output: str, exchange: str, symbol: str):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    file_exists = os.path.exists(output) and os.path.getsize(output) > 0
    f = open(output, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(CSV_COLUMNS)
        f.flush()

    feed = BtcPriceFeed(symbol=symbol, exchange=exchange).start()

    # Wait for first tick
    print(f"Connecting to {exchange} ({symbol})...", flush=True)
    deadline = time.time() + 15
    while feed.get_latest_mid() is None and time.time() < deadline:
        time.sleep(0.1)
    if feed.get_latest_mid() is None:
        print("ERROR: No data received after 15s. Check connection.")
        feed.stop()
        f.close()
        return

    print(f"Recording 1s OHLCV bars -> {output}")
    print("Press Ctrl+C to stop.\n")

    bar_open = bar_high = bar_low = bar_close = None
    tick_count = 0
    current_second = int(time.time())

    try:
        while not _shutdown:
            mid = feed.get_latest_mid()
            if mid is None:
                time.sleep(0.01)
                continue

            now_second = int(time.time())

            # New second — flush previous bar
            if now_second != current_second:
                if bar_open is not None:
                    writer.writerow([
                        current_second,
                        round(bar_open, 2),
                        round(bar_high, 2),
                        round(bar_low, 2),
                        round(bar_close, 2),
                        tick_count,
                    ])
                    f.flush()
                    print(
                        f"  {current_second}  O={bar_open:.2f}  H={bar_high:.2f}  "
                        f"L={bar_low:.2f}  C={bar_close:.2f}  ticks={tick_count}",
                        flush=True,
                    )

                # Reset bar
                bar_open = mid
                bar_high = mid
                bar_low = mid
                bar_close = mid
                tick_count = 1
                current_second = now_second
            else:
                # Update bar
                if bar_open is None:
                    bar_open = mid
                    bar_high = mid
                    bar_low = mid
                bar_high = max(bar_high, mid)
                bar_low = min(bar_low, mid)
                bar_close = mid
                tick_count += 1

            time.sleep(0.05)
    finally:
        # Flush last bar
        if bar_open is not None:
            writer.writerow([
                current_second,
                round(bar_open, 2),
                round(bar_high, 2),
                round(bar_low, 2),
                round(bar_close, 2),
                tick_count,
            ])
            f.flush()

        feed.stop()
        f.close()
        print(f"\nSaved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Record live BTC 1s OHLCV bars")
    parser.add_argument("--output", default="data/btc_live_1s.csv", help="Output CSV path")
    parser.add_argument("--exchange", default="coinbase", choices=["coinbase", "binance_us"])
    parser.add_argument("--symbol", default=None, help="Symbol (default: BTC-USD for coinbase, btcusd for binance_us)")
    args = parser.parse_args()

    symbol = args.symbol
    if symbol is None:
        symbol = "BTC-USD" if args.exchange == "coinbase" else "btcusd"

    run(args.output, args.exchange, symbol)


if __name__ == "__main__":
    main()
