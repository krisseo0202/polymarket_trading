#!/usr/bin/env python3
"""
Launch bot.py, BTC 1s recorder, and Polymarket orderbook collector in parallel.

All three processes share the same process group so a single Ctrl+C stops everything.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --no-bot          # collectors only
    python scripts/run_all.py --btc-output data/btc_live_1s.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ANSI colors for log prefix
_COLORS = {
    "bot":       "\033[96m",   # cyan
    "btc":       "\033[93m",   # yellow
    "orderbook": "\033[92m",   # green
}
_RESET = "\033[0m"


def _prefix(name: str) -> str:
    color = _COLORS.get(name, "")
    return f"{color}[{name}]{_RESET}"


def launch(
    bot: bool,
    btc_output: str,
    btc_exchange: str,
    ob_output: str,
    ob_interval: float,
    bot_args: list[str],
) -> Dict[str, subprocess.Popen]:
    """Start each subprocess, returning name→Popen mapping.

    Order: bot first (so it initializes feeds/strategy), then collectors.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    procs: Dict[str, subprocess.Popen] = {}

    # Trading bot — start first so it can initialize before collectors add noise
    if bot:
        bot_cmd = [sys.executable, "bot.py"] + bot_args
        procs["bot"] = subprocess.Popen(bot_cmd, cwd=_PROJECT_ROOT, env=env)
        print(f"{_prefix('bot')} PID {procs['bot'].pid}")
        print(f"{_prefix('bot')} Waiting 5s for bot to initialize...")
        time.sleep(5)

    # BTC 1s recorder
    btc_cmd = [
        sys.executable, "scripts/record_btc_1s.py",
        "--output", btc_output,
        "--exchange", btc_exchange,
    ]
    procs["btc"] = subprocess.Popen(btc_cmd, cwd=_PROJECT_ROOT, env=env)
    print(f"{_prefix('btc')} PID {procs['btc'].pid}  →  {btc_output}")

    # Polymarket orderbook collector
    ob_cmd = [
        sys.executable, "scripts/collect_live_window.py",
        "--output", ob_output,
        "--interval", str(ob_interval),
    ]
    procs["orderbook"] = subprocess.Popen(ob_cmd, cwd=_PROJECT_ROOT, env=env)
    print(f"{_prefix('orderbook')} PID {procs['orderbook'].pid}  →  {ob_output}")

    return procs


def wait_and_shutdown(procs: Dict[str, subprocess.Popen]) -> None:
    """Wait for any process to exit, then SIGTERM the rest."""
    try:
        while True:
            for name, proc in list(procs.items()):
                ret = proc.poll()
                if ret is not None:
                    print(f"\n{_prefix(name)} exited with code {ret}")
                    _stop_all(procs, exclude=name)
                    return
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n{'='*50}")
        print("Ctrl+C received — shutting down all processes...")
        print(f"{'='*50}")
        _stop_all(procs)


def _stop_all(procs: Dict[str, subprocess.Popen], exclude: Optional[str] = None) -> None:
    for name, proc in procs.items():
        if name == exclude or proc.poll() is not None:
            continue
        print(f"{_prefix(name)} sending SIGTERM...")
        proc.terminate()

    deadline = time.time() + 8
    for name, proc in procs.items():
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"{_prefix(name)} SIGKILL (did not exit in time)")
            proc.kill()
        code = proc.returncode
        print(f"{_prefix(name)} stopped  (exit code {code})")


def _timestamped_path(base: str) -> str:
    """Insert a UTC timestamp before the file extension.

    Example: data/btc_live_1s.csv → data/btc_live_1s_20260407T220300Z.csv
    """
    root, ext = os.path.splitext(base)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{root}_{ts}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bot + data collectors in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-bot", action="store_true", help="Skip launching bot.py (collectors only)")
    parser.add_argument("--btc-output", default=None, help="BTC 1s CSV path (default: timestamped)")
    parser.add_argument("--btc-exchange", default="coinbase", choices=["coinbase", "binance_us"])
    parser.add_argument("--ob-output", default=None, help="Orderbook CSV path (default: timestamped)")
    parser.add_argument("--ob-interval", type=float, default=1.0, help="Orderbook poll interval (seconds)")
    parser.add_argument("bot_args", nargs="*", help="Extra args forwarded to bot.py")
    args = parser.parse_args()

    # Auto-generate timestamped filenames so each session's data is preserved
    if args.btc_output is None:
        args.btc_output = _timestamped_path("data/btc_live_1s.csv")
    if args.ob_output is None:
        args.ob_output = _timestamped_path("data/live_orderbook_snapshots.csv")

    print("=" * 50)
    print("  Polymarket BTC 5-min — Parallel Launcher")
    print("=" * 50)
    components = ["btc-recorder", "orderbook-collector"]
    if not args.no_bot:
        components.append("bot")
    print(f"  Starting: {', '.join(components)}")
    print(f"\n  Session outputs:")
    print(f"    BTC data:   {args.btc_output}")
    print(f"    OB data:    {args.ob_output}")
    if not args.no_bot:
        print(f"    Bot log:    logs/btc_updown_bot.log")
        print(f"    Decisions:  data/decision_log_*.jsonl (timestamped at bot start)")
    print(f"\n  Press Ctrl+C to stop all.\n")

    procs = launch(
        bot=not args.no_bot,
        btc_output=args.btc_output,
        btc_exchange=args.btc_exchange,
        ob_output=args.ob_output,
        ob_interval=args.ob_interval,
        bot_args=args.bot_args,
    )
    print()
    wait_and_shutdown(procs)
    print("\nAll processes stopped.")


if __name__ == "__main__":
    main()
