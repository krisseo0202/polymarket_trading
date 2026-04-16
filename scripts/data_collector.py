#!/usr/bin/env python3
"""
Unified data collector: runs all three collection pipelines in parallel,
partitions output by hour, and maintains a slot-key mapping.

Supports multi-asset collection (BTC, ETH, SOL, DOGE, XRP).

Output layout:
    data/YYYY-MM-DD/HH/{asset}_1s.csv
    data/YYYY-MM-DD/HH/snapshots_{asset}.jsonl
    data/YYYY-MM-DD/HH/history_{asset}.csv
    data/YYYY-MM-DD/HH/slot_key_map.csv

Processes per asset:
  1. record_crypto_1s.py   — 1-second OHLCV bars (Coinbase WebSocket)
  2. collect_snapshots.py  — orderbook snapshots + Chainlink prices every 5s
  3. collect_history.py    — closed 5-min market outcomes (hourly batch)

At each hour boundary the daemons are gracefully restarted with new output
paths so each hour's data lands in its own directory.

Usage:
    python scripts/data_collector.py
    python scripts/data_collector.py --assets BTC ETH SOL
    python scripts/data_collector.py --assets BTC ETH SOL DOGE XRP --no-history
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ANSI colors for log prefix
_COLORS = {
    "price":     "\033[93m",  # yellow
    "snapshots": "\033[92m",  # green
    "history":   "\033[96m",  # cyan
    "mapper":    "\033[95m",  # magenta
    "rotate":    "\033[91m",  # red
    "s3":        "\033[94m",  # blue
}
_RESET = "\033[0m"

_shutdown = False


def _prefix(name: str) -> str:
    color = _COLORS.get(name, "")
    return f"{color}[{name}]{_RESET}"


def _current_hour_dir(base: str = "data") -> str:
    """Return the hourly partition directory: data/YYYY-MM-DD/HH"""
    now = datetime.now(timezone.utc)
    return os.path.join(base, now.strftime("%Y-%m-%d"), now.strftime("%H"))


def _current_hour_tag() -> str:
    """Return current UTC hour as 'YYYY-MM-DD/HH' for comparison."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d/%H")


def _hourly_paths(base: str = "data", assets: Optional[List[str]] = None) -> dict:
    """Return all output paths for the current hour partition."""
    hour_dir = _current_hour_dir(base)
    os.makedirs(hour_dir, exist_ok=True)
    assets = assets or ["BTC"]
    paths: dict = {
        "dir": hour_dir,
        "map": os.path.join(hour_dir, "slot_key_map.csv"),
    }
    for asset in assets:
        a = asset.lower()
        paths[f"price_{asset}"] = os.path.join(hour_dir, f"{a}_1s.csv")
        paths[f"snapshots_{asset}"] = os.path.join(hour_dir, f"snapshots_{a}.jsonl")
        paths[f"history_{asset}"] = os.path.join(hour_dir, f"history_{a}.csv")
    return paths


# ---------------------------------------------------------------------------
# Slot-key mapping: joins collect_snapshots <-> collect_history
# ---------------------------------------------------------------------------

SLOT_KEY_COLUMNS = [
    "slot_ts",
    "slot_utc",
    "outcome",
    "strike_price",
    "volume",
    "snapshot_count",
    "first_snapshot_ts",
    "last_snapshot_ts",
    "avg_yes_mid",
    "avg_no_mid",
    "avg_realized_vol_30s",
]


def build_slot_key_map(
    snapshots_path: str,
    history_path: str,
    output_path: str,
) -> int:
    """Build a CSV mapping slot_ts -> snapshot aggregates + history outcome.

    Returns the number of mapped slots.
    """
    # --- Aggregate snapshots by slot_ts ---
    slot_snaps: Dict[int, List[dict]] = {}
    if os.path.exists(snapshots_path):
        with open(snapshots_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "outcome":
                    continue
                slot_ts = rec.get("slot_ts")
                if slot_ts is not None:
                    slot_snaps.setdefault(int(slot_ts), []).append(rec)

    # --- Load history outcomes keyed by slot_ts ---
    slot_history: Dict[int, dict] = {}
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    slot_ts = int(row["slot_ts"])
                    slot_history[slot_ts] = row
                except (KeyError, ValueError):
                    continue

    # --- Merge on slot_ts ---
    all_slots = sorted(set(slot_snaps.keys()) | set(slot_history.keys()))
    if not all_slots:
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SLOT_KEY_COLUMNS)
        writer.writeheader()

        for slot_ts in all_slots:
            snaps = slot_snaps.get(slot_ts, [])
            hist = slot_history.get(slot_ts, {})

            snap_count = len(snaps)
            snap_tss = [s.get("snapshot_ts", 0) for s in snaps if s.get("snapshot_ts")]
            yes_mids = [s["yes_mid"] for s in snaps if s.get("yes_mid") is not None]
            no_mids = [s["no_mid"] for s in snaps if s.get("no_mid") is not None]
            vols = [s["realized_vol_30s"] for s in snaps if s.get("realized_vol_30s") is not None]

            slot_utc = (
                hist.get("slot_utc")
                or datetime.fromtimestamp(slot_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )

            writer.writerow({
                "slot_ts": slot_ts,
                "slot_utc": slot_utc,
                "outcome": hist.get("outcome", ""),
                "strike_price": hist.get("strike_price", ""),
                "volume": hist.get("volume", ""),
                "snapshot_count": snap_count,
                "first_snapshot_ts": f"{min(snap_tss):.3f}" if snap_tss else "",
                "last_snapshot_ts": f"{max(snap_tss):.3f}" if snap_tss else "",
                "avg_yes_mid": f"{sum(yes_mids) / len(yes_mids):.4f}" if yes_mids else "",
                "avg_no_mid": f"{sum(no_mids) / len(no_mids):.4f}" if no_mids else "",
                "avg_realized_vol_30s": f"{sum(vols) / len(vols):.6f}" if vols else "",
            })

    return len(all_slots)


# ---------------------------------------------------------------------------
# S3 sync
# ---------------------------------------------------------------------------

_S3_BUCKET = "k-polymarket-data"


def _s3_sync_available() -> bool:
    """Check if AWS CLI is installed and configured."""
    return shutil.which("aws") is not None


def _sync_to_s3(local_dir: str, s3_bucket: str = _S3_BUCKET) -> bool:
    """Sync a local hour directory to S3. Returns True on success."""
    # local_dir is like "data/2026-04-15/03"
    # Upload to s3://bucket/data/2026-04-15/03/
    s3_path = f"s3://{s3_bucket}/{local_dir}/"
    cmd = ["aws", "s3", "sync", local_dir, s3_path, "--region", "us-west-2"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"{_prefix('s3')} synced {local_dir} -> {s3_path}")
            return True
        else:
            print(f"{_prefix('s3')} FAILED: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"{_prefix('s3')} timeout syncing {local_dir}")
        return False
    except FileNotFoundError:
        print(f"{_prefix('s3')} aws CLI not found — skipping sync")
        return False


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _launch_daemons(
    paths: dict,
    assets: List[str],
    exchange: str,
    snap_interval: int,
) -> Dict[str, subprocess.Popen]:
    """Launch per-asset price recorder and snapshot collector daemons."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    procs: Dict[str, subprocess.Popen] = {}

    # Single multi-asset price recorder
    price_cmd = [
        sys.executable, "scripts/record_crypto_1s.py",
        "--assets", *assets,
        "--exchange", exchange,
        "--output-dir", paths["dir"],
        "--csv-only",
    ]
    procs["price"] = subprocess.Popen(price_cmd, cwd=_PROJECT_ROOT, env=env)
    print(f"{_prefix('price')} PID {procs['price'].pid}  ->  {paths['dir']}/  ({', '.join(assets)})")

    # Per-asset snapshot collectors
    for asset in assets:
        snap_key = f"snapshots_{asset}"
        snap_cmd = [
            sys.executable, "scripts/collect_snapshots.py",
            "--asset", asset,
            "--interval", str(snap_interval),
            "--output", paths[snap_key],
        ]
        procs[snap_key] = subprocess.Popen(snap_cmd, cwd=_PROJECT_ROOT, env=env)
        print(f"{_prefix('snapshots')} [{asset}] PID {procs[snap_key].pid}  ->  {paths[snap_key]}")

    return procs


def _run_history_batch(history_output: str, hours: int, asset: str = "BTC") -> Optional[subprocess.Popen]:
    """Launch a one-shot history collection and return the Popen."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [
        sys.executable, "scripts/collect_history.py",
        "--asset", asset,
        "--hours", str(hours),
        "--output", history_output,
    ]
    proc = subprocess.Popen(cmd, cwd=_PROJECT_ROOT, env=env)
    print(f"{_prefix('history')} [{asset}] PID {proc.pid}  batch ({hours}h) ->  {history_output}")
    return proc


def _stop_procs(procs: Dict[str, subprocess.Popen], timeout: float = 8.0) -> None:
    """SIGTERM all running procs, then SIGKILL stragglers."""
    for name, proc in procs.items():
        if proc.poll() is not None:
            continue
        print(f"{_prefix(name)} sending SIGTERM...")
        proc.terminate()

    deadline = time.time() + timeout
    for name, proc in procs.items():
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"{_prefix(name)} SIGKILL (did not exit in time)")
            proc.kill()
        print(f"{_prefix(name)} stopped  (exit code {proc.returncode})")


def main() -> None:
    global _shutdown

    parser = argparse.ArgumentParser(
        description="Unified Polymarket data collector — hourly partitioned",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--assets", nargs="+", default=["BTC"],
                        help="Asset tickers to collect (default: BTC). Use BTC ETH SOL DOGE XRP for all.")
    parser.add_argument("--exchange", default="binance", choices=["coinbase", "binance", "binance_us"])
    parser.add_argument("--snap-interval", type=int, default=5, help="Snapshot poll interval (seconds)")
    parser.add_argument("--history-hours", type=int, default=2, help="Hours of history per batch")
    parser.add_argument("--no-history", action="store_true", help="Skip periodic history collection")
    parser.add_argument("--map-interval", type=int, default=600, help="Seconds between slot-key map rebuilds")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--s3-bucket", default=_S3_BUCKET, help="S3 bucket for syncing data")
    parser.add_argument("--no-s3", action="store_true", help="Disable S3 sync")
    args = parser.parse_args()

    s3_enabled = not args.no_s3 and _s3_sync_available()
    if not args.no_s3 and not s3_enabled:
        print("  [warn] AWS CLI not found — S3 sync disabled")
        print("         Install awscli and run 'aws configure' to enable.\n")

    def _handle_signal(sig, frame):
        global _shutdown
        _shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    assets = [a.upper() for a in args.assets]

    # --- Initial hour ---
    paths = _hourly_paths(args.data_dir, assets)
    current_hour = _current_hour_tag()

    print("=" * 60)
    print("  Polymarket Data Collector  (hourly partitioned, multi-asset)")
    print("=" * 60)
    print(f"  Assets:  {', '.join(assets)}")
    print(f"  Layout:  {args.data_dir}/YYYY-MM-DD/HH/")
    print(f"           {{asset}}_1s.csv | snapshots_{{asset}}.jsonl | history_{{asset}}.csv")
    print(f"  Current: {paths['dir']}/")
    print(f"  S3 sync: {'s3://' + args.s3_bucket + '/' if s3_enabled else 'disabled'}")
    print(f"\n  Press Ctrl+C to stop all.\n")

    # Launch daemons for current hour
    procs = _launch_daemons(
        paths=paths,
        assets=assets,
        exchange=args.exchange,
        snap_interval=args.snap_interval,
    )

    # Initial history batch (one per asset)
    history_procs: Dict[str, subprocess.Popen] = {}
    if not args.no_history:
        for asset in assets:
            history_procs[asset] = _run_history_batch(
                paths[f"history_{asset}"], args.history_hours, asset=asset,
            )

    last_map_build = 0.0

    print()
    try:
        while not _shutdown:
            now = time.time()
            new_hour = _current_hour_tag()

            # ---- Hourly rotation ----
            if new_hour != current_hour:
                print(f"\n{_prefix('rotate')} hour changed: {current_hour} -> {new_hour}")

                # Build final map for the ending hour (use first asset's snapshots/history)
                first = assets[0]
                try:
                    count = build_slot_key_map(
                        paths[f"snapshots_{first}"], paths[f"history_{first}"], paths["map"]
                    )
                    print(f"{_prefix('mapper')} final map for {current_hour}  ({count} slots)")
                except Exception as exc:
                    print(f"{_prefix('mapper')} error: {exc}")

                # Stop daemons
                _stop_procs(procs, timeout=5.0)

                # Sync completed hour to S3
                if s3_enabled:
                    _sync_to_s3(paths["dir"], args.s3_bucket)

                # New hour paths
                current_hour = new_hour
                paths = _hourly_paths(args.data_dir, assets)
                print(f"{_prefix('rotate')} new dir: {paths['dir']}/")

                # Relaunch daemons with new output paths
                procs = _launch_daemons(
                    paths=paths,
                    assets=assets,
                    exchange=args.exchange,
                    snap_interval=args.snap_interval,
                )

                # New history batch per asset
                if not args.no_history:
                    for asset_name, hproc in history_procs.items():
                        if hproc.poll() is None:
                            hproc.terminate()
                            hproc.wait(timeout=5)
                    history_procs.clear()
                    for asset_name in assets:
                        history_procs[asset_name] = _run_history_batch(
                            paths[f"history_{asset_name}"], args.history_hours, asset=asset_name,
                        )

                last_map_build = now
                print()

            # ---- Check daemon health ----
            for name, proc in list(procs.items()):
                ret = proc.poll()
                if ret is not None and not _shutdown:
                    print(f"\n{_prefix(name)} exited unexpectedly (code {ret}), restarting...")
                    del procs[name]
                    # Restart the dead daemon individually
                    if name == "price":
                        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                        price_cmd = [
                            sys.executable, "scripts/record_crypto_1s.py",
                            "--assets", *assets,
                            "--exchange", args.exchange,
                            "--output-dir", paths["dir"],
                            "--csv-only",
                        ]
                        procs["price"] = subprocess.Popen(price_cmd, cwd=_PROJECT_ROOT, env=env)
                        print(f"{_prefix('price')} restarted PID {procs['price'].pid}")
                    elif name.startswith("snapshots_"):
                        asset_name = name.split("_", 1)[1]
                        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                        snap_cmd = [
                            sys.executable, "scripts/collect_snapshots.py",
                            "--asset", asset_name,
                            "--interval", str(args.snap_interval),
                            "--output", paths[name],
                        ]
                        procs[name] = subprocess.Popen(snap_cmd, cwd=_PROJECT_ROOT, env=env)
                        print(f"{_prefix('snapshots')} [{asset_name}] restarted PID {procs[name].pid}")

            # ---- Periodic slot-key map rebuild ----
            if now - last_map_build >= args.map_interval:
                first = assets[0]
                try:
                    count = build_slot_key_map(
                        paths[f"snapshots_{first}"], paths[f"history_{first}"], paths["map"]
                    )
                    print(
                        f"{_prefix('mapper')} rebuilt slot_key_map.csv  "
                        f"({count} slots)"
                    )
                except Exception as exc:
                    print(f"{_prefix('mapper')} error: {exc}")
                last_map_build = now

            time.sleep(1.0)

    except KeyboardInterrupt:
        pass

    print(f"\n{'=' * 60}")
    print("Shutting down all processes...")
    print(f"{'=' * 60}")

    for asset_name, hproc in history_procs.items():
        if hproc.poll() is None:
            procs[f"history_{asset_name}"] = hproc
    _stop_procs(procs)

    # Final map build
    first = assets[0]
    try:
        count = build_slot_key_map(
            paths[f"snapshots_{first}"], paths[f"history_{first}"], paths["map"]
        )
        print(f"\n{_prefix('mapper')} final slot_key_map.csv  ({count} slots)")
    except Exception as exc:
        print(f"{_prefix('mapper')} final build failed: {exc}")

    # Final S3 sync for current (partial) hour
    if s3_enabled:
        _sync_to_s3(paths["dir"], args.s3_bucket)

    print("\nDone.")


if __name__ == "__main__":
    main()
