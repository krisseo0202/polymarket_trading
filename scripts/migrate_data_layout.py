#!/usr/bin/env python3
"""
Migrate flat data/ files into date-partitioned directories.

Before:
  data/btc_live_1s_20260408T062427Z.csv
  data/live_orderbook_snapshots_20260408T062427Z.csv
  data/decision_log_20260408T062428Z.jsonl

After:
  data/2026-04-08/btc_live_1s_20260408T062427Z.csv
  data/2026-04-08/live_orderbook_snapshots_20260408T062427Z.csv
  data/2026-04-08/decision_log_20260408T062428Z.jsonl

Static/analysis files (PNGs, HTMLs, caches) go to data/analysis/.
Legacy files without a timestamp suffix use their mtime for the date key.

Safe to run multiple times (idempotent — skips files already in date dirs).

WARNING: Do NOT migrate files that are being actively written by a running
data collector (record_btc_1s.py, collect_live_window.py). Stop the bot
first, run the migration, then restart. Moving an open file on WSL/NTFS
corrupts the filesystem metadata.

Usage:
    python scripts/migrate_data_layout.py          # dry run (default)
    python scripts/migrate_data_layout.py --apply   # actually move files
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime, timezone

DATA_DIR = "data"

# Patterns that carry live session data — these get date-partitioned.
SESSION_PATTERNS = [
    re.compile(r"^(btc_live_1s)(?:_(\d{8}T\d{6}Z))?(\.csv)$"),
    re.compile(r"^(live_orderbook_snapshots)(?:_(\d{8}T\d{6}Z))?(\.csv)$"),
    re.compile(r"^(decision_log)(?:_(\d{8}T\d{6}Z))?(\.jsonl)$"),
]

# Files that are static analysis output — not date-session-specific.
ANALYSIS_EXTENSIONS = {".png", ".html"}
ANALYSIS_NAMES = {
    "backtest_td_rsi_results.csv",
    "btc_1m_cache.csv",
    "btc_1s.csv",
}


def extract_date(filename: str, filepath: str) -> str | None:
    """Return 'YYYY-MM-DD' date key for a session file, or None for non-session files."""
    for pat in SESSION_PATTERNS:
        m = pat.match(filename)
        if m:
            ts_str = m.group(2)
            if ts_str:
                # Embedded timestamp: 20260408T062427Z → 2026-04-08
                return f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]}"
            else:
                # Legacy file (no timestamp in name) — use mtime
                mtime = os.path.getmtime(filepath)
                return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    return None


def is_analysis_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in ANALYSIS_EXTENSIONS or filename in ANALYSIS_NAMES


def plan_moves(data_dir: str) -> list[tuple[str, str]]:
    """Return a list of (src, dst) paths for the migration."""
    moves: list[tuple[str, str]] = []

    for filename in sorted(os.listdir(data_dir)):
        src = os.path.join(data_dir, filename)
        if not os.path.isfile(src):
            continue  # skip subdirectories

        # Analysis file → data/analysis/
        if is_analysis_file(filename):
            dst = os.path.join(data_dir, "analysis", filename)
            if src != dst:
                moves.append((src, dst))
            continue

        # Session file → data/YYYY-MM-DD/
        date_key = extract_date(filename, src)
        if date_key:
            dst = os.path.join(data_dir, date_key, filename)
            if src != dst:
                moves.append((src, dst))
            continue

        # Unrecognized — leave in place
        # (e.g., retrain_history.jsonl, probability_ticks.jsonl)

    return moves


def main():
    parser = argparse.ArgumentParser(description="Migrate data/ to date-partitioned layout")
    parser.add_argument("--apply", action="store_true", help="Actually move files (default: dry run)")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    moves = plan_moves(args.data_dir)
    if not moves:
        print("Nothing to migrate.")
        return

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] {len(moves)} file(s) to move:\n")

    for src, dst in moves:
        print(f"  {src}")
        print(f"    → {dst}")

        if args.apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    if not args.apply:
        print("\nRe-run with --apply to execute the moves.")
    else:
        print(f"\nDone. {len(moves)} file(s) moved.")


if __name__ == "__main__":
    main()
