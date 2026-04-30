"""Download historical BTC/USDT klines from data.binance.vision.

Downloads daily ZIP archives of 1s or 1m klines and concatenates into a single CSV.

Usage:
    python scripts/fetch_binance_btc.py --start 2025-04-01 --end 2025-04-03
    python scripts/fetch_binance_btc.py --start 2025-04-01 --end 2025-04-03 --interval 1m
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT"
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
OUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Binance switched from ms to us timestamps on 2025-01-01
US_CUTOFF = date(2025, 1, 1)


def _download_day(day: date, interval: str) -> pd.DataFrame | None:
    """Download and extract one daily ZIP. Returns DataFrame or None."""
    filename = f"BTCUSDT-{interval}-{day.isoformat()}.zip"
    url = f"{BASE_URL}/{interval}/{filename}"

    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)

    return df


def fetch_klines(start: date, end: date, interval: str = "1s") -> pd.DataFrame:
    """Download klines for [start, end) date range."""
    frames: list[pd.DataFrame] = []
    day = start
    while day < end:
        print(f"  {day} ...", end=" ", flush=True)
        df = _download_day(day, interval)
        if df is not None:
            frames.append(df)
            print(f"{len(df)} rows")
        else:
            print("not found")
        day += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=OUT_COLUMNS)

    df = pd.concat(frames, ignore_index=True)

    # Convert open_time to epoch seconds
    # Post-2025-01-01: microseconds; before: milliseconds
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    if start >= US_CUTOFF:
        ts = ts / 1_000_000.0
    else:
        ts = ts / 1_000.0

    out = pd.DataFrame({
        "timestamp": ts,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
    })
    return out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download BTC klines from data.binance.vision")
    p.add_argument("--start", required=True, help="Start date inclusive (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date exclusive (YYYY-MM-DD)")
    p.add_argument("--interval", default="1s", help="Kline interval: 1s, 1m, etc.")
    p.add_argument("--output", default=None, help="Output CSV path (default: data/btc_{interval}.csv)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output = args.output or f"data/btc_{args.interval}.csv"

    print(f"Downloading BTCUSDT {args.interval} klines: {start} → {end}")
    df = fetch_klines(start, end, interval=args.interval)
    print(f"Total: {len(df)} rows")

    if not df.empty:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Saved to {output}")
        print(df.head())
        print(f"Time range: {df['timestamp'].min():.0f} → {df['timestamp'].max():.0f}")


if __name__ == "__main__":
    main()
