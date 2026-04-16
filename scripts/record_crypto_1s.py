#!/usr/bin/env python3
"""
Record live 1-second OHLCV bars for multiple cryptocurrencies simultaneously.

Supports any Coinbase or Binance.US product. Default: ETH-USD and SOL-USD.
Stores bars to PostgreSQL (primary) with optional CSV fallback.

Usage:
    # Postgres (default) — reads DB_* env vars
    python scripts/record_crypto_1s.py
    python scripts/record_crypto_1s.py --assets ETH SOL BTC

    # CSV-only (no Postgres dependency)
    python scripts/record_crypto_1s.py --csv-only
    python scripts/record_crypto_1s.py --csv-only --output-dir data/custom

    # Custom DB connection
    DB_HOST=myhost DB_NAME=polymarket python scripts/record_crypto_1s.py

Environment variables for Postgres:
    DB_HOST     (default: localhost)
    DB_PORT     (default: 5432)
    DB_NAME     (default: polymarket)
    DB_USER     (default: poly)
    DB_PASSWORD (default: poly_dev_password)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.crypto_feed import CryptoPriceFeed

_shutdown = False
log = logging.getLogger("record_crypto")


def _handle_signal(sig, frame):
    global _shutdown
    log.info("Shutdown signal received")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Reuse symbol maps from CryptoPriceFeed
SYMBOLS = {
    "coinbase": CryptoPriceFeed.COINBASE_SYMBOLS,
    "binance": CryptoPriceFeed.BINANCE_SYMBOLS,
    "binance_us": CryptoPriceFeed.BINANCE_US_SYMBOLS,
}

PRICE_DECIMALS = {
    "BTC": 2,
    "ETH": 2,
    "SOL": 4,
    "DOGE": 6,
    "XRP": 4,
}


# ---------------------------------------------------------------------------
# Postgres sink
# ---------------------------------------------------------------------------

def get_db_config() -> dict:
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", "5432")),
        "dbname": os.environ.get("DB_NAME", "polymarket"),
        "user": os.environ.get("DB_USER", "poly"),
        "password": os.environ.get("DB_PASSWORD", "poly_dev_password"),
    }


def get_conn():
    import psycopg2
    return psycopg2.connect(**get_db_config())


def ensure_tables():
    """Create the crypto_1s_bars table and hypertable (if TimescaleDB is available)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crypto_1s_bars (
                    ts   TIMESTAMPTZ NOT NULL,
                    asset TEXT        NOT NULL,
                    open  DOUBLE PRECISION,
                    high  DOUBLE PRECISION,
                    low   DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume INTEGER,
                    PRIMARY KEY (asset, ts)
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_1s_bars_ts
                ON crypto_1s_bars (ts DESC);
            """)
        conn.commit()

    # Attempt TimescaleDB hypertable in a separate transaction so a failure
    # doesn't roll back the CREATE TABLE above.
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT create_hypertable(
                        'crypto_1s_bars', 'ts',
                        if_not_exists => TRUE,
                        migrate_data  => TRUE
                    );
                """)
            conn.commit()
            log.info("TimescaleDB hypertable enabled for crypto_1s_bars")
    except Exception:
        log.debug("TimescaleDB not available — using plain table")

    log.info("Postgres table crypto_1s_bars ready")


class PgSink:
    """Batched Postgres writer for 1s bars with persistent connection."""

    BATCH_SIZE = 60   # flush every ~60 bars (1 minute of data)
    MAX_BATCH = 600   # drop oldest if DB is down for 10+ minutes

    def __init__(self):
        self._batch: List[Tuple] = []
        self._total_written = 0
        self._conn = None

    def _get_conn(self):
        """Return persistent connection, reconnecting if needed."""
        import psycopg2
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**get_db_config())
        return self._conn

    def append(self, asset: str, ts_epoch: int, o: float, h: float, l: float, c: float, vol: int):
        ts = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
        self._batch.append((ts, asset, o, h, l, c, vol))
        if len(self._batch) > self.MAX_BATCH:
            dropped = len(self._batch) - self.MAX_BATCH
            self._batch = self._batch[dropped:]
            log.warning(f"Postgres batch overflow — dropped {dropped} oldest bars")
        if len(self._batch) >= self.BATCH_SIZE:
            self.flush()

    def flush(self):
        if not self._batch:
            return
        try:
            from psycopg2.extras import execute_values
            conn = self._get_conn()
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO crypto_1s_bars (ts, asset, open, high, low, close, volume)
                    VALUES %s
                    ON CONFLICT (asset, ts) DO UPDATE SET
                        open   = EXCLUDED.open,
                        high   = EXCLUDED.high,
                        low    = EXCLUDED.low,
                        close  = EXCLUDED.close,
                        volume = EXCLUDED.volume;
                    """,
                    self._batch,
                )
            conn.commit()
            self._total_written += len(self._batch)
            self._batch.clear()
        except Exception as e:
            log.error(f"Postgres flush failed: {e}")
            # Close broken connection so next flush reconnects
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def close(self):
        self.flush()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    @property
    def total_written(self) -> int:
        return self._total_written


# ---------------------------------------------------------------------------
# CSV sink (fallback / offline use)
# ---------------------------------------------------------------------------

class CsvSink:
    """Per-asset CSV writer that rotates files every hour.

    File layout:  <output_dir>/<date>/<HH>/<asset>_live_1s.csv
    Example:      data/2026-04-14/04/btc_live_1s.csv
                  data/2026-04-14/05/btc_live_1s.csv
    """

    def __init__(self, asset: str, output_dir: str):
        self.asset = asset
        self._output_dir = output_dir
        self.total_written = 0
        self.output_path = ""  # updated on each rotation

        self._file = None
        self._writer = None
        self._current_hour: Optional[str] = None  # "YYYY-MM-DD/HH"

    def _rotate_if_needed(self, ts_epoch: int):
        """Open a new file if the hour (UTC) has changed."""
        dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
        hour_key = dt.strftime("%Y-%m-%d/%H")

        if hour_key == self._current_hour:
            return

        # Close previous file
        if self._file is not None:
            self._file.close()
            log.info(f"  [{self.asset}] rotated CSV — closed {self.output_path}")

        # Open new file
        self._current_hour = hour_key
        self.output_path = os.path.join(
            self._output_dir, hour_key, f"{self.asset.lower()}_live_1s.csv"
        )
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        file_exists = os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0
        self._file = open(self.output_path, "a", newline="")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(CSV_COLUMNS)
            self._file.flush()
        log.info(f"  [{self.asset}] writing to {self.output_path}")

    _FLUSH_INTERVAL = 60  # flush to disk every 60 rows

    def append(self, ts_epoch: int, o: float, h: float, l: float, c: float, vol: int):
        self._rotate_if_needed(ts_epoch)
        self._writer.writerow([ts_epoch, o, h, l, c, vol])
        self.total_written += 1
        if self.total_written % self._FLUSH_INTERVAL == 0:
            self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()


# ---------------------------------------------------------------------------
# Bar accumulator (shared by both sinks)
# ---------------------------------------------------------------------------

class BarAccumulator:
    """Accumulates ticks into 1-second OHLCV bars for one asset."""

    def __init__(self, asset: str):
        self.asset = asset
        self.decimals = PRICE_DECIMALS.get(asset, 4)
        self.bar_open: Optional[float] = None
        self.bar_high: Optional[float] = None
        self.bar_low: Optional[float] = None
        self.bar_close: Optional[float] = None
        self.tick_count: int = 0
        self.current_second: int = int(time.time())

    def tick(self, mid: float, now_second: int) -> Optional[Tuple[int, float, float, float, float, int]]:
        """Process a tick. Returns (ts, o, h, l, c, vol) if a bar completed, else None."""
        completed = None

        if now_second != self.current_second:
            if self.bar_open is not None:
                completed = (
                    self.current_second,
                    round(self.bar_open, self.decimals),
                    round(self.bar_high, self.decimals),
                    round(self.bar_low, self.decimals),
                    round(self.bar_close, self.decimals),
                    self.tick_count,
                )
            self.bar_open = mid
            self.bar_high = mid
            self.bar_low = mid
            self.bar_close = mid
            self.tick_count = 1
            self.current_second = now_second
        else:
            if self.bar_open is None:
                self.bar_open = mid
                self.bar_high = mid
                self.bar_low = mid
            self.bar_high = max(self.bar_high, mid)
            self.bar_low = min(self.bar_low, mid)
            self.bar_close = mid
            self.tick_count += 1

        return completed

    def flush_final(self) -> Optional[Tuple[int, float, float, float, float, int]]:
        if self.bar_open is not None:
            return (
                self.current_second,
                round(self.bar_open, self.decimals),
                round(self.bar_high, self.decimals),
                round(self.bar_low, self.decimals),
                round(self.bar_close, self.decimals),
                self.tick_count,
            )
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(assets: list[str], exchange: str, output_dir: str, csv_only: bool):
    use_pg = not csv_only
    pg_sink: Optional[PgSink] = None
    csv_sinks: Dict[str, CsvSink] = {}

    if use_pg:
        try:
            ensure_tables()
            pg_sink = PgSink()
            log.info("Postgres sink active")
        except Exception as e:
            raise RuntimeError("Postgres connection failed") from e

    feeds: Dict[str, CryptoPriceFeed] = {}
    bars: Dict[str, BarAccumulator] = {}

    for asset in assets:
        symbol_map = SYMBOLS.get(exchange, {})
        symbol = symbol_map.get(asset)
        if symbol is None:
            log.error(f"Unknown asset '{asset}' for exchange '{exchange}'. "
                      f"Known: {list(symbol_map.keys())}")
            return

        feed = CryptoPriceFeed(symbol=symbol, exchange=exchange).start()
        feeds[asset] = feed
        bars[asset] = BarAccumulator(asset)

        # CSV sinks auto-rotate on hour boundaries: data/<date>/<HH>/<asset>_live_1s.csv
        csv_sinks[asset] = CsvSink(asset, output_dir)

    # Wait for feeds
    log.info(f"Connecting to {exchange} for {', '.join(assets)}...")
    deadline = time.time() + 15
    while time.time() < deadline:
        if all(f.get_latest_mid() is not None for f in feeds.values()):
            break
        time.sleep(0.1)

    connected = [a for a, f in feeds.items() if f.get_latest_mid() is not None]
    failed = [a for a, f in feeds.items() if f.get_latest_mid() is None]

    if failed:
        log.warning(f"No data received for {', '.join(failed)} after 15s")
        if not connected:
            log.error("No feeds connected. Check network.")
            for f in feeds.values():
                f.stop()
            return

    for asset in connected:
        mid = feeds[asset].get_latest_mid()
        dec = bars[asset].decimals
        csv_pattern = f"{output_dir}/<date>/<HH>/{asset.lower()}_live_1s.csv"
        log.info(f"  {asset}: price={mid:.{dec}f}  csv={csv_pattern}"
                 + ("  pg=crypto_1s_bars" if use_pg else ""))

    sink_label = "Postgres + CSV" if use_pg else "CSV-only"
    log.info(f"Recording 1s bars [{sink_label}] for {', '.join(connected)}")

    try:
        while not _shutdown:
            now = time.time()
            now_second = int(now)

            for asset in connected:
                mid = feeds[asset].get_latest_mid()
                if mid is None:
                    continue

                completed = bars[asset].tick(mid, now_second)
                if completed:
                    ts, o, h, l, c, vol = completed
                    dec = bars[asset].decimals

                    csv_sinks[asset].append(ts, o, h, l, c, vol)
                    if pg_sink:
                        pg_sink.append(asset, ts, o, h, l, c, vol)

                    log.info(
                        f"  [{asset}] {ts}  O={o:.{dec}f}  H={h:.{dec}f}  "
                        f"L={l:.{dec}f}  C={c:.{dec}f}  ticks={vol}"
                    )

            # Sleep until ~50ms before next second boundary to catch the last tick
            sleep_s = max(0.01, math.ceil(now) - now - 0.05)
            time.sleep(sleep_s)
    finally:
        # Flush remaining bars
        for asset in connected:
            final = bars[asset].flush_final()
            if final:
                ts, o, h, l, c, vol = final
                csv_sinks[asset].append(ts, o, h, l, c, vol)
                if pg_sink:
                    pg_sink.append(asset, ts, o, h, l, c, vol)

            csv_sinks[asset].close()
            feeds[asset].stop()
            log.info(f"  [{asset}] {csv_sinks[asset].total_written} bars -> {csv_sinks[asset].output_path}")

        if pg_sink:
            pg_sink.close()
            log.info(f"  [Postgres] {pg_sink.total_written} bars total -> crypto_1s_bars")

        log.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Record live 1s OHLCV bars for multiple cryptocurrencies"
    )
    parser.add_argument(
        "--assets", nargs="+", default=["BTC"],
        help="Asset tickers to record (default: BTC). Use BTC ETH SOL for all three."
    )
    parser.add_argument(
        "--exchange", default="binance", choices=["coinbase", "binance", "binance_us"],
        help="Exchange backend (default: coinbase)"
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Base output directory for CSV files (default: data/)"
    )
    parser.add_argument(
        "--csv-only", action="store_true",
        help="Skip Postgres, write CSV only"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
        help="Log level (default: INFO)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    assets = [a.upper() for a in args.assets]
    run(assets, args.exchange, args.output_dir, args.csv_only)


if __name__ == "__main__":
    main()
