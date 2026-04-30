"""One-shot rebuild of the snapshot dataset under the new signed-feature schema.

Loads 2026-04-16..2026-04-22 from S3 (bucket k-polymarket-data) via
``src.backtest.s3_snapshot_loader.load_dates``, which internally calls the
(now signed) ``build_live_features``. Writes a parquet to
``experiments/signed-v1/dataset.parquet``.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from src.backtest.s3_snapshot_loader import load_dates


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("signed-v1-build")

    start = date(2026, 4, 16)
    end = date(2026, 4, 22)
    dates = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    log.info("Loading %d dates: %s..%s", len(dates), dates[0], dates[-1])
    df = load_dates(dates=dates, bucket="k-polymarket-data")
    log.info("Loaded rows=%d slots=%d cols=%d",
             len(df), df["slot_ts"].nunique(), len(df.columns))

    out_path = _REPO / "experiments" / "signed-v1" / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Per-date coverage for the training script to use.
    import pandas as pd
    tag = pd.to_datetime(df["slot_ts"], unit="s", utc=True).dt.date.astype(str)
    for d_str, count in tag.value_counts().sort_index().items():
        log.info("  %s: %d rows / %d slots",
                 d_str, count,
                 df.loc[tag == d_str, "slot_ts"].nunique())


if __name__ == "__main__":
    main()
