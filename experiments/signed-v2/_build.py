"""Rebuild the snapshot dataset under the signed-v2 schema.

Same loader as signed-v1, but extended date range (13 days vs 7) and
the new feature builder that emits 171 columns including:
  • hour_sin / hour_cos / is_weekend
  • recent_up_rate_5 / 10 / 20
  • ut_trend_disagreement

Loads 2026-04-16..2026-04-28 from S3 (k-polymarket-data) via
``src.backtest.s3_snapshot_loader.load_dates`` and writes a parquet to
``experiments/signed-v2/dataset.parquet``.
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
    log = logging.getLogger("signed-v2-build")

    start = date(2026, 4, 16)
    end = date(2026, 4, 28)
    dates = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    log.info("Loading %d dates: %s..%s", len(dates), dates[0], dates[-1])
    df = load_dates(dates=dates, bucket="k-polymarket-data")
    log.info(
        "Loaded rows=%d slots=%d cols=%d",
        len(df), df["slot_ts"].nunique(), len(df.columns),
    )

    # Sanity check: confirm the v2 columns are present and not all zero.
    for col in (
        "hour_sin", "hour_cos", "is_weekend",
        "recent_up_rate_5", "recent_up_rate_10", "recent_up_rate_20",
        "ut_trend_disagreement",
    ):
        if col not in df.columns:
            log.warning("v2 column missing from dataset: %s", col)
            continue
        nz = (df[col] != 0.0).sum()
        log.info("  %s: nonzero=%d/%d (%.1f%%)",
                 col, nz, len(df), nz / max(len(df), 1) * 100)

    out_path = _REPO / "experiments" / "signed-v2" / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    import pandas as pd
    tag = pd.to_datetime(df["slot_ts"], unit="s", utc=True).dt.date.astype(str)
    for d_str, count in tag.value_counts().sort_index().items():
        slots = df.loc[tag == d_str, "slot_ts"].nunique()
        log.info("  %s: %d rows / %d slots", d_str, count, slots)


if __name__ == "__main__":
    main()
