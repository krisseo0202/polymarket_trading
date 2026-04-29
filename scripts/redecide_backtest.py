"""Stage 2 backtest determinism check.

For each TRADED record in a decision-log JSONL, take the feature vector
the live bot computed (the ``f_*`` fields) and feed it back through the
loaded model. The recomputed ``prob_yes`` must match what was recorded.

If the match rate falls below the threshold, the model is not
deterministic on its own feature inputs — a problem worth fixing
before any offline training experiment, because every retrain comparison
relies on this property.

Usage::

    python scripts/redecide_backtest.py \\
        --decision-log data/2026-04-26/decision_log_*.jsonl \\
        --model-dir models/signed_v1_trim \\
        --since "2026-04-26 00:00:00"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.models.xgb_fb_model import XGBFBModel


_DEFAULT_TOLERANCE = 0.005
_DEFAULT_MATCH_RATE = 0.99


def _load_traded_records(path: Path, since_ts: float) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ts", 0) < since_ts:
                continue
            if not r.get("action"):
                continue
            if r.get("prob_yes") is None:
                continue
            out.append(r)
    return out


def _feature_row(record: dict, feature_names: List[str]) -> np.ndarray:
    """Build the feature vector the model expects from the record's ``f_*`` keys."""
    return np.asarray(
        [[float(record.get(f"f_{name}", 0.0) or 0.0) for name in feature_names]],
        dtype=float,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--since", required=True,
                        help="UTC start of replay window, 'YYYY-MM-DD HH:MM:SS'.")
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE,
                        help="Per-record max |delta prob_yes| to count as a match.")
    parser.add_argument("--min-match-rate", type=float, default=_DEFAULT_MATCH_RATE,
                        help="Pass when match_rate ≥ this. Default 0.99.")
    parser.add_argument("--show-mismatches", action="store_true",
                        help="Print up to 30 records where re-decide differs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    since_ts = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S").timestamp()
    records = _load_traded_records(args.decision_log, since_ts=since_ts)
    if not records:
        print("No TRADED records found in the window. Nothing to check.")
        return 0

    model = XGBFBModel.load(str(args.model_dir), logger=logger)
    if not model.ready:
        print(f"Failed to load model from {args.model_dir}.")
        return 2

    feature_names = model.feature_names
    print(f"=== Stage 2: re-decide determinism check ===")
    print(f"records: {len(records)}    model: {model.model_version}    "
          f"features: {len(feature_names)}")
    print()

    matches = 0
    mismatches: List[dict] = []
    for r in records:
        x = _feature_row(r, feature_names)
        try:
            raw = float(model._model.predict_proba(x)[0, 1])
        except Exception as exc:
            logger.warning("predict_proba failed for record at ts=%s: %s", r.get("ts"), exc)
            continue
        prob = raw
        if model._calibrator is not None:
            try:
                prob = float(model._calibrator.transform([raw])[0])
            except Exception as exc:
                logger.warning("calibrator failed for record at ts=%s: %s", r.get("ts"), exc)
        prob = max(0.0, min(1.0, prob))

        recorded = float(r["prob_yes"])
        delta = prob - recorded
        if abs(delta) <= args.tolerance:
            matches += 1
        else:
            mismatches.append({
                "ts": r.get("ts"),
                "recorded": recorded,
                "redecide": prob,
                "delta": delta,
            })

    n = len(records)
    match_rate = matches / n
    print(f"matches:    {matches}/{n} ({match_rate*100:.2f}%)  tolerance ±{args.tolerance}")
    print(f"mismatches: {len(mismatches)}")
    if mismatches:
        deltas = [m["delta"] for m in mismatches]
        print(f"  delta mean:   {np.mean(deltas):+.6f}")
        print(f"  delta median: {np.median(deltas):+.6f}")
        print(f"  delta max:    {max(abs(d) for d in deltas):.6f}")
        if args.show_mismatches:
            print()
            for m in mismatches[:30]:
                ts_iso = datetime.utcfromtimestamp(m["ts"]).isoformat()
                print(f"  {ts_iso}  recorded={m['recorded']:.6f}  "
                      f"redecide={m['redecide']:.6f}  delta={m['delta']:+.6f}")
            if len(mismatches) > 30:
                print(f"  ... and {len(mismatches) - 30} more")

    if match_rate >= args.min_match_rate:
        print()
        print(f"PASS — match rate {match_rate*100:.2f}% ≥ {args.min_match_rate*100:.0f}%.")
        return 0

    print()
    print(f"FAIL — match rate {match_rate*100:.2f}% below required {args.min_match_rate*100:.0f}%.")
    print("The model is not deterministic on its own recorded feature vectors. ")
    print("Investigate before any offline training experiment.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
