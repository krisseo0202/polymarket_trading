# Unified train/val/test Period CLI — Plan

## Goal

Replace the ad-hoc `split_by_slot(df, val_ratio)` copies duplicated across
training/backtest scripts with a single CLI convention:

```
--training-period START:END
--valid-period   START:END   (optional)
--test-period    START:END   (optional, held-out final eval)
```

## Design decisions

- **One shared module**: `src/backtest/period_split.py`. All scripts that
  split data call `resolve_split_from_args(args, df, val_ratio=...)`.
- **Backward compatibility**: when no `--training-period` is passed, the
  module falls back to the legacy `--val-ratio` / `--valid-fraction`
  behavior. Existing CI and muscle-memory survive unchanged.
- **Formats accepted on each side of the `:` separator**:
  - `YYYY-MM-DD` — 00:00:00 UTC start-of-day
  - `YYYY-MM-DDTHH:MM:SSZ` — ISO 8601 UTC (trailing Z required)
  - Unix seconds (int or float)
- **Separator**: `:` is the default. Use `..` to disambiguate when both
  sides are ISO datetimes (which themselves contain `:`).
- **Half-open ranges**: `start_ts <= ts < end_ts`. Consistent with
  pandas `between` behavior.
- **Validation**: periods must be chronological (training < validation < test)
  and non-overlapping. Enforced at parse time.
- **Empty windows**: a period that matches zero rows raises `ValueError`
  rather than silently producing empty frames.

## Files touched

| File | Change |
| --- | --- |
| `src/backtest/period_split.py` | New module (328 lines). |
| `scripts/train.py` | Uses `add_period_arguments` + `resolve_split_from_args`. Held-out test block. meta.json records period config. |
| `scripts/backtest_logreg_edge.py` | Same CLI; `--valid-period` folds into train (no separate val phase). |
| `scripts/feature_probe.py` | Same CLI; `--test-period` is accepted but ignored (probe has no final test phase). |
| `tests/backtest/test_period_split.py` | 29 unit tests covering parse, validate, split, CLI hookup. |
| `tests/scripts/test_period_integration.py` | 7 integration tests against the refactored scripts. |

## Acceptance tests (all passing)

- [x] `YYYY-MM-DD`, ISO-datetime, and Unix-seconds all parse on either side.
- [x] Mixed formats (date-only + ISO datetime with internal colons) work.
- [x] `..` separator accepted as an alias.
- [x] Invalid specs raise with useful messages.
- [x] Overlapping or out-of-order periods raise.
- [x] `split_by_periods` filters by the configured time column (`slot_ts`
      by default; `snapshot_ts` or `contract_id` also supported).
- [x] Empty windows raise.
- [x] Legacy `split_by_slot(df, val_ratio)` keeps tuple return for prior tests.
- [x] train.py meta.json captures `split_mode`, `periods`, and `test` metrics.
- [x] No regression in the 383-test suite — new total is 412 passing.

## Usage examples

```
# Basic: train on March, validate on first half of April, test on second half.
python scripts/train.py \
  --selection experiments/runA/selection.yaml \
  --out models/runA \
  --training-period 2026-03-01:2026-04-01 \
  --valid-period   2026-04-01:2026-04-15 \
  --test-period    2026-04-15:2026-05-01

# ISO datetime (use .. to avoid colon ambiguity)
python scripts/backtest_logreg_edge.py \
  --training-period 2026-03-01T00:00:00Z..2026-04-01T00:00:00Z \
  --test-period     2026-04-01T00:00:00Z..2026-04-15T00:00:00Z

# Legacy (no period flags → --val-ratio behavior)
python scripts/feature_probe.py --dataset data/ds.parquet --out probes/p1
```

## meta.json audit shape (train.py)

```jsonc
{
  "split_mode": "periods",            // or "val_ratio"
  "val_ratio_fallback": null,          // set only when split_mode=="val_ratio"
  "periods": {
    "training":   {"label": "training",   "start_ts": 1772323200, "end_ts": 1775001600, "start_utc": "2026-03-01T00:00:00+00:00", "end_utc": "2026-04-01T..."},
    "validation": {...},
    "test":       {...}
  },
  "n_train_slots": 42,
  "n_val_slots": 10,
  "n_test_slots": 15,
  "test": {                            // only present when --test-period given
    "logreg": {"brier": 0.21, "sharpe": 0.8, "pnl": 12.3, "n_trades": 55},
    "xgb":    {...}
  }
}
```

## Notes on script-specific behavior

- **feature_probe.py**: validation is required (the probe compares feature
  importance on a held-out set). A `--test-period`, if passed, is logged
  and ignored — the probe has no final-test phase.
- **backtest_logreg_edge.py**: no calibrator / validation phase. A
  `--valid-period`, if passed, is folded into the training set with a
  console note. `--test-period` is **required** alongside `--training-period`.
- **train.py**: validation is required. `--test-period` is optional; when
  present, both LogReg and XGB are re-scored on the test frame after
  training + calibration, and the metrics are written to `meta["test"]`.

## Follow-ups (not in scope for this PR)

- Surface `meta.json`'s period block in any dashboards that read training
  artifacts.
- If the `multi_tf_features` warmup (3 days of BTC) becomes a constraint,
  consider a `--warmup-period` flag that reserves an additional read-only
  window before `--training-period` starts.
