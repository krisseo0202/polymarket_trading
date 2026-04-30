# Tier-1 feature expansion: YES/NO coherence + within-slot path + full-depth book

## Context

The current feature builder (`src/models/feature_builder.py`, 44 cols in
`src/models/schema.py`) uses only top-5 VWAP summaries of the Polymarket book
and no within-slot state. The S3 snapshots at
`s3://k-polymarket-data/data/<date>/<hour>/` already store (per
`scripts/collect_snapshots.py`):

- full `yes_bids`/`yes_asks`/`no_bids`/`no_asks` depth dicts,
- `snapshot_ts`, `slot_ts` (enables within-slot aggregation),
- Chainlink `strike` + exchange `btc_now` + source field,
- 300s rolling BTC history for vol/return horizons.

Every one of the features below is derivable from data already on disk. No new
collection is required.

## Scope (this PR)

Add three feature families and retrain XGB. Stop there. Items marked DEFERRED
below are intentionally out of scope.

### Family A — Full-depth book (stateless, per-snapshot)

Computed from `yes_bids`/`yes_asks`/`no_bids`/`no_asks` dicts. Current schema
only uses top-5 VWAP; these use the whole book.

| Feature | Definition |
|---|---|
| `yes_microprice` | `(bid_size*ask + ask_size*bid) / (bid_size + ask_size)` at L1 |
| `yes_depth_slope` | linear-reg slope of cumulative_size vs price-distance-from-mid, top 10 levels |
| `yes_depth_concentration` | `top_level_size / total_depth` (concentration = thin book beneath L1) |
| `yes_L1_imbalance` | L1-only `(bid - ask) / (bid + ask)` (vs existing top-5 VWAP imbalance) |
| `no_microprice`, `no_depth_slope`, `no_depth_concentration`, `no_L1_imbalance` | mirror of above |

### Family B — YES/NO coherence (stateless, per-snapshot)

| Feature | Definition |
|---|---|
| `mid_sum_residual` | `(yes_mid + no_mid) - 1.0` (arbitrage/mispricing residual) |
| `mid_sum_residual_abs` | `abs(mid_sum_residual)` (regime signal) |
| `spread_asymmetry` | `(yes_spread - no_spread) / (yes_spread + no_spread)` |
| `depth_asymmetry` | `(yes_total_depth - no_total_depth) / (yes_total_depth + no_total_depth)` |

### Family C — Within-slot path (stateful across a slot, resets on rollover)

Requires a per-slot running aggregator over BTC ticks since `slot_ts`.

| Feature | Definition |
|---|---|
| `slot_high_excursion_bps` | `(slot_max_btc - strike) / strike * 10000` |
| `slot_low_excursion_bps` | `(slot_min_btc - strike) / strike * 10000` |
| `slot_drift_bps` | `(btc_now - strike) / strike * 10000` (signed) |
| `slot_time_above_strike_pct` | seconds-above-strike / seconds-elapsed-since-slot-open |
| `slot_strike_crosses` | count of sign flips of `(btc - strike)` since slot_open |

## Non-goals (DEFERRED, will propose as TODOs after)

- Tier-2 features (probability velocity, spread dynamics, longer-horizon vol,
  time-of-day). Ship Tier-1 first, measure, then add.
- Tier-3 regime features (needs cross-slot joins — separate dataset shape).
- Changing the sizing logic or thresholds — purely a feature/model update.
- Live volume flow (Polymarket `/trades` not currently saved — separate data
  collection project).

## Files to change

- [ ] `src/models/schema.py` — append 17 new columns to `FEATURE_COLUMNS`;
      update `DEFAULT_FEATURE_VALUES`. Bump `MODEL_NAME` suffix (`_v2`) so
      prod keeps loading old model until the new one is validated.
- [ ] `src/models/feature_builder.py`:
  - new `_add_full_depth_features(features, yes_book_full, no_book_full)`
  - new `_add_coherence_features(features)`  *(uses values already in `features`)*
  - new `_add_slot_path_features(features, slot_state)` where `slot_state` is
    passed in by the caller
  - `build_live_features` signature gains `slot_state: Optional[SlotPathState]`
    kwarg (default None → slot_path features stay at 0.0 — back-compat)
- [ ] NEW `src/models/slot_path_state.py` — `SlotPathState` dataclass holding
      `slot_ts`, running `max_btc`, `min_btc`, `last_sign`, `cross_count`,
      `time_above`, `last_update_ts`. Methods: `update(ts, btc, strike)`,
      `reset(new_slot_ts)`, `to_features(now_ts, btc_now, strike)`.
- [ ] `src/strategies/btc_updown_xgb.py` (or wherever live features are
      assembled) — own a `SlotPathState` instance, call `update()` each tick,
      `reset()` on slot rollover, pass into `build_live_features`.
- [ ] `src/backtest/snapshot_dataset.py`:
  - `load_probability_ticks` must preserve the full `yes_bids`/`yes_asks`
    dicts (currently it only reads scalar columns). Confirm; extend if needed.
  - `_build_feature_row` must construct the full `OrderBook` from dicts (not
    just top-of-book) and drive a `SlotPathState` per slot so backtest
    features match live features exactly.
- [ ] `scripts/train_xgb_v2.py` (NEW, or param on existing `train_logreg_v3`
      analogue) — train on new feature set, write `models/btc_updown_xgb_v2_*/`
      artifacts including `schema.json` listing columns.
- [ ] `CLAUDE.md` — Feature set section: note v2 adds depth/coherence/path.

## Backward compat

Old models keep loading. `FEATURE_COLUMNS` is the v2 schema; v1 models read
their own `schema.json`. Strategy loads whichever model it's configured for.
This PR does not flip prod to v2 — that happens only after the validation gate
in the eval milestone below passes.

## Acceptance tests

- [ ] `tests/models/test_feature_builder_tier1.py` — table-driven cases:
  - full-depth features: 3-level synthetic book → assert microprice, slope,
    concentration match hand-computed values.
  - coherence: yes_mid=0.55, no_mid=0.46 → `mid_sum_residual=0.01`,
    `spread_asymmetry` sign.
  - slot-path: feed sequence of (ts, btc) ticks, assert max/min/crosses after
    each update, assert reset clears state.
- [ ] `tests/backtest/test_snapshot_dataset_depth.py` — ingest a minimal 3-tick
  fake S3 record containing real `yes_bids`/`yes_asks` dicts, assert the built
  row has non-zero depth-slope/concentration (not 0.0 default).
- [ ] `tests/backtest/test_live_vs_backtest_parity.py` — feed the same ticks
  through both `build_live_features` (with `SlotPathState`) and the backtest
  row-builder, assert every Tier-1 feature matches within 1e-9.
- [ ] Existing XGB strategy tests must still pass with the v1 model
  (`test_btc_updown_xgb_strategy.py`).

## Evaluation (gates going to prod)

Run before touching live config:

1. Build the training dataset from S3 snapshots for all saved dates.
2. Train XGB v1 (current 44 cols) and v2 (61 cols) on the same walk-forward
   split. Same hyperparams, same seed.
3. Compare on held-out tail:
   - Brier score: v2 must beat v1 by ≥ 0.003 absolute.
   - Calibration: v2 ECE ≤ v1 ECE.
   - Reliability diagram: no new systematic bias in 0.4–0.6 range (where we
     actually trade).
   - Backtest PnL: v2 Sharpe ≥ v1 Sharpe on same fill model. If v2 wins Brier
     but loses PnL, STOP and investigate — that usually means the new features
     help confidence scoring but hurt discrimination.
4. Feature-importance / permutation importance on v2. If any Tier-1 feature
   has zero importance, flag for removal.

If all four pass, propose flipping config to v2 in a follow-up PR (not this
one). If any fail, the feature work still lands (no regression to v1), and we
iterate on selection.

## Open questions for review

1. **`SlotPathState` location**: put it on the strategy instance, or inside
   `cycle_runner` as engine-level state? I proposed strategy — simpler
   ownership, but means the engine can't share it across strategies. Probably
   fine since only XGB uses it.
2. **Schema versioning strategy**: bump `MODEL_NAME` to `btc_updown_xgb_v2`
   vs. keep name and version field? I proposed bumping — clearer on-disk.
3. **Feature count creep**: 61 cols on what is probably ~thousands of training
   slots. Any concern about overfitting? (Mitigation: v2 gate is walk-forward
   Brier on held-out tail; bad features will fail there.)
4. **Chainlink divergence feature — DEFERRED (verified against a real
   snapshot).** `strike` is chainlink *slot-open*; `btc_now` is the exchange
   price. The current chainlink price is not saved (only used as a fallback
   in `collect_snapshots.py:418-422`), so `btc_now - strike` conflates
   divergence with price drift, which `slot_drift_bps` already captures.
   Follow-up task: extend the collector to record `chainlink_now` +
   `chainlink_now_ts` alongside `btc_now`, then revisit after a few days of
   dual-feed data.
