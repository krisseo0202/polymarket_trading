# Broad-feature retrain on full S3 history

## Goal
Train a new logreg_fb candidate using all available S3 data (2026-04-16..20)
with a wider feature set: top-N depth, side-split slopes, fair-value residual,
plus existing Family A–D features. Confirm or kill these via the probe before
spending Optuna time.

## New features to add to `build_live_features`

### Family A+ — top-N depth (per token)
- [ ] `{prefix}_top3_bid_depth` — sum of size at top-3 bid levels
- [ ] `{prefix}_top3_ask_depth` — sum of size at top-3 ask levels
- [ ] `{prefix}_top3_imbalance` — (top3_bid - top3_ask) / (top3_bid + top3_ask)
- [ ] `{prefix}_bid_slope` — depth slope, BID side only (existing `depth_slope` averages both)
- [ ] `{prefix}_ask_slope` — depth slope, ASK side only

### Family E — fair-value residual
- [ ] `fair_value_p_up` — Brownian closed-form `Φ(moneyness / (σ √tte))`
      with σ = btc_vol_30s scaled to slot units
- [ ] `yes_bid_residual` = `yes_bid - fair_value_p_up`
- [ ] `microprice_residual` = `yes_microprice - fair_value_p_up`

## Steps
- [ ] Add features to `feature_builder.py` + `schema.py`
- [ ] Add unit tests (`test_feature_builder_topn.py`, `test_feature_builder_fair_value.py`)
- [ ] Run `pytest tests/models/` — must stay green
- [ ] Run `feature_probe.py --start-date 2026-04-16 --end-date 2026-04-20` →
      `experiments/full-s3-broad/probe/`
- [ ] Read probe report; build `selection_broad.yaml` with surviving features
- [ ] `train.py` for fast directional read
- [ ] If positive, `tune_logreg_fb.py` for the production candidate
- [ ] Compare to `models/full_week_interactions_tuned` (current prod) — gate
      promotion via the train.py promote_gate
- [ ] If gate passes, flip `config/config.yaml:214`

## Out of scope (intentional)
- Multi-TF macro indicators — already in schema, still warming on 5 days
- Stacking / 2-stage residual models — fair-value closed-form first
