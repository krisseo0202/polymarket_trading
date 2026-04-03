# TODOS

Deferred work captured from engineering review (2026-04-02).

---

## [ ] Diagnose XGB feature leakage via feature importance

**What:** Run `booster.get_score(importance_type='gain')` on the trained XGB model and inspect
whether `yes_ask`, `no_ask`, `yes_bid`, `no_bid` rank in the top features.

**Why:** These features are market-implied probability proxies — the ask IS the crowd's p_up
estimate. If the model trains heavily on them, `edge = p_model - ask` will be near zero by
construction and the model cannot systematically outperform the market.

**Pros:** Fast to diagnose (one script call). If top features are book prices, can remove them
and retrain to force the model to find alpha in BTC microstructure alone.

**Cons:** Removing book features requires full retrain + re-validation.

**Context:** Feature columns are in `src/models/schema.py:FEATURE_COLUMNS`. Model artifact is
`models/btc_updown_xgb.json`. Run: `booster.get_score(importance_type='gain')`.

**Depends on:** Nothing.

---

## [ ] Retrain XGB model after ddof=0 → ddof=1 fix

**What:** Re-run `scripts/train_btc_updown_xgb.py` after the `_realized_vol` ddof fix.

**Why:** The vol features `btc_vol_30s` and `btc_vol_60s` now produce slightly higher values
(sample std > population std, especially noticeable with small windows like 6 returns). The
trained model was fit to the old distribution — running it on the new feature values is
out-of-distribution for those two features.

**Pros:** Full benefit of the ddof fix realized in predictions.

**Cons:** Needs labeled historical data and validation run.

**Context:** Fix was applied in `src/models/feature_builder.py:_realized_vol`. Effect is ~10%
higher vol features on short windows. Validate Brier score and calibration against baseline
before promoting new model.

**Depends on:** ddof fix already merged (done in this PR).
