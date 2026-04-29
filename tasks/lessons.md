# Lessons

Short retrospective bullets. Per CLAUDE.md: add 1–3 bullets after every
significant task. Long-form incident write-ups go in
`tasks/postmortem_<date>.md` and are linked from here.

## 2026-04-25 — 131-trade paper session, −$414 PnL

Full write-up: [`tasks/postmortem_2026-04-25.md`](postmortem_2026-04-25.md).

- **Diagnose by side.** A 44% global win rate hid a YES-side that made +$90 (59% win) and a NO-side that lost −$384 (40% win). Bucket calibration revealed the NO branch was anti-calibrated by ~0.19 while YES was perfect. **Always split metrics by side, edge bucket, and TTE before drawing conclusions.**
- **Isotonic calibrators can be asymmetric across the prediction range.** A single global Brier looked fine on validation; live, the lower-half predictions were broken. **Per-side calibration tables belong in the standard post-run report.**
- **Tighten gates against the buckets that lost, not the ones that worked.** 71% of trades clustered at edge ∈ [0.01, 0.04); the only profitable bucket was edge ∈ [0.06, 0.10). Raising `delta` 0.01 → 0.04 was surgical; raising `min_confidence` would have done the same job with worse coupling to future model versions.
- **Session-loss caps without rolling caps are too coarse.** Daily $1000 cap let the bot bleed −$414. A 20-trade rolling cap (sum or win-rate) catches a regime change in minutes, not hours.

## 2026-04-24 — silent strategy rejection (`enabled: false`)

Full write-up: PR [#26](https://github.com/krisseo0202/polymarket_trading/pull/26).

- **DEBUG-level rejection logs are silent at INFO.** A `validate_signal` short-circuit on `enabled=False` produced 53 "TRADED" decision-log entries with zero actual fills. **Promote rejection logs to INFO + include the gate that fired** (`enabled`, `min_confidence`, `signal.confidence`).
- **Optimistic in-flight position state is a footgun.** Setting `active_token_id`/`entry_price` *before* the order lands creates phantom positions when the order is rejected downstream. **Reconcile from inventory in `_auto_recover_position`** — if state is set but inventory is empty, clear it.

## 2026-04-22 — signed TD/UT indicator schema

Full write-up: PR [#25](https://github.com/krisseo0202/polymarket_trading/pull/25).

- **Sign-collapsed pair columns** (e.g. `td_setup = bull_setup − bear_setup`) are lossless when the two are mutually exclusive per bar, and they make tree models easier to fit. Same applies to `ut_signal`, `td_display`, etc.
- **Trust the sentinel test before training big.** Of 96 newly-added indicator features, only 11 cleared the random-sentinel floor on the post-probe set. The trimmed 45-feature XGB beat the 114-feature version on test Sharpe (2.04 vs 1.40) — fewer features, more skill.
