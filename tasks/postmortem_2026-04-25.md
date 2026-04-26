# Postmortem — 131-trade paper session, −$414 PnL

**Run:** 2026-04-25 06:54 → 19:34 UTC, 257 cycles, 131 trades, 58 wins (44%), daily_pnl −$414.62.
**Strategy:** `xgb_fb` against `models/signed_v1_trim/` (45 features, post-probe trimmed).

## What we found

`scripts/diagnose_run.py` joined the JSONL decision log with bot.log Settlement records (119 of 122 settlements joined cleanly).

### 1. Outcome distribution skew

Distinct slots traded: 110. **Up 56% / Down 44%.** Training data was on a different base rate, so the model was not recalibrated for this regime.

### 2. NO side is the entire loss

| Side | n | win% | sum_PnL | avg/trade |
|------|---:|---:|---:|---:|
| YES | 41 | 58.5% | **+$90** | +$2.20 |
| NO | 78 | 39.7% | **−$384** | −$4.92 |

YES is profitable. NO is the disaster.

### 3. Per-side calibration — NO is anti-calibrated by ~0.19

When the bot trades NO (i.e., model says p_yes < 0.5):

| Bot's predicted P(Up) | Actual P(Up) | Error |
|---:|---:|---:|
| 0.414 (mean over NO trades) | **0.603** | **−0.189** |

When the bot trades YES (model says p_yes > 0.5):

| Bot's predicted P(Up) | Actual P(Up) | Error |
|---:|---:|---:|
| 0.585 | 0.585 | **0.000** |

The model is well-calibrated in the upper half of its predicted-probability distribution. The lower half is broken. Almost certainly a calibrator issue (`xgb_calibrator.pkl` was fit on validation data with different lower-half label statistics than what we see live).

### 4. Edge bucket: bleed concentrated at threshold

| edge bucket | n | win% | sum_PnL |
|---|---:|---:|---:|
| [0.00, 0.02) | 13 | 31% | −$127 |
| [0.02, 0.04) | 24 | 42% | −$115 |
| [0.04, 0.06) | 21 | 38% | −$139 |
| **[0.06, 0.10)** | 43 | **65%** | **+$214** |
| [0.10, 0.20) | 14 | 29% | −$107 |
| [0.20, +) | 4 | 25% | −$18 |

The only profitable edge bucket is [0.06, 0.10). Below 0.06 → bleed (model overconfident at the threshold). Above 0.10 → also bleeds (probably feature-drift outliers; the model's high-edge predictions are unreliable).

## Top 3 root causes (ranked by $ lost)

1. **Asymmetric calibrator fit** — NO predictions anti-calibrated by ~0.19. Same model, well-calibrated on YES. Isotonic regression on the validation set produced wildly different effective slopes in the two halves of the predicted-probability distribution.
2. **Threshold-edge trap** — 71% of trades clustered at edge ∈ [0.01, 0.04) where win rate was ~40%. `delta` was 0.01, well below the [0.06, 0.10) sweet spot.
3. **No streak protection** — `max_session_loss_usdc` was $1000 (too lax). Bot ran to −$414 unchecked. A 20-trade rolling kill switch would have caught the bleed at ~−$200.

## Fixes shipped

- **`config/config.yaml` xgb_fb section:** `delta` 0.01 → 0.04, `max_spread_pct` 0.075 → 0.04, `min_stable_ticks` 1 → 2, `kelly_fraction` 0.10 → 0.05, `min_confidence` 0.1 → 0.55. These cut the threshold-edge trap.
- **`config/config.yaml` risk section:** `max_session_loss_usdc` 1000 → 200. New `rolling_window_n=20`, `rolling_pnl_floor_usdc=-100`, `rolling_winrate_floor=0.30` — catches bad streaks within ~20 trades.
- **`src/engine/risk_manager.py`:** `RiskManager.rolling_kill_switch()` halts new entries when the rolling sum/win-rate breaches threshold. Wired into `calculate_position_size()`.
- **`src/models/xgb_fb_model.py`:** `max_prob_yes_for_no` 0.46 → 0.32. Only the NO bucket where p_hat ≤ 0.32 (p_chosen ≥ 0.68) was profitable. Threshold tightening cuts ~85% of NO trades, keeps the 12 that worked.
- **`scripts/diagnose_run.py`:** repeatable post-run analyzer.

Expected replay (assuming gates fire on the same trades):

| | Before | After (replay) |
|---|---:|---:|
| NO trades | 78 | ~12 |
| NO PnL | −$384 | ~+$6 |
| Total trades | 119 | ~53 |
| Total PnL | −$294 | ~+$96 |

## Open follow-ups

- **Refit `xgb_calibrator.pkl`** on at least 2 weeks of recent resolved data so the lower half of the predicted-prob distribution gets proper isotonic recalibration. Until then, the `max_prob_yes_for_no=0.32` gate is the protective layer.
- **Investigate why NO calibration drifted from YES.** Could be base-rate skew, feature drift, or a numerical artifact of the isotonic fit. Worth a small notebook before the retrain.
- **Once-per-slot trade cap** for `LogRegEdgeStrategy` — `prob_edge` already has `max_trades_per_slot`; the same knob would help here. Not applied yet because the rolling kill switch covers the worst case.

## Lessons

- Per-side calibration can be wildly different even on a well-fitting global model. **Always diagnose by side**, not just globally.
- **44% win rate on its own is ambiguous** — could be a sign flip (catastrophic) or a calibration gap (recoverable). The inversion check (Task 4 of the diagnose run) cleared the sign-flip possibility before any fix landed.
- **A single bad bucket can be 71% of trades.** Tightening the threshold gate (`delta`) was the surgical fix; raising `min_confidence` would have done the same job but with worse interaction with future model versions.
- **Session-loss caps without rolling caps are too coarse.** A 20-trade rolling window catches a regime change in ~5 minutes; the daily cap takes hours.
