# Online retraining for logreg_v4

## Context
The bot pulls BTC ticks + orderbook snapshots into append-only CSVs during live trading. `logreg_v4` is currently a static checkpoint trained offline. We want it to refresh on a cadence so it stays calibrated as the regime drifts, without the instability of pure SGD online learning.

Design: **rolling-window batch retrain** inside a daemon thread. The live cycle polls for "new model ready" and hot-swaps only when the strategy is flat. A validation gate rejects regressions.

## Approach
- Daemon thread sleeps `interval_s`, then runs one retrain pass.
- Retrain slices `data/live_orderbook_snapshots.csv` + `data/btc_live_1s.csv` to the last `window_s` seconds, calls `train_logreg_v3.derive_labels` + `build_dataset` directly (no subprocess), fits `StandardScaler` → `LogisticRegression(C=0.1)` → `IsotonicRegression` (matches how v4 was originally trained).
- Walk-forward split: 80/20 by slot chronology.
- **Validation gate**: score the *currently-loaded prod model* on the same validation slice, accept only if `new_brier ≤ prod_brier + tolerance_brier`. On rejection, log and skip.
- **Atomic handoff**: write the new model into a *dated* directory (`models/logreg_v4_YYYYMMDD_HHMMSS/`) — never overwrite the baseline. Retrainer stores the path in a thread-safe field; the cycle runner polls.
- **Hot swap**: only when `strategy.is_flat(by_token)`. Calls `strategy.reload_model(new_dir)` which just does a fresh `LogRegV4Model.load()` and replaces `self.model_service`.
- **History log**: every retrain attempt (accepted or skipped) appends a record to `data/retrain_history.jsonl` with trained_at, n_rows, new_brier, prod_brier, accepted, reason.

## Files
- [ ] NEW `src/engine/model_retrainer.py` — Retrainer class, daemon thread, core loop.
- [ ] MODIFY `src/strategies/logreg_edge.py` — add `reload_model(new_dir)` method.
- [ ] MODIFY `src/utils/startup.py` — construct Retrainer from config, attach to Services, start it.
- [ ] MODIFY `src/engine/cycle_runner.py` — poll retrainer + swap when flat inside `run_cycle()`.
- [ ] MODIFY `config/config.yaml` — `strategies.logreg.retrain` block.
- [ ] NEW `data/retrain_history.jsonl` — created on first write by Retrainer (not pre-created).

## Key interfaces
```python
class Retrainer:
    def start(self) -> None
    def stop(self) -> None
    def has_ready_model(self) -> bool
    def consume_ready_model(self) -> Optional[str]
    def _retrain_once(self) -> dict  # returns history record

class LogRegEdgeStrategy:
    def reload_model(self, new_dir: str) -> bool
```

## Config
```yaml
strategies:
  logreg:
    retrain:
      enabled: true
      interval_s: 3600
      window_s: 172800        # 48h
      tolerance_brier: 0.005
      min_train_rows: 500     # refuse to retrain if the window is too small
      ob_csv: "data/live_orderbook_snapshots.csv"
      btc_csv: "data/btc_live_1s.csv"
      history_path: "data/retrain_history.jsonl"
      output_parent: "models"
      model_prefix: "logreg_v4"
```

## Safety properties
- Retrainer is a daemon thread: dies with the process, no persistent state to corrupt.
- CSV reads tolerate append-concurrent writes (pandas reads a snapshot; worst case = one row short).
- Directory writes are isolated (each retrain gets its own new dir), no file overwrite races.
- Validation gate makes the retrain a pure no-op on rejection — the live model never degrades.
- Hot swap is cycle-synchronized and strategy-flat-gated, so no mid-slot model switch.
- Schema check in `LogRegV4Model.__init__` catches any feature-count mismatch and disables the new model; live strategy keeps serving the old one.

## Acceptance tests
- [ ] `python3 -c "from src.engine.model_retrainer import Retrainer"` imports with no cycles.
- [ ] Manual `_retrain_once()` call on current `data/` produces a new `models/logreg_v4_*` directory.
- [ ] `LogRegV4Model.load(new_dir)` returns `ready=True` on that directory.
- [ ] A retrain-history JSONL line is appended with the required fields.
- [ ] Validation gate rejects a deliberately-mutilated "bad" candidate without touching prod.
- [ ] `strategy.reload_model(new_dir)` swaps `self.model_service` and preserves `strategy.is_flat()` semantics.


---


# RiskManager: units mismatch between signal.size (shares) and max_position_size (unclear)

## Context
`src/engine/risk_manager.py:118-124` computes
`max_size = min(limits.max_position_size, balance * limits.max_position_pct)`
and then gates on `signal.size > max_size`. But `signal.size` is the share count
returned by the strategy (e.g., 18.2 shares), while `max_position_size` is
configured as a bare number (`200` in `config.yaml`) with no units label, and
`balance * max_position_pct` is USDC. The comparison is shares-vs-what, and
whether it behaves correctly depends on the balance.

**Why it matters now:** the `logreg` strategy just had its strategy-level USDC
ceiling replaced with a Kelly + floor + RiskManager cap design. The RiskManager
cap only binds cleanly at the current $10k paper default. On a smaller live
account (e.g., $2k), `max_position_pct = 0.05` gives a cap of 100 — which if
read as shares at ~$0.55 per share = ~$55 notional, not $100. The cap shifts
under the user's feet because the units aren't explicit.

## The fix
Refactor the RiskManager to work in USDC-denominated notional:
- Compare `signal.size * signal.price` (notional) to limits, not `signal.size`.
- Rename `max_position_size` → `max_position_usdc` (or keep the name but
  document it as USDC, and validate the unit at load time).
- Migrate callers: `prob_edge`, `logreg_edge`, `btc_updown`, and the tests that
  assert size limits.

## Acceptance
- [ ] New test: $2k balance, Kelly says $100 notional, RiskManager rejects correctly.
- [ ] New test: $10k balance, Kelly says $50 notional, RiskManager passes.
- [ ] No existing strategy test regresses.
- [ ] Config migration path: accept legacy `max_position_size` with a deprecation warning.

## Why deferred
Blast radius is every strategy. Current in-flight work (Kelly floor/ceiling
rework, LogRegV4Model rename, v5 microstructure feature plumbing) touches
unrelated paths. A clean RiskManager refactor deserves its own PR with a
dedicated test pass.

# Ladder-entry backtest (4-rung scaled limit orders) — added 2026-04-20

## Context
When the strategy decides BUY at price `x` for size `y`, currently we submit
one limit order. Idea: split into 4 resting limit orders at
`x, x-0.01, x-0.02, x-0.03` with size `y/4` each, hoping for a better
average entry when the book ticks down briefly after submit.

Tradeoff: in the favorable case (model is right, market reprices toward us)
only the top rung fills and we capture 1/4 of the intended size. In the
adverse case (book drifts down before reverting) we get a better avg entry
on full size. Whether the EV trade is positive is the open question — back-
test it before touching the live execution path.

## Approach (offline replay)
Build a slot-level simulator that replays saved 5s book snapshots forward
from the model's decision time. For each historical slot where the model
fires:
1. Take the model's intended `(side, price x, size y)` at decision time.
2. Submit 4 hypothetical limit orders at `x, x-0.01, x-0.02, x-0.03`
   each sized `y/4` (rounded to tick).
3. Walk forward through the saved snapshots from decision time to slot end.
4. **Fill model (v1, taker-cross approximation):** rung at price `p`
   marks as fully filled at the first snapshot where `best_ask <= p`.
   Overestimates fill rate (ignores queue position) — accept that as
   the upper bound for a first cut.
5. Resolve at slot end: payout = 1.0 if outcome matches side, 0.0 otherwise.
   Compute realized PnL for ladder vs. single-order baseline (single order
   = `y` shares filled at `x` if `best_ask <= x` at decision time, else 0).
6. Apply the same fee/slippage model as `src/backtest/fill_sim.py` so the
   numbers are comparable to existing backtests.

## Files
- [ ] NEW `scripts/backtest_ladder_entry.py` — CLI entry point. Loads
      snapshots via `src/backtest/s3_snapshot_loader.py`, runs model
      predictions, simulates both ladder and single-order, prints summary.
- [ ] NEW `src/backtest/ladder_sim.py` — pure simulator: takes a sequence
      of (ts, best_ask) ticks + a list of rung prices/sizes, returns
      filled qty per rung and avg entry. ~150 LOC, unit-testable.
- [ ] NEW `tests/backtest/test_ladder_sim.py` — happy path, no fills,
      partial fills, all-rungs fill, ask never drops to top rung.

## CLI
```
.venv/bin/python scripts/backtest_ladder_entry.py \
  --strategy logreg_fb \
  --model-dir models/full_week_interactions_tuned \
  --training-period 2026-04-13:2026-04-20 \
  --rung-spacing 0.01 \
  --rungs 4
```
Defaults match the `logreg_fb` config in `config/config.yaml`.

## Output
Per-strategy summary printed to stdout + written to
`experiments/ladder/<run_id>/summary.json`:
- `n_slots_fired` — slots where model triggered an entry decision
- ladder vs single, side-by-side: total PnL, mean entry, fill rate
  (avg fraction of intended `y` actually filled), Sharpe, win rate
- per-rung fill rate breakdown (how often did rung 1, 2, 3, 4 fill)

## Acceptance
- [ ] Unit tests pass for `ladder_sim.py` (4 cases above).
- [ ] Backtest runs end-to-end on at least 7 days of S3 snapshots
      without crashing.
- [ ] Output reports both ladder and single-order on identical slot
      universe (same slot fires the same model decision in both).
- [ ] If ladder underperforms single-order on PnL by > 5% on the test
      window, mark the experiment as negative and DO NOT proceed to
      live plumbing.

## Open questions
- Fill model v2: should we tighten to "fills only when bid actually
  trades through" (requires a trade tape — we don't have one). v1
  taker-cross approximation is the pragmatic upper bound.
- Tick rounding: Polymarket uses 0.01 ticks for liquid markets. Confirm
  rung spacing of 0.01 is valid for all price ranges (vs 0.001 for
  thin/edge markets).

## Why backtest first
Live plumbing changes are non-trivial: `state.active_order_ids` becomes
a list, cancel logic walks the list, paper-fill simulator handles 4 orders
per signal. If the offline replay says the ladder is EV-negative on
historical data, none of that work is worth doing. Backtest tells us
in minutes what live testing would take days to confirm.

# Continuous-valued features per technical indicator — added 2026-04-20

## Context
Today's indicators (`indicators/rsi.py`, `ut_bot.py`, `fvg.py`,
`td_sequential.py`) emit a mix of continuous values and discrete events.
For ML inputs we want every indicator to expose at least one
**continuous, monotonically meaningful** feature so models can learn
gradients, not just step functions. Binary triggers (ut_buy, ut_sell)
are too sparse on a per-bar basis to carry signal.

## Proposed continuous outputs
| Indicator | Existing | Add |
|---|---|---|
| RSI | `rsi` (continuous, 0–100) | `rsi_centered = rsi - 50`; optional `rsi_slope_n` |
| UT Bot | `trail`, `buy`, `sell` (binary) | `ut_distance = (close - trail) / close` (signed % gap to trailing stop); `ut_atr_distance = (close - trail) / atr` (gap normalized by volatility); `bars_since_buy`, `bars_since_sell` |
| FVG | scalar end-of-window counts only | per-bar: `bull_count_active`, `bear_count_active`, `dist_to_nearest_bull_fvg` (% from close), `dist_to_nearest_bear_fvg`, `nearest_fvg_age_bars` |
| TD Sequential | `td_up`, `td_dn` (0–13 ordinal) | already ordinal — usable. Also: `dist_to_tdst_support`, `dist_to_tdst_resistance` as % of close (NaN → cap at large sentinel or use `has_*` flag) |

Distances should be signed and normalized (% of close or × ATR) so they
generalize across price regimes.

## Files
- [ ] MODIFY `indicators/ut_bot.py` — add `distance`, `atr_distance`,
      `bars_since_buy`, `bars_since_sell` arrays to `IndicatorResult.values`.
- [ ] MODIFY `indicators/fvg.py` — surface per-bar arrays
      (`bull_count_arr`, `bear_count_arr`, `dist_to_nearest_*`,
      `nearest_age_bars`) alongside the existing scalar end-of-window summary.
- [ ] MODIFY `indicators/td_sequential.py` — add `dist_to_tdst_support`,
      `dist_to_tdst_resistance` (% of close, NaN-aware).
- [ ] MODIFY `indicators/rsi.py` — add `rsi_centered`; optional `rsi_slope_n`.
- [ ] MODIFY `scripts/show_indicators_1s.py` — surface the new continuous
      columns in the printed DataFrame so we can sanity-check.
- [ ] NEW `tests/indicators/test_continuous_outputs.py` — assert each
      indicator emits the new arrays at length N, NaN handling matches
      spec, and distances flip sign correctly across the trail/level.

## Acceptance
- [ ] Every indicator's `IndicatorResult.values` contains at least one
      per-bar continuous array suitable as a direct model feature.
- [ ] No binary-only indicator remains; binary signals are kept but
      paired with a continuous companion (e.g. `ut_buy` + `ut_distance`).
- [ ] `scripts/show_indicators_1s.py` tail prints new columns with
      sensible numeric ranges (no all-zeros, no all-NaN beyond warmup).
- [ ] Unit tests pass.

## Why this matters for the model
Logistic regression / boosted trees can use a continuous distance as a
smooth predictor (e.g. "ut_distance is increasingly bullish as it grows
positive"). A binary `ut_buy` only fires once at the crossover and is
zero everywhere else — model can't condition on "how far above the
trail are we right now". The continuous companions give the model
something to weight every bar.


# Auto-retrain agent — verify trading logs, then re-train

## Context
We already have:
- `src/engine/model_retrainer.py` — daemon-thread retrainer for `logreg_v4`. Hourly cadence, 48h rolling window, walk-forward 80/20 split, Brier non-regression gate, atomic-symlink hot-swap.
- `scripts/diagnose_run.py` (added 2026-04-25) — joins JSONL decision log with `bot.log` Settlement records and emits per-side calibration tables, edge buckets, TTE buckets.
- `scripts/train.py` `promote_gate` — Brier + Sharpe + PnL non-regression vs probe.

What's missing:
- The existing Retrainer is **logreg-only**. `xgb_fb` (current production strategy) has no retrain loop.
- The retrainer reads from CSV (`data/**/live_orderbook_snapshots*.csv`), not from the JSONL decision log that carries the full ~164-feature vector and the post-fix gates' diagnostics.
- The promotion gate is **Brier-only**. The 04-25 disaster (NO-side anti-calibrated by 0.19, YES side fine) would not have been caught by global Brier — both sides averaged out.
- No verification step BEFORE training. Bad data (feed outages, disabled-strategy runs, regime skew) silently retrains the model on garbage.

## Goal
Out-of-process auto-retrain agent that runs on a schedule (systemd timer / cron),
reads the most recent N days of decision logs + settlements from S3, **verifies**
the data is healthy, retrains the active strategy's model only if verification
passes, and promotes the candidate only if it beats the current model on a
richer set of gates than Brier alone.

## Approach
**Stage 1 — VERIFY (gate before training).** Lift `diagnose_run.py`'s analysis
into `src/diagnose/run_audit.py` so both the CLI tool and the agent share one
implementation. Auto-skip training when:
- `n_settlements_joined < 500` — sample-count floor.
- `|per_side_calibration_gap| > 0.10` for either YES or NO — calibration is
  actively broken; retraining will propagate the bug.
- `feature_status="ready"` rate `< 80%` — feed problems, not model problems.
- Outcome distribution outside `[0.40, 0.60]` Up rate — regime skew.
Each fail logs a structured record to `data/auto_retrain_history.jsonl` with
`{ts, stage:"verify", reason, metrics}` so failures are debuggable post-hoc.

**Stage 2 — TRAIN.** Invoke `scripts/train.py` with explicit periods built from
the rolling window. Reuses existing CLI; no new training entry points.

**Stage 3 — PROMOTE GATE (extended).** Today `promote_gate` checks Brier + Sharpe
+ PnL globally. Extend to also require:
- Per-side calibration gap on the candidate is no worse than current prod by 0.05.
- Test-set Sharpe non-regression.
- Feature schema is a superset of prod (so the bot can load it without breaking).

**Stage 4 — HANDOFF.** Atomic symlink: `models/<strategy>/CURRENT` → newest
candidate dir. `bot.py` reads via the symlink each cycle. Hot-swap iff strategy
is flat. Pattern matches TFServing / MLflow.

## Architecture
```
                bot.py (live trading)
                       │ writes
                       ▼
    ┌──────────── S3: k-polymarket-data/data/<date>/ ────────────┐
    │  decision_log_*.jsonl  +  bot.log Settlement lines         │
    └────────────────────────┬───────────────────────────────────┘
                             │ reads
                             ▼
                ┌─────────────────────────────┐
                │  scripts/auto_retrain_agent │  ← systemd timer
                ├─────────────────────────────┤
                │  STAGE 1: VERIFY  (gates)   │
                │  STAGE 2: TRAIN   (train.py)│
                │  STAGE 3: PROMOTE (gate++)  │
                │  STAGE 4: HANDOFF (symlink) │
                └────────────┬────────────────┘
                             │ atomic symlink update
                             ▼
                models/<strategy>/CURRENT → models/<strategy>_<ts>/
```

## Files
- [ ] NEW `src/diagnose/run_audit.py` — extract `diagnose_run.py` core into
      reusable `audit_run(decision_log_path, bot_log_path, since_ts) -> AuditResult`.
- [ ] REFACTOR `scripts/diagnose_run.py` — thin CLI on top of `run_audit`.
- [ ] NEW `scripts/auto_retrain_agent.py` — orchestrator, ~250 LOC. Args:
      `--strategy xgb_fb --window-days 7 --dry-run`.
- [ ] MODIFY `scripts/train.py:promote_gate` — add per-side calibration check
      and feature-schema-superset check.
- [ ] NEW `infra/systemd/trading-retrainer.service` + `.timer` — runs every 6h.
- [ ] MODIFY `src/utils/startup.py` — read model dir via `CURRENT` symlink (~10 LOC).
- [ ] NEW `data/auto_retrain_history.jsonl` — created on first write.

## Verification gates (Stage 1)
| Gate | Threshold | Reason |
|---|---|---|
| `n_settlements_joined` | ≥ 500 | sample size |
| `per_side_calibration_gap` (YES, NO) | each ≤ 0.10 | the 04-25 NO disaster (-0.19) |
| `feature_status_ready_rate` | ≥ 0.80 | feed health |
| `outcome_up_rate` | within [0.40, 0.60] | regime sanity |
| `slots_with_trades` | ≥ 100 | enough decisions |

## Promotion gates (Stage 3, extending `promote_gate`)
| Gate | Threshold |
|---|---|
| `candidate_brier ≤ prod_brier + 0.005` | existing (Brier non-regression) |
| `candidate_sharpe ≥ prod_sharpe - 0.10` | existing |
| `per_side_calibration_gap_candidate ≤ per_side_gap_prod + 0.05` | NEW |
| `feature_set_candidate ⊇ feature_set_prod` | NEW (schema superset) |

## Hot-swap protocol
- Candidate persisted to `models/<strategy>_<UTC_ts>/`
- Atomic symlink update: `models/<strategy>/CURRENT_NEW → <candidate_dir>; mv -T CURRENT_NEW CURRENT`
- Bot reads `CURRENT` at each cycle; reload only when `strategy.is_flat()`
- Old candidates retained (rollback). Optional retention policy: keep last 10.

## Tests
- [ ] `tests/diagnose/test_run_audit.py` — every gate fires on synthetic data
      that violates exactly one threshold.
- [ ] `tests/scripts/test_auto_retrain_agent.py` — end-to-end with mocked S3,
      mocked train.py, asserts each stage produces a history record.
- [ ] `tests/strategies/test_current_symlink_load.py` — startup picks up
      latest model when `CURRENT` symlink rotates.

## Acceptance
- [ ] `python scripts/auto_retrain_agent.py --strategy xgb_fb --dry-run` walks
      all 4 stages on real S3 data, prints structured JSON, makes zero changes.
- [ ] Live: systemd timer fires every 6h, history JSONL accumulates one record
      per run, `models/xgb_fb/CURRENT` symlink rotates only when all gates pass.
- [ ] Retrain rejection on the 04-25 reference data (NO calibration gap = 0.19)
      is logged with `reason: "per_side_calibration_gap_no=0.189"`.

## Why this matters for production
- The current bot has no closed loop. We trained `signed_v1_trim` once on 04-22
  and have been running on it ever since, even after diagnose found systematic
  miscalibration. An auto-agent shrinks the "noticed → retrained → live" cycle
  from days to hours.
- The verification gates encode the lessons from `tasks/postmortem_2026-04-25.md`
  directly into the retrain loop. Future-us doesn't have to remember.
- Out-of-process keeps training off the bot's hot path — a runaway
  `XGBClassifier.fit` will not steal CPU from the 5s trading cycle.

---

# LLM-in-the-loop research — cycle 1 (analyst)

Design doc: `~/.gstack/projects/krisseo0202-polymarket_trading/seohj-refactor-unify-realized-pnl-apply-design-20260427-220121.md`

## Pick up here next session

- [ ] **Verify Stage 4 of `when-we-have-20-gleaming-rocket.md`**
      Background job kicked off ~2026-04-27 20:46 UTC was rebuilding
      `experiments/signed-v2/dataset.parquet` over the 13-day S3 window
      (04-16..04-28). Check if it finished, errored, or is still running.
      If done: train signed_v2 via `scripts/train.py` with
      `experiments/signed-v2/selection.yaml`. Promote gate: Brier ≤
      signed_v1_trim baseline AND per-side calibration gap < 0.10 on test.

- [ ] **Run full test suite + summary report**
      `pytest` over the calendar / recent_outcomes / ut_disagreement /
      replay_session test files added today. Confirm all green before
      moving on to the LLM loop.

- [ ] **Build `scripts/blind_replay.py` (~4 hours, ~200 LOC)**
      Cycle-1 test of the analyst LLM. Loads 04-25 decision_log + bot.log,
      runs `diagnose_run`-style aggregation (calibration table, edge
      buckets, per-side breakdown), strips postmortem-derived language,
      calls `claude-opus-4-7` with role="senior quant analyst" and output
      schema `{root_cause, evidence, proposed_fix}`. Logs response JSONL
      to `data/blind_replay/`.
      - Reuse `scripts/diagnose_run.py:_parse_settlements` and
        `_parse_decisions`. Do NOT re-implement.
      - Sanitize: no postmortem variable names or column descriptions
        leak into the prompt.

- [ ] **Score the response against the 3-part rubric**
      PASS = (a) names NO-side as the loss source, (b) identifies
      calibration as the failure mode (not "model is bad"), (c) proposes
      a specific threshold or feature change. Reproducibility check:
      re-run produces structurally similar response.

- [ ] **If PASS: open cycle 2 TODO**
      `scripts/feature_proposal.py` mirroring the analyst pattern but
      proposing specific feature additions. Outputs MUST go through
      `scripts/feature_probe.py` random-sentinel test before any train.

- [ ] **If FAIL: investigate**
      Two most likely causes: (a) diagnose output too aggregated → feed
      per-trade feature vectors instead, (b) prompt too narrow → restructure
      role / output schema. Do not loosen the rubric.

## Why this matters
The 04-25 NO-side fix (manual cycle: analyst → propose → ship → measure)
flipped PnL from −$414 to +$133 over 48 hours. We have one human-cycle
data point. Cycle 1 of `blind_replay.py` tests whether an LLM can play
the analyst role independently. PASS → scale to feature-engineer LLM.
FAIL → the loop is theater and we redesign.

---

# CLOB V2 migration (BLOCKING — bot is halted)

## Context
Polymarket cut over to CLOB V2 on **2026-04-28 ~11:00 UTC**. Symptoms:
- Bot's `get_balance()` started returning 0.00 at 04:25 PDT (= 11:25 UTC),
  exactly inside the migration window.
- Lifetime-peak circuit breaker latched (already disabled in
  `config/config.yaml:309-316` — see commit on this branch).
- Live trading currently halted. `paper_trading: true` is the safe state
  while the migration is in flight.

V1 SDKs (`py-clob-client>=0.1`) no longer function against production.
Collateral asset changed: **USDC.e → pUSD**. On-chain funds did not move,
but they now sit in USDC.e and must be wrapped before the V2 exchange
will recognize them as collateral.

Sketch of the `src/api/client.py` diff is in conversation history — three
hunks, ~12 lines. SDK preserves `OrderArgs` / `create_and_post_order` /
`OpenOrderParams` so `place_order`, `cancel_order`, etc. need no changes.

## Pre-flight gate check (run this first to know where you are)

Four gates between code and live trading. Before doing anything else,
verify which are open and which aren't. This avoids the
"is-the-bot-actually-trading?" confusion that cost us hours on 04-28.

```bash
# 1. Process running?
ps aux | grep -E "python.*bot.py" | grep -v grep
# (empty output = bot is NOT running)

# 2. Paper or live mode?
grep "paper_trading:" config/config.yaml | head -1
# (true = paper; false = live)

# 3. pUSD wrapped + V2 endpoint sees it?
.venv/bin/python scripts/check_v2_balance.py
# (exit 0 + non-zero balance = wrapped & visible; exit 1 = not yet)

# 4. Last cycle showed non-zero balance?
tail -n 50 logs/btc_updown_bot.log | grep -E "Cycle [0-9]+ done.*balance=" | tail -3
# (balance=0.00 on a live run = SDK or wrap broken; balance=<positive> = ok)
```

All four green → bot is placing real orders. Any one red → it isn't.

## Pick up here

- [x] **Install the V2 SDK** — done 2026-04-28.
      `py-clob-client-v2 1.0.0` installed in `.venv` alongside V1 (V1 not
      removed yet for rollback safety). Grep of the installed package
      confirmed V2 preserved the V1 surface: `AssetType.COLLATERAL`,
      `BalanceAllowanceParams`, `OrderArgs`, `OpenOrderParams`, `BUY`/`SELL`,
      and `ClobClient(host, key, chain_id, signature_type, funder)` are all
      unchanged. The originally-sketched `get_collateral_balance` rename
      was unnecessary.

- [x] **Apply the `src/api/client.py` diff** — done 2026-04-28.
      Reduced to a single import-block rename (three lines:
      `py_clob_client.*` → `py_clob_client_v2.*`). `requirements.txt` pinned
      to `py-clob-client-v2>=1.0.0`. Smoke test:
      `PolymarketClient(paper_trading=True).get_balance()` returns
      `10000.0`; full pytest = 500 pass / 1 unrelated pre-existing failure.

- [x] **Deprecate V1 (`py-clob-client`)** — done 2026-04-28.
      `pip uninstall -y py-clob-client` removed `0.34.6`. Repo grep for
      `py_clob_client\b` returns zero hits in main source/tests (one stale
      hit in `.claude/worktrees/agent-*/` — a separate agent's worktree,
      not main repo). Import smoke test still green post-uninstall:
      `PolymarketClient(paper_trading=True).get_balance() == 10000.0`.
      Pytest run uncovered 5 strategy failures in
      `tests/test_prob_edge_strategy.py` (e.g. `test_stop_loss_exit`
      expects SELL, strategy returns BUY) — these are pre-existing on
      this branch, unrelated to CLOB migration. File a separate ticket
      for prob_edge ↔ test reconciliation.

- [ ] **One-time on-chain wrap: USDC.e → pUSD**
      Call the Collateral Onramp's `wrap()` for the funder/proxy wallet.
      Until this happens, even a perfectly-migrated client reads
      `balance=0` because the on-chain pUSD balance is genuinely 0.
      Ref: https://docs.polymarket.com/v2-migration

- [ ] **Smoke test V2 endpoint with real creds (read-only)**
      With `PRIVATE_KEY` and `PROXY_FUNDER` exported in env:
      `.venv/bin/python scripts/check_v2_balance.py`
      Expect exit 0 + `balance = <wrapped amount>` and
      `allowance >= balance`. If exit 1 with balance=0, the wrap hasn't
      landed yet. If exit 1 with allowance=0, run
      `client.update_balance_allowance(...)` to top it up.
      Optional: `--host https://clob-v2.polymarket.com` to hit the V2
      testing URL specifically (post-cutover, prod URL also resolves to V2).

- [ ] **Flip back to prod and re-enable live**
      Restore `api.host` to `https://clob.polymarket.com` (V2 takes over
      the prod URL post-cutover), flip `paper_trading: false` in
      config.yaml:13, restart bot.

- [ ] **Verify the $348 → $0 anomaly is gone**
      Tail the bot.log for ~3 cycles after restart. Confirm
      `cycle_runner.py:287` `Cycle N done | balance=...` shows a
      non-zero balance. Re-enable circuit breaker after a stable
      session if you want belt-and-suspenders.

- [ ] **Re-place any orders that were live at cutover**
      All V1-signed open orders were wiped by the migration. Bot's
      reconciler will see zero open orders for prior tokens; new entries
      will sign under V2 cleanly.

## Out of scope (capture, do not build now)
- Order body schema changes (`nonce`/`feeRateBps` → `timestamp`/`builder`).
  The SDK handles this transparently; only relevant if we ever construct
  raw order bodies ourselves.
- The lifetime-peak circuit breaker rewrite. Disabled is fine; the
  rolling kill-switch + `max_session_loss_usdc` cover the same risk
  surface and auto-clear at session boundaries.

## Why this is the right ordering
Smoke test on the testing URL with paper-trading on means we exercise
the full V2 code path with zero capital risk. Wrap before flip means we
never see a phantom `balance=0` again. Verify cycle logs after flip
means we catch any V2 surprise within minutes, not days.
