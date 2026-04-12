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
