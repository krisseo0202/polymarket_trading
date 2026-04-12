# Polymarket BTC 5-Min Up/Down Trading Bot

Automated bot for Polymarket's 5-minute BTC Up/Down binary markets. Predicts `p_up`, sizes with fractional Kelly, enforces strict risk limits, logs everything.

Markets resolve using Chainlink BTC/USD stream start vs end price.

## Run

```bash
.venv/bin/python bot.py              # live loop (paper by default)
.venv/bin/python dashboard.py        # local web dashboard
.venv/bin/python quant_desk.py       # TUI for inspection
```

Config lives in `config/config.yaml`. Secrets in `.env`.

## System layout

```
bot.py               entrypoint; single-cycle → sleep loop
dashboard.py         Dash UI over decision logs + PnL
quant_desk.py        terminal dashboard

src/
  api/               Polymarket CLOB client + types
  models/            p_up estimators (logreg_v4, v5, xgb, baseline)
  strategies/        decision rules (logreg_edge is primary)
  engine/
    cycle_runner     orchestrates one decision cycle
    risk_manager     per-trade + daily limits, circuit breakers
    execution        order placement, cancel, fill reconciliation
    inventory        position state per market
    pnl              equity + drawdown tracking
    model_retrainer  online retrain hook
    state_store      persistent slot state
  utils/             config, logging, startup

data/                live snapshots (BTC 1s, orderbooks, decision logs)
models/              trained model artifacts (logreg_v4, v5, v5_tuned, xgb)
experiments/v5/      Optuna studies + LLM suggestions (round_0, round_1, …)
scripts/             training, tuning, analysis (see below)
tests/               unit + integration
```

## Decision flow (`src/engine/cycle_runner.py`)

1. `SYNC` — pull orderbook + BTC 1s + current positions
2. `EVALUATE` — strategy calls model → `p_up`, computes edge vs fillable price
3. `RISK_CHECK` — `risk_manager` gates size, caps exposure
4. `PLACE/CANCEL` — `execution` submits limit, reconciles fills
5. `ROLLOVER` — on slot end, `inventory` settles, `pnl` updates

Exactly one active market at a time. At most one net position per market.

## Models

| Model | Features | Where |
|---|---|---|
| `logreg_v4` | 18 BTC-only (momentum, vol, vwap, rsi, td, ob) | `models/logreg_v4/` |
| `logreg_v5` | 13 compact microstructure + momentum | `models/logreg_v5/` |
| `logreg_v5_tuned` | 14 (Optuna+LLM-tuned round 2) | `models/logreg_v5_tuned/` |
| `xgb` | same v4 feature set | `models/btc_updown_xgb*` |

v4 is best calibrated; v5_tuned is the aggressive-sizing backtest winner. See `data/v4_vs_v5tuned_backtest.png`.

## Optuna + LLM tuning loop

Closed loop: Optuna searches hyperparameters multi-objectively (minimize Brier, maximize backtest equity), then Claude reads the Pareto front and proposes the next round's feature subset and tightened ranges.

```bash
ANTHROPIC_API_KEY=... .venv/bin/python scripts/tune_and_suggest_loop.py \
    --rounds 3 --trials-per-round 100
```

Relevant scripts:

| Script | Purpose |
|---|---|
| `scripts/train_and_compare_v5.py` | Train v5, score v4 on same held-out, backtest both |
| `scripts/tune_v5_optuna.py` | Single Optuna round (100 trials, NSGA-II) |
| `scripts/llm_suggest_v5.py` | Claude proposes next round's config from study results |
| `scripts/tune_and_suggest_loop.py` | Closed-loop orchestrator |
| `scripts/compare_llm_vs_control.py` | LLM-assisted vs pure-Optuna comparison |
| `scripts/finalize_tuned_v5.py` | Persist a chosen Pareto trial as `models/logreg_v5_tuned/` |
| `scripts/_v5_feature_docs.py` | Feature descriptions (LLM prompt context) |

Artifacts: `experiments/v5/round_<n>/{study.db, top_trials.json, suggestion.json, pareto.png}`.

## Risk invariants (enforced in `risk_manager`)

- Max fraction of account per market (configurable, 1–3%)
- Max 1 open BTC 5-min position at a time
- Daily loss limit halts trading
- Cooldown after N consecutive losses
- Never simultaneously long Up and Down on the same slot
- All prices must lie in [0, 1]

## Observability

Every cycle logs: `market_id`, `slug`, `time_to_expiry`, orderbook summary, `p_up`, chosen side, edge after costs, sizing fraction, executions, latencies, balance, daily PnL, skip reason.

Structured JSONL at `data/decision_log_*.jsonl`.

## Paper vs live

Paper is the default. Flip `PAPER_TRADING=false` in `.env` only after:
- N days of paper runs matching config
- Small-capital ramp with strict caps
- Kill switch verified
- Monitoring dashboard live

## Disclaimer

Trading can lose money quickly. This is not financial advice.
