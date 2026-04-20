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

## `bot.py` CLI parameters

All flags are optional. Anything left unset falls back to `config/config.yaml`.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--asset` | str | from config | Asset to trade. Supported: `BTC`, `ETH`, `SOL`, `DOGE`, `XRP`. Maps to the Polymarket slug prefix (`<asset>-updown-5m`). |
| `--strategy` | str | from config | Decision rule. Choices: `coin_toss`, `logreg_edge`, `logreg`, `logreg_fb`, `prob_edge` (from `src/utils/startup.STRATEGIES`), plus legacy `btc_updown`, `btc_updown_xgb`, `btc_vol_reversion`, `td_rsi`. |
| `--paper` | flag | false | Force paper-trading mode (ignore config's `paper_trading`). |
| `--live` | flag | false | Force live mode. Requires `PRIVATE_KEY` + `PROXY_FUNDER` in env. Prompts for confirmation unless `--no-confirm`. |
| `--no-confirm` | flag | false | Skip the interactive confirmation prompt when entering live mode. |
| `--clean` | flag | false | Start a clean session: clears `bot_state.json`, trade logs, and decision logs before the first cycle. |
| `--delta` | float | `0.025` | Minimum post-cost edge threshold for entry. A trade only fires when `edge_yes` or `edge_no` ≥ delta. |
| `--balance` | float | `10000` | Starting paper-trading balance in USDC. Ignored in live mode (real wallet balance used). |
| `--position-size` | float | `30` | Max USDC at risk per trade. Hard cap applied after Kelly sizing. |
| `--kelly` | float | `0.15` | Fractional-Kelly multiplier. `f = kelly × f_kelly`. Lower = more conservative. |
| `--exit-rule` | str | `default` | Exit strategy. `default` lets the strategy decide (may exit early on adverse move). `hold_to_expiry` always holds until slot resolution. |

### Environment variables

| Variable | Required when | Purpose |
|---|---|---|
| `PRIVATE_KEY` | live mode | Polymarket signer key (hex). Never log or commit. |
| `PROXY_FUNDER` | live mode | On-chain proxy-wallet address that holds USDC. |
| `PAPER_TRADING` | optional | `true`/`false` override for config's paper-trading flag. |
| `ANTHROPIC_API_KEY` | tuning loop only | Required by `scripts/tune_and_suggest_loop.py` for LLM-assisted Optuna suggestions. |

### Config keys (`config/config.yaml`) that `bot.py` reads

| Key | Purpose |
|---|---|
| `trading.paper_trading` | Default paper/live toggle (overridden by `--paper`/`--live`). |
| `trading.interval` | Cycle interval in seconds. Should be `300` for 5-min markets. |
| `strategies.<name>.*` | Per-strategy config block (model path, kelly, delta, size caps, min confidence). |
| `risk.max_total_exposure` | Fraction of account at risk across all markets. |
| `risk.max_daily_loss` | Daily loss fraction that halts trading. |
| `risk.max_session_loss_usdc` | Absolute USDC loss cap per bot session. |

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

## Strategies

Selectable via `--strategy` or `strategies.<name>` in `config.yaml`.

| Strategy | What it does |
|---|---|
| `logreg_edge` | Primary. Loads a trained logreg model, computes `p_up`, enters when `edge = p_model − price_fillable − costs ≥ delta`. |
| `logreg` / `logreg_fb` | Variants of the above with different feature-builder plumbing. |
| `prob_edge` | Simpler edge rule using a non-ML probability estimate. |
| `coin_toss` | Calibration baseline — always predicts 0.5. Used to sanity-check risk/execution plumbing without real signal. |
| `btc_updown` | Legacy. Rule-based Up/Down on short-term BTC returns. |
| `btc_updown_xgb` | Legacy. XGBoost model over the v4-era feature set. |
| `btc_vol_reversion` | Legacy. Volatility-reversion heuristic. |
| `td_rsi` | Legacy. TD-sequential + RSI combo. |

## Models & feature sets

All models predict `p_up` ∈ [0, 1] for the active 5-min slot. Features are computed at decision time only; nothing uses future information.

| Model | # feat | Feature set | Artifact dir |
|---|---|---|---|
| `logreg_v4` | 18 | `time_to_expiry`, `ret_5s/15s/30s/60s/180s`, `vol_15s/30s/60s`, `vol_ratio_15_60`, `volume_surge_ratio`, `vwap_deviation`, `cvd_60s`, `rsi_14`, `td_setup_net`, `spread`, `ob_imbalance`, `ob_cross_imbalance` | `models/logreg_v4/` |
| `logreg_v5` | 13 | `ret_30s/60s`, `depth_ratio`, `depth_skew`, `imbalance`, `spread`, `wall_flag`, `depth_change`, `mid_return_5s`, `acceleration`, `rolling_std_30s`, `range_30s`, `time_to_expiry` | `models/logreg_v5/` |
| `logreg_v5_tuned` | 14 | v5 minus `depth_skew`/`range_30s`; plus `vol_ratio_15_60`, `cvd_60s`, `vwap_deviation` (Optuna+LLM round 2 winner) | `models/logreg_v5_tuned/` |
| `logreg_v6` | 11 | `ret_180s`, `mid_return_5s`, `acceleration`, `cvd_60s`, `vwap_deviation`, `vol_15s`, `range_30s`, `rolling_std_30s`, `depth_skew`, `wall_flag`, `td_setup_net` (heavier on orderflow + TD) | `models/logreg_v6/` |
| `btc_updown_xgb` | 34 | BTC (`btc_mid`, `btc_ret_15s/30s/60s`, `btc_vol_30s/60s`), `strike_price`, `seconds_to_expiry`, `moneyness`, `distance_to_strike_bps`, both-side orderbook (`yes_bid/ask/mid/spread/spread_pct/book_imbalance/ret_30s` and `no_*`), FVG markers (`active_bull_gap`, `active_bear_gap`, `latest_gap_distance_pct`), TD setups + countdowns (`bull_setup`, `bear_setup`, `buy_cd`, `sell_cd`, `buy_9`, `sell_9`, `buy_13`, `sell_13`) | `models/btc_updown_xgb*` |

**Feature families:**

- **BTC momentum** — returns over rolling windows (`ret_5s` … `ret_180s`), acceleration.
- **Volatility** — realized-vol over 15/30/60s windows, ratios, rolling-std, range.
- **Orderflow** — cumulative volume delta (`cvd_60s`), VWAP deviation, volume surge ratio.
- **Microstructure** — spread, top-of-book imbalance, cross-imbalance, depth ratio, depth skew, depth change, wall detection (`wall_flag`).
- **Reference levels** — `dist_to_strike`, `moneyness`, `distance_to_strike_bps` (spot vs slot-open).
- **Indicators** — RSI-14, TD-sequential (`td_setup_net`, `bull_setup`, `bear_setup`, `buy_cd`, `sell_cd`, `buy_9/13`, `sell_9/13`), fair-value-gap markers.
- **Time** — `time_to_expiry_sec` / `seconds_to_expiry`.
- **Polymarket orderbook** (xgb only) — `yes/no_bid`, `yes/no_ask`, `yes/no_mid`, `yes/no_spread`, `yes/no_book_imbalance`, `yes/no_ret_30s`.

v4 is best calibrated; v5_tuned is the aggressive-sizing backtest winner. See `data/v4_vs_v5tuned_backtest.png`. v6 is the most recent checkpoint (orderflow-heavy). The toy logreg trained on-the-fly by `scripts/backtest_logreg_edge.py` uses a smaller 8-feature subset (`ret_15s/30s/60s`, `rolling_vol_60s`, `rsi_14`, `dist_to_strike`, `ma_12_gap`, `time_to_expiry_sec`) for baseline sanity-checking only — it is not deployed.

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

## Backtesting with realistic costs

| Script | Purpose |
|---|---|
| `scripts/backtest_logreg_edge.py` | Trains a toy logreg and backtests edge entry on live orderbook snapshots. Writes `logreg_backtest_results.png`. |
| `scripts/sweep_backtest_logreg.py` | Honest delta sweep. First-eligible entry, Polymarket crypto taker fee (7.2% × p(1-p)), linear top-3 depth-walk slippage. Defaults to `--tte-min 240s` decision zone. |
| `scripts/audit_backtest_quality.py` | Backtest quality audit: split-disjointness, feature coefficients, metrics by time-to-expiry bucket, reliability curve, tte-restricted trading with fees. |

Fee model: Polymarket charges **7.2% × shares × p × (1 − p)** on taker fills in crypto-category markets (zero at price extremes, max at p=0.5). Makers pay nothing; hold-to-settlement is free. See `https://docs.polymarket.com/trading/fees`.

Slippage model (linear): `fill_price = best_ask + (size / ask_depth_3) × spread`, with partial fills when the order exceeds top-3 depth.

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
