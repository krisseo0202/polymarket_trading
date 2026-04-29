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

## How the bot works & formulas

This section defines the math the code runs. Every formula below is traceable
to a specific file — citations are given so you can verify against source.

### What the bot is doing, one paragraph

Polymarket runs a rolling series of 5-minute binary markets: "will BTC end
above where it started?". Two tokens trade against each other (`YES` pays \$1
if Up, `NO` pays \$1 if Down). The bot fits a model that produces `p_up ∈ [0, 1]`
from microstructure + BTC features, compares it against the live orderbook
price, and buys whichever side is mispriced relative to the model *after*
costs. Each 5-minute slot is an independent bet; positions are typically held
to expiry for a known binary payoff. Sizing uses fractional Kelly with hard
USDC caps. Risk is enforced per-trade, per-slot, and per-day.

### `p_up` — the probability of Up

The model's forecast that the YES token resolves in-the-money at slot close.
All strategies that trade on a probability ("edge" strategies) consume this
same scalar. Implementations:

- `logreg_v4` — 18-feature logistic regression with isotonic calibration
  (`src/models/logreg_v4_model.py`)
- `logreg_fb` / `logreg_edge` — 13-feature variants
  (`src/models/logreg_fb_model.py`, `src/models/logreg_model.py`)
- `btc_updown_baseline` — analytic geometric-Brownian-motion proxy used by
  `prob_edge` when no trained model is supplied (`src/models/baseline_model.py`)

Every model exposes the same interface: `predict(snapshot) → PredictionResult`
with `prob_yes`, `model_version`, and a `feature_status` string
(`src/models/prediction.py`).

### Edge — raw and net

Raw edge is how much the model's probability exceeds the price you'd pay.
Both `logreg_edge` and `prob_edge` use the same formulas; the difference
is how they derive the "price you'd pay".

**Raw edge** (`src/strategies/logreg_edge.py` and `prob_edge.py`):

```
edge_yes = p_up  − yes_ask         # buy YES  if > threshold
edge_no  = p_no  − no_ask          # buy NO   if > threshold
```

where `p_no = 1 − p_up`. For `logreg_edge` this is written equivalently as
`edge_yes = p_hat − q_t − c_t` with `q_t = (yes_bid + yes_ask) / 2` (mid)
and `c_t = (yes_ask − yes_bid) / 2` (half-spread). The algebra collapses to
the same thing: `q_t + c_t = yes_ask`.

**Net edge** — raw edge after costs. `prob_edge` uses depth-aware VWAP
instead of top-of-book `ask` and subtracts the Polymarket taker fee:

```
expected_fill = vwap(book, intended_size_usdc)
net_edge_yes  = p_up − expected_fill − fee(expected_fill, size)
```

The entry rule is `net_edge ≥ required`, where `required` tightens as
time-to-expiry shrinks (`prob_edge._required_edge(tte)` — the market is
better calibrated and execution risk is higher near close, so we demand
more edge). Re-entries pay an additional `re_entry_edge_mult` multiplier
(default 1.5×) to compensate for within-slot correlation.

**Edge stability gate** (`src/strategies/_edge_stability.py`, added this session):

```
fire signal only if edge ≥ threshold on min_stable_ticks consecutive ticks
                within the same slot
```

Counters reset at slot rollover and on any below-threshold tick. Default is
`min_stable_ticks = 1` (no debouncing). Bump to 2–3 to filter single-tick
orderbook flickers.

### Kelly sizing

Binary-contract Kelly for a BUY at price `x` with model probability `p`:

```
f_kelly = (p − x) / (1 − x)         # full Kelly fraction
f       = k × f_kelly               # fractional, k ∈ [0.10, 0.25] typical
size_usdc = clip(f × balance, min_position_size_usdc, max_position_size_usdc)
```

`k` is `strategies.<name>.kelly_fraction` in config. The strategy-level
USDC ceiling is enforced explicitly inside the strategy (do **not** rely on
`RiskManager` alone — its check is share-denominated, not USDC).

### Fee model — Polymarket taker fee

Polymarket charges **`fee_rate × shares × p × (1 − p)`** on taker fills in
crypto-category markets (zero at price extremes, max at `p = 0.5`). Makers
pay nothing; hold-to-settlement is free. `prob_edge` subtracts this from
edge when `fee_enabled: true`:

```
fee_per_share = taker_fee_rate × (p × (1 − p)) ^ taker_fee_exponent
```

Docs: `https://docs.polymarket.com/trading/fees`. Current crypto-category
values: `taker_fee_rate = 0.072`, `taker_fee_exponent = 1.0` (see
`config/config.yaml`).

### Slippage model — linear top-3 depth walk

For an order of size `s` USDC against a book with top-3 depth `d_3` and
half-spread `c_t`:

```
fill_price ≈ best_ask + (s / d_3) × spread
```

Partial fills apply when the order exceeds `d_3`. This is the rule used
by `scripts/sweep_backtest_logreg.py` and mirrors the live fill estimator
in `prob_edge._estimate_buy_vwap`.

### Information Coefficient (IC) and Information Ratio (IR)

Feature-quality metrics computed offline by
`scripts/compute_ic_ir.py`. Used to rank features before committing them
to a model.

**Per-feature IC** — Spearman rank correlation between the feature value
at decision time and the realized binary label (`1` if Up, `0` if Down)
across all snapshots:

```
IC_feature = spearman( feature_value_at_snapshot,  label )
```

Spearman (rank) is used instead of Pearson because most features have
heavy tails and several are monotone transforms of each other.

**Time-series IR** (stability of a signal across time) — compute `IC_t` on
each of `n_bins` chronological chunks of the data, then:

```
IR_ts = mean(IC_t) / std(IC_t)
```

High `|IR_ts|` means the signal is consistently directional; low means it
only worked in a subset of the history.

**Combined cross-sectional IR** — a portfolio-level read of "how much alpha
does the whole feature set carry, adjusted for collinearity":

```
N_eff    = (Σ λ_i)^2 / Σ λ_i^2          # PCA eigenvalues of feature covariance
IR_combined = mean|IC| × √N_eff
```

The naive `mean|IC| × √N` massively overstates IR here because many features
are near-duplicates (e.g. `btc_ret_5s` ↔ `btc_ret_15s`, `yes_mid` ↔ `no_mid`).
`N_eff` deflates `N` to the number of *independent* signals.

Run it: `.venv/bin/python scripts/compute_ic_ir.py` → `data/analysis/ic_ir_report.csv`.

### Features — families and what each means

All features are computed at decision time from data available at that moment;
nothing peeks at future ticks. Canonical list in `src/models/schema.FEATURE_COLUMNS`.

| Family | Examples | What it captures |
|---|---|---|
| **BTC momentum** | `btc_ret_5s`, `btc_ret_15s`, `btc_ret_30s`, `btc_ret_60s`, `btc_ret_180s` | Log-returns of BTC mid over rolling windows. Short windows catch pump-and-fade; longer windows catch drift. |
| **Volatility** | `btc_vol_15s`, `btc_vol_30s`, `btc_vol_60s`, `vol_ratio_15_60` | Realized vol (stdev of log-returns) and short-vs-long ratio. Vol regime drives p(strike-cross). |
| **Reference levels** | `strike_price`, `moneyness`, `distance_to_strike_bps` | `moneyness = (btc_mid − strike) / strike`. Near zero = coin flip; far from zero = near-certain outcome. |
| **Time** | `seconds_to_expiry` | Seconds until slot resolution. Interacts with vol: more time + more vol = more uncertainty. |
| **Orderflow** | `volume_surge_ratio`, `btc_vwap_deviation`, `cumulative_volume_delta_60s` | Proxies for who's buying vs selling BTC. CVD > 0 = more up-ticks. |
| **Polymarket microstructure** | `yes_bid`, `yes_ask`, `yes_mid`, `yes_spread`, `yes_spread_pct`, `yes_book_imbalance`, `yes_ret_30s`, and `no_*` counterparts | Top-of-book view of the market you're about to trade against. Wide spread = higher execution cost. |
| **Book depth (Family A)** | `yes_microprice`, `yes_depth_slope`, `yes_depth_concentration` (+ `no_*`) | Full-book features. Microprice = size-weighted mid, pulls toward the thicker side. Depth slope = how fast size falls off from the top. |
| **Cross-book coherence (Family B)** | `mid_sum_residual`, `mid_sum_residual_abs`, `spread_asymmetry`, `depth_asymmetry` | YES and NO should sum to 1; deviation flags market inefficiency. |
| **Within-slot path (Family C)** | `slot_high_excursion_bps`, `slot_low_excursion_bps`, `slot_drift_bps`, `slot_time_above_strike_pct`, `slot_strike_crosses` | Stateful features tracked across the 5-min window. "How close have we already come to settlement?" |
| **FVG indicator** | `active_bull_gap`, `active_bear_gap`, `latest_gap_distance_pct` | Fair-value gaps in BTC candles — unfilled imbalances that often get revisited. |
| **TD Sequential** | `bull_setup`, `bear_setup`, `buy_cd`, `sell_cd`, `buy_9`, `sell_9`, `buy_13`, `sell_13` | DeMark's exhaustion indicator. `9` setup marks a likely reversal; `13` countdown marks exhaustion. |
| **UT Bot** | `ut_bot_trend`, `ut_bot_distance_pct`, `ut_bot_buy_signal`, `ut_bot_sell_signal` | ATR-trailing-stop trend filter; boolean buy/sell on crosses. |
| **Multi-timeframe macro** | `rsi_{tf}`, `ut_{tf}_{trend\|distance_pct\|buy_signal\|sell_signal}`, `td_{tf}_{bull_setup\|bear_setup\|buy_cd\|sell_cd\|buy_9\|sell_9\|buy_13\|sell_13}` for `tf ∈ {1m, 3m, 5m, 15m, 30m, 60m, 240m}` | Same three indicators (RSI, UT Bot, TD Sequential) computed on 7 aggregation horizons from 1-minute to 4-hour. 91 columns total. See `src/models/multi_tf_features.py`. |

Indicator math (one-liners):

- **RSI(14)** — Wilder-smoothed relative-strength index over the last 14 bars:
  `RSI = 100 − 100/(1 + avg_gain/avg_loss)` with `avg_new = (avg_prev × 13 + current) / 14`
  (`indicators/rsi.py`).
- **UT Bot(ATR 10)** — `trail` is an ATR-trailing stop; `trend = 1 if close > trail else 0`;
  buy/sell signals fire on cross (`indicators/ut_bot.py`).
- **TD Sequential** — 9-bar setup (close compared to close[4]), followed by 13-bar
  countdown (close vs high/low[2]) (`indicators/td_sequential.py`).

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
