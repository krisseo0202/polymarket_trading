# CLAUDE.md — Polymarket BTC 5-Min Up/Down Trading Bot

This repo is a production-grade trading system. Treat it that way.
Your job (as Claude Code) is to ship correct, testable, observable changes that improve long-run expected value while respecting strict risk controls.

This is NOT financial advice. Trading can lose money quickly.

## Mission

Build an automated bot that trades Polymarket “BTC Up or Down — 5 minutes” markets with:
- deterministic decision rules,
- explicit edge thresholds,
- fractional Kelly sizing,
- strict risk limits,
- execution-aware order placement,
- full observability + reproducible backtests.

The bot MUST respect one critical reality:
- The market resolves using Chainlink BTC/USD stream start vs end price for that 5-minute window.
  Do not assume an exchange’s spot feed matches settlement perfectly.

## Workflow Orchestration

### Plan Mode Default
Before coding:
1) Write a plan to `tasks/todo.md` with checkboxes and acceptance tests.
2) If approach/scope is non-obvious, ask for confirmation BEFORE implementation.
3) Mark progress as you go; keep diffs small.

### Subagent Strategy
When useful, delegate:
- Research (docs, math, model choices)
- Refactors / cleanup
- Debugging failing tests
Then integrate.

### Self-Improvement Loop
After every significant task:
- Add 1–3 bullets to `tasks/lessons.md` about what broke, what surprised you, and what to do next time.

### Verification Before Done
A change is “done” only if:
- tests pass,
- backtest sanity checks pass (if strategy logic changed),
- logs show required decision breakdown,
- risk invariants are enforced.

### Demand Elegance (Balanced)
Prefer the simplest correct implementation that is hard to misuse.
Avoid gold-plating or abstract frameworks unless necessary.

## Repo Operating Commands

### Running
- `python bot.py --strategy <name> --interval 300`
- Prefer paper trading by default until explicitly enabled.

### Testing
- Run all tests: `pytest`
- Single test file: `pytest tests/test_<x>.py`
- Single test: `pytest -k <name>`
- Live tests require secrets; never run on mainnet funds without explicit user request.

### Training / backtest periods
All training + backtest scripts accept a unified date-range CLI via
`src/backtest/period_split.py`:

```
--training-period START:END   # required to enable period mode
--valid-period   START:END    # optional
--test-period    START:END    # optional, held-out final eval
```

Accepted formats (each side): `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SSZ` (UTC),
or Unix seconds. Use `..` as separator when both sides are ISO datetimes.
Ranges are half-open (`start <= ts < end`). When no period flags are given,
scripts fall back to the legacy `--val-ratio` / `--valid-fraction`.
Full design + examples: `tasks/todo_period_splits.md`.

### Dependencies
- Keep requirements minimal.
- Prefer pinned versions for trading-critical libs.
- Record environment changes in README or `tasks/lessons.md`.

## System model for BTC 5-minute markets

### Market definition
Each market is a discrete 5-minute window.
Outcome:
- “Up” if end price >= start price
- “Down” otherwise
Resolution uses Chainlink BTC/USD stream.

### What we are predicting
Our model outputs `p_up` = P(Up resolves) for the current 5-minute window at decision time.

Market-implied probability is approximated by:
- Up ask/bid midpoint (execution-aware), OR
- the best available price we can realistically fill at.

Edge definition:
- `edge_up = p_up - price_up_fillable`
- `edge_down = (1 - p_up) - price_down_fillable`

We only trade when edge clears threshold AFTER expected costs.

## Baseline model spec

### Minimal Poisson/Kelly baseline (first shippable)
Goal: produce a calibrated `p_up` with minimal complexity.

Model the BTC midprice as a compound Poisson jump process in 5-minute horizon:
- estimate jump arrival rate λ from recent high-frequency returns,
- estimate jump size distribution from rolling window,
- approximate probability that cumulative jump sum over 5 minutes is > 0.

Deliverable baseline:
- A function `predict_p_up(features) -> float` returning [0,1]
- Calibration step on historical windows (reliability curve / Brier score)

If Poisson fit is unstable, fall back to a logistic regression on the same feature set.

### Recommended ML alternative (second milestone)
A simple, robust classifier/regressor that trains fast and is hard to overfit:
- Gradient boosted trees (e.g., LightGBM/XGBoost) OR
- Regularized logistic regression with engineered features

Target metrics:
- Brier score improvement vs baseline
- Calibration (ECE) acceptable before deploying sizing changes

## Feature set (minimum viable)

Must-haves:
- Market features: best bid/ask, spread, depth at top N levels, midpoint, short-term price velocity of market probability
- External BTC features: 5s/15s/60s returns, realized volatility, orderflow proxy (if available), time-to-expiry seconds
- Microstructure flags: “near close” (e.g., <60s), spread widening, book thinning

Nice-to-haves:
- Cross-venue BTC price microstructure to anticipate Chainlink stream moves
- Latency estimates for order placement/cancel round trips
- A toxicity proxy (e.g., VPIN-like imbalance on Polymarket flow if you have prints)

## Position sizing

### Kelly for binary contracts (BUY Up)
If price is `x` (0<x<1) and model probability is `p`:
- EV per $1 risk is `p - x`
- Full Kelly fraction is: `f_kelly = (p - x) / (1 - x)`

For BUY Down at price `y`:
- use `p_down = 1 - p_up`
- `f_kelly = (p_down - y) / (1 - y)`

### Fractional Kelly (default)
Because `p_up` is uncertain:
- `f = k * f_kelly`
- default `k` in [0.10, 0.25] unless user overrides
- hard cap per-trade fraction regardless of f_kelly

### Empirical Kelly haircuts (preferred when backtest exists)
If you have a return distribution from realistic fills:
- run Monte Carlo resampling (path reordering)
- size to control 95th percentile drawdown
- apply uncertainty haircut based on CV of edge estimates

## Risk limits (hard)

These limits MUST be enforced by the RiskManager and verified in tests:

Capital and exposure
- Max fraction of account at risk per market: e.g. 1–3% (configurable)
- Max simultaneous open BTC 5-minute positions: 1 (per active window)
- Max notional open orders: capped (configurable)
- No doubling-down / martingale

Loss controls
- Daily loss limit: stop trading after hitting limit
- Per-trade stop-loss optional, but if used must consider market’s 5-minute resolution cadence
- Cooldown after N consecutive losses or severe slippage events

Execution safety
- Never hold both Up and Down in the SAME 5-minute market unless the strategy is explicitly “arb/market-make”
- Cancel stale orders before placing new ones
- Never place orders without verifying current market id and time-to-expiry

## Execution rules

Polymarket uses limit orders; “market orders” are marketable limit orders.
Default behavior:
- Prefer maker-style entry when time-to-expiry is sufficient
- Use taker-style entry only when:
  - edge is large enough to pay spread/fees,
  - and you require immediate fill

Order placement
- Compute fillable price from orderbook depth (not just top-of-book)
- Use GTD for short windows (expire before settlement)
- If near the end of the window, avoid opening new positions unless edge is extreme and fill is guaranteed

Order management
- Maintain an OrderTracker; reconcile open orders and positions every cycle
- Cancel/replace logic must be deterministic and rate-limited
- Treat partial fills as first-class: update position state immediately

Latency priorities
- Cancel latency matters: stale resting orders are a primary loss source
- Log: detection timestamp, submit timestamp, cancel timestamp, fill timestamp, and end-to-end round trip

## State machine

The bot must follow an explicit state machine for safety and testability.

High-level states:
- INIT → LOAD_STATE → SYNC → SELECT_MARKET → EVALUATE → (PLACE_ORDER → MANAGE_POSITION)* → ROLLOVER → ...
- ERROR → COOLDOWN → RECOVER

Key invariants:
- Exactly one active market_id at a time
- At most one net position per active market_id
- `price` always in [0,1]
- Positions must be reconciled from the exchange / API, not guessed from local memory

## Logging and observability (required)

Every cycle MUST log:
- market_id, slug, start/end timestamps, time_to_expiry
- orderbook snapshot summary (bid/ask, spread, depth)
- model output p_up, chosen side, edge after costs, sizing fraction, intended shares
- execution actions (orders submitted/canceled/filled), latency, slippage estimate
- risk: current balance, daily pnl, drawdown estimate, reason for skip/trade

Alerts / guardrails (at minimum):
- daily loss limit hit
- unusually high cancel failures
- fill rate collapse
- price outside [0,1] (should never happen)
- repeated market-id mismatch / rollover confusion

## Backtest and validation checklist

A strategy change is NOT acceptable without:
- walk-forward / out-of-sample validation
- realistic fill modeling (limit vs taker, partial fills, slippage)
- fee modeling (including any market-specific fees)
- sensitivity analysis on thresholds / k (fractional Kelly)
- calibration check for p_up

## Deployment checklist

Before turning on real trading:
- paper trading for X days with identical configuration
- verify env vars and secret handling (no secrets in logs)
- small-capital ramp with strict caps
- monitoring dashboard is live
- emergency stop works (kill switch)

## Onboarding

If you are new to this repo:
1) Read this file fully.
2) Run unit tests.
3) Run paper trading on BTC 5-minute markets.
4) Implement baseline model; validate calibration; only then touch sizing.
5) Add one improvement at a time; keep diffs small and measurable.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
