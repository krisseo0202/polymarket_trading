# Research: End-to-End BTC Probability Engine + Cycle-Aligned Execution

**Branch**: `001-btc-prob-cycle-engine` | **Date**: 2026-03-18

---

## Decision 1: Probability Estimation Approach

**Decision**: Use the existing `simulate_up_prob()` GBM Monte Carlo function (quant_desk.py) at 1,000 paths as the **Minimal** implementation. This is already built and returns a valid probability in [0,1].

**Rationale**: At 1,000 paths, empirical runtime is well under 500ms on a modern CPU (NumPy vectorized). The 95% CI half-width for a 50/50 binary with N=1000 is ±1.96 × √(0.25/1000) ≈ ±3.1 percentage points — sufficient to make a trade/no-trade decision with a 3pp minimum edge threshold.

**Minimal → Sophisticated path**:
- Minimal: `simulate_up_prob(start, current, time_left, vol, n_paths=1000)`
- Medium: Increase to 10,000 paths or add Black-Scholes closed-form: `p = N(d2)` where `d2 = (ln(S/K) + (µ - σ²/2)·T) / (σ√T)` — same result, zero variance, sub-millisecond runtime
- Sophisticated: multi-factor model (jump diffusion, vol-of-vol, order book imbalance as drift signal)

**Alternatives considered**:
- Black-Scholes closed form: faster and exact, but requires adding `scipy.stats.norm.cdf`; deferred to medium phase
- Historical simulation: more data-realistic but requires weeks of tick data; out of scope

---

## Decision 2: Volatility Estimation

**Decision**: Use `estimate_realized_vol(prices, window_sec=300, method="ema")` — EMA-weighted over the full 5-minute cycle window.

**Rationale**: EMA weights recent observations more heavily, which is better for detecting regime changes (e.g., a sudden BTC spike in the last 30 seconds changes the Vol regime quickly). The 300-second window matches the cycle length exactly.

**Minimal → Sophisticated path**:
- Minimal: EMA vol from 300s price buffer
- Medium: Separate short-window vol (30s) vs long-window vol (300s); use higher of two for risk-off behavior
- Sophisticated: GARCH(1,1) intra-day vol forecasting

**Alternatives considered**:
- `method="std"`: simpler but weights all observations equally; less responsive to recent moves
- Parkinson estimator (high-low range): requires OHLCV data not available from bookTicker stream

---

## Decision 3: Edge Detection Logic

**Decision**: Create a new `src/utils/edge_detector.py` module with a pure-function `detect_edge()` that compares bot probability against market odds. This component **does not currently exist** in the codebase.

**Rationale**: The gap between `quant_desk.py` (probability estimation) and order placement is currently unbridged. A discrete, independently-testable edge detection function is the missing link. Keeping it as a pure function (no I/O, no side effects) makes it trivially unit-testable.

**Formula**:
```
up_edge   = bot_up_prob - market_up_odds
down_edge = (1 - bot_up_prob) - market_down_odds

if up_edge   >= min_edge_threshold → signal = "BUY_UP",   edge = up_edge
if down_edge >= min_edge_threshold → signal = "BUY_DOWN", edge = down_edge
if both above threshold            → pick side with larger edge
else                               → signal = "NO_TRADE",  edge = 0
```

**Kelly fraction** computed from the detected edge: `kelly_fraction(p=bot_prob, q=1-bot_prob, odds=(1/market_odds - 1))`.

**Alternatives considered**:
- Folding edge detection into BTCUpDownStrategy: rejected — would couple probability engine to strategy class and make unit testing harder
- Information ratio threshold: more principled but requires historical edge distribution; deferred to sophisticated phase

---

## Decision 4: Price Feed

**Decision**: Use `BtcPriceFeed` from `btc_feed.py` as the authoritative feed. `ws_feed.py` is a lower-level building block that `btc_feed.py` improves upon (adds staleness detection, reconnect watchdog, rolling mid buffer).

**Rationale**: `BtcPriceFeed` already exposes `is_healthy()`, `get_latest_mid()`, and `get_recent_prices(seconds)` — exactly the interface the probability pipeline needs. No new feed code required.

---

## Decision 5: Cycle Anchor Strategy

**Decision**: Use `aligned_cycle_anchor(cycle_len=300)` from `cycle_scheduler.py` as the default anchor. This aligns to the nearest past wall-clock 5-minute mark (e.g., 14:05:00, 14:10:00), matching Polymarket's market resolution windows.

**Rationale**: Already implemented and tested in the demo. Using bot-start-relative time would drift from market resolution windows over multiple restarts.

**Cycle-start price capture**: At each cycle boundary rollover (detected via `cycle_index` increment), the pipeline captures the current BTC mid-price as the reference "start price" for that cycle.

---

## Decision 6: Pipeline Architecture

**Decision**: Thin integration module `src/utils/pipeline.py` (or `src/pipeline.py`) that owns the wiring logic. It creates the `BtcPriceFeed`, registers a callback with `run_last_second_strategy`, and the callback runs: (1) health check, (2) get prices, (3) compute vol, (4) estimate probability, (5) fetch market odds, (6) detect edge, (7) log / conditionally place order.

**Rationale**: Keeping the wiring in one file makes it easy to inspect, test, and replace components. Each component (feed, vol, prob, edge) remains independently importable and testable.

**Minimal → Sophisticated path**:
- Minimal: Sequential synchronous callback (all steps run in the callback, no caching)
- Medium: Cache market odds between callbacks (odds don't change every 500ms); add circuit breaker (skip if edge too small 3 cycles in a row)
- Sophisticated: Async pipeline, order management, position sizing with full Kelly

---

## Decision 7: Test Strategy

**Decision**: All tests use pytest with `unittest.mock.patch` for external dependencies. No new test frameworks needed.

**Coverage gaps** (existing tests do NOT cover):
- `btc_feed.py` → need `tests/utils/test_btc_feed.py`
- `cycle_scheduler.py` → need `tests/utils/test_cycle_scheduler.py`
- `quant_desk.py` → need `tests/utils/test_probability.py`
- `market_utils.fetch_market_odds` → need to extend or create `tests/utils/test_market_utils.py`
- `edge_detector.py` (new) → need `tests/utils/test_edge_detector.py`
- End-to-end → need `tests/integration/test_pipeline.py`

Existing tests (`test_kelly.py`, `test_volatility.py`) are complete and pass.

---

## Resolved Clarifications

- **Which feed?** → `btc_feed.py` (`BtcPriceFeed`). `ws_feed.py` is out of scope.
- **Start price**: Captured at cycle boundary, held fixed per cycle.
- **Min edge threshold**: 0.03 (3 percentage points) as default, configurable.
- **Path count**: 1,000 for minimal; 10,000 or Black-Scholes for medium phase.
- **Dry-run mode**: All computation runs; skip only the `client.place_order()` call.
