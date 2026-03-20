# Feature Specification: End-to-End Last-Second BTC Probability Engine + Cycle-Aligned Execution

**Feature Branch**: `001-btc-prob-cycle-engine`
**Created**: 2026-03-18
**Status**: Draft
**Scope**: Validation + hardening of existing pipeline components. Not a rewrite.

---

## Context

The trading bot has six independently-built utility components that together form a
probability-driven execution pipeline:

1. **Real-time BTC price feed** — streams live BTC/USDT prices with reconnect and staleness detection
2. **Realized volatility estimator** — computes annualized volatility from a rolling price window
3. **Monte Carlo probability engine** — estimates the probability BTC finishes above cycle-start price
4. **Polymarket odds fetcher** — retrieves current market-implied probabilities via API
5. **Cycle scheduler** — fires an execution callback exactly once per 5-minute window, at the last moment
6. **Kelly criterion sizer** — sizes the position proportional to the detected edge

These components exist but have never been validated together end-to-end. Additionally, the
**edge detection step** (comparing bot-estimated probability against Polymarket odds to decide
whether and which side to trade) does not yet exist as a discrete component.

This feature validates each stage, adds missing test coverage, creates the edge detection
component, and wires the full pipeline into a dry-run-capable integration harness.

---

## Clarifications

### Session 2026-03-19

- Q: When UP+DOWN odds sum below threshold (thin/illiquid market), should the pipeline skip that cycle's trade? → A: Skip when `up_odds + down_odds < 0.85`; edge detector returns `NO_TRADE` and logs a liquidity warning.
- Q: What are the canonical string values for the trade side field in EdgeDecision? → A: `"BUY_UP"` / `"BUY_DOWN"` / `"NO_TRADE"` (Python-idiomatic, action-oriented).
- Q: When `fetch_market_odds` raises RuntimeError at trigger time (all retries exhausted), how should the pipeline handle it? → A: Catch inside the callback, log a warning with the reason, skip the cycle's trade — identical behaviour to an unhealthy feed.

---

## User Scenarios & Testing

### User Story 1 — Reliable Real-Time BTC Price Signal (Priority: P1)

As the bot operator, I need the BTC price feed to deliver a fresh, accurate
mid-price at any moment, so that probability estimates are never computed on stale data.

**Why this priority**: Every downstream component (volatility, probability, edge) depends on
a live BTC price. Stale or missing data causes silent garbage output. This is the foundation.

**Independent Test**: Start the feed in an isolated test environment. After a brief warm-up
period, confirm the feed reports a valid price and a healthy status. Then simulate a connection
interruption and confirm the feed recovers without manual intervention.

**Acceptance Scenarios**:

1. **Given** the price feed has just started, **When** a few seconds pass, **Then** the feed reports a valid numeric price and a healthy status.
2. **Given** the feed is actively receiving data, **When** the underlying connection is forcibly interrupted, **Then** the feed reconnects automatically and resumes valid price delivery within 10 seconds.
3. **Given** no price update has been received for more than 2 seconds, **When** the feed's health status is checked, **Then** the status is reported as unhealthy and a warning is recorded in the logs.
4. **Given** the feed has been running for at least 60 seconds, **When** recent price history is requested for that window, **Then** the feed returns a time-ordered list of price observations with no gaps greater than 5 seconds.

---

### User Story 2 — Accurate Probability Estimate (Priority: P2)

As the bot operator, I need the system to compute the probability that BTC finishes at
or above the cycle-start price, using live realized volatility and the current BTC price,
so that the edge comparison against Polymarket odds is meaningful.

**Why this priority**: The probability estimate is the signal. Errors here corrupt every
downstream decision.

**Independent Test**: Provide a known synthetic price series and fixed inputs (current price,
start price, time remaining). Verify the output is a valid probability and that known
boundary conditions (price far above start with almost no time remaining) produce probabilities
near 1.0.

**Acceptance Scenarios**:

1. **Given** the current BTC price equals the cycle-start price and 60 seconds remain, **When** the probability is estimated, **Then** the result falls in the range [0.45, 0.55], reflecting near-coin-flip uncertainty.
2. **Given** a price series of 60 or more observations, **When** realized volatility is computed, **Then** the result is a positive number that rises when prices move more erratically.
3. **Given** fewer than 2 price observations are available, **When** volatility is requested, **Then** the system returns zero without raising an error or crashing.
4. **Given** less than 1 second of cycle time remains, **When** probability is estimated, **Then** the computation completes without error and returns a valid value between 0 and 1.

---

### User Story 3 — Edge Detection Against Polymarket Odds (Priority: P3)

As the bot operator, I need the system to compare its estimated UP probability against
the current Polymarket market odds and identify when a meaningful edge exists, so that
trades are only signalled when the bot has a genuine expected advantage.

**Why this priority**: Without a formal edge check, the bot places random bets. This story
creates the decision gate between the probability engine and the order system.

**Independent Test**: Provide known bot-probability and market-odds inputs directly to the
edge detector (bypassing live API calls). Assert that signals are produced only when edge
exceeds the configured minimum threshold, and that the side with the larger edge wins when
both sides show positive edge simultaneously.

**Acceptance Scenarios**:

1. **Given** the bot estimates a 65% chance of BTC finishing up and the market prices the UP outcome at 55%, **When** the edge is evaluated, **Then** a `BUY_UP` signal is produced with the detected edge (~10%) recorded.
2. **Given** the bot estimates 52% and the market prices UP at 55%, **When** the edge is evaluated, **Then** no trade signal is produced because the edge is below the minimum threshold.
3. **Given** the Polymarket odds service is unreachable, **When** odds are requested, **Then** the system retries a configurable number of times, logs each failure, and raises a clear error after all retries are exhausted — it does not silently return stale or zero values.
4. **Given** both UP and DOWN sides simultaneously show positive edge (market mispricing), **When** the edge is evaluated, **Then** the side with the larger edge is chosen.

---

### User Story 4 — Cycle-Aligned Execution Gate (Priority: P4)

As the bot operator, I need the pipeline to trigger the probability-check-and-trade
sequence exactly once per 5-minute cycle, within the final 30 seconds of that cycle,
so that orders are placed at the optimal moment relative to market resolution.

**Why this priority**: Orders placed too early are exposed to needless time-decay risk.
Firing twice in one cycle wastes capital. Missing the window means no trade at all.

**Independent Test**: Configure a short test cycle (10 seconds) with a 3-second trigger
window. Run the scheduler for two complete cycles. Assert the callback fired exactly twice
and that each firing occurred within the trigger window.

**Acceptance Scenarios**:

1. **Given** the scheduler is aligned to the start of a 5-minute cycle, **When** 270 seconds elapse (4.5 minutes), **Then** the execution callback fires for the first time.
2. **Given** the callback has already fired in the current cycle, **When** the cycle resets and the next cycle begins, **Then** the callback fires exactly once in the new cycle — it never fires twice in the same cycle.
3. **Given** a shutdown signal is issued, **When** the scheduler receives it, **Then** the scheduler stops cleanly within 1 second with no hanging background threads.
4. **Given** the execution callback throws an unhandled exception, **When** the scheduler catches it, **Then** the error is logged, the scheduler does not crash, and the callback fires again normally in the next cycle.

---

### User Story 5 — End-to-End Pipeline Integration (Priority: P5)

As the bot operator, I need a single entry-point that wires all stages together
(price feed → volatility → probability → edge detection → cycle gate), so that
the full pipeline can be started with a market ID and runs unattended until stopped.

**Why this priority**: Each component passing in isolation does not prove the integration
works. Silent failures at component boundaries are the most dangerous class of bugs.

**Independent Test**: Start the full pipeline in dry-run mode (no real orders) against a
real Polymarket BTC market. Let it run for one complete 5-minute cycle. Confirm that a
probability estimate and an edge decision are both logged before the cycle boundary without
any crash or manual intervention.

**Acceptance Scenarios**:

1. **Given** a valid Polymarket BTC Up/Down market and an active price feed, **When** the pipeline runs through one complete 5-minute cycle, **Then** probability and edge decisions are logged exactly once within the trigger window.
2. **Given** dry-run mode is enabled, **When** the pipeline fires, **Then** no real orders are submitted, but all intermediate results (volatility, probability, edge, Kelly fraction) are logged.
3. **Given** the price feed becomes unhealthy during a cycle, **When** the trigger window opens, **Then** the pipeline logs a warning, skips order submission for that cycle, and resumes normally in the next cycle.
4. **Given** the pipeline has been running for multiple consecutive cycles, **When** it receives a stop signal, **Then** all background threads exit cleanly and no processes hang.

---

### Edge Cases

- What happens when the price feed has fewer than 2 data points at trigger time (just started)?
- What if the bot starts 90% of the way through a 5-minute cycle — does the first trigger fire correctly?
- ~~What if Polymarket odds for UP and DOWN sum to less than 0.90 (extremely thin order book)?~~ → **Resolved**: If `up_odds + down_odds < 0.85`, the edge detector returns `NO_TRADE` and logs a liquidity warning. No order is placed.
- What if BTC price has moved so far from the cycle-start price that the estimated probability is effectively 0 or 1?
- What if the Monte Carlo path count is configured very low (< 100) — is the result too noisy to act on?
- How should the pipeline handle the coexistence of two WebSocket feed implementations; which is authoritative?

---

## Requirements

### Functional Requirements

- **FR-001**: The price feed MUST report a current health status that accurately reflects whether a live price has been received within the last 2 seconds.
- **FR-002**: The price feed MUST automatically reconnect after a connection interruption and restore healthy status within 10 seconds, without operator action.
- **FR-003**: The price feed MUST maintain a rolling buffer of recent price observations covering at least the last 5 minutes.
- **FR-004**: The volatility estimator MUST return zero (not raise an error) when fewer than 2 price observations are available.
- **FR-005**: The probability engine MUST accept current price, cycle-start price, realized volatility, and time remaining, and return a value in [0.0, 1.0].
- **FR-006**: The edge detector MUST be a discrete, independently testable component that accepts bot probability and market odds as inputs and returns a trade signal (`BUY_UP` / `BUY_DOWN` / `NO_TRADE`) and edge size.
- **FR-007**: The edge detector MUST only produce a trade signal when the detected edge exceeds a configurable minimum threshold (default: 0.03 / 3 percentage points). Additionally, if `up_odds + down_odds < 0.85` (thin order book), the edge detector MUST return `NO_TRADE` regardless of computed edge and record the liquidity skip reason.
- **FR-008**: The Polymarket odds fetcher MUST retry on transient network failures and raise a clear error (not return stale or zero values) after all configured retries are exhausted.
- **FR-009**: The cycle scheduler MUST fire the execution callback exactly once per 5-minute cycle when remaining cycle time falls at or below a configurable trigger window (default: 30 seconds).
- **FR-010**: The cycle scheduler MUST use wall-clock 5-minute grid alignment as the default anchor, not bot-start-relative time.
- **FR-011**: The end-to-end pipeline MUST support a dry-run mode where all computation runs but no orders are submitted.
- **FR-012**: The pipeline MUST skip order submission and log a warning if the price feed is unhealthy when the trigger window opens. The same skip-and-warn behaviour applies if `fetch_market_odds` raises a `RuntimeError` (all retries exhausted) — both failure modes are treated identically: catch inside the callback, log, skip, resume next cycle.
- **FR-013**: All pipeline components MUST terminate cleanly when signalled to stop, with no hanging threads or processes.

### Non-Functional Requirements

- **NFR-001**: The full probability estimate (volatility + Monte Carlo) MUST complete within 500 milliseconds using the default configuration.
- **NFR-002**: The trigger window jitter (difference between the configured threshold and the actual callback firing time) MUST be less than 1 second in all cases.
- **NFR-003**: All component integration points MUST be covered by at least one automated test that does not require a live network connection.
- **NFR-004**: The test suite MUST be runnable with no wallet credentials or live exchange connections required.

### Key Entities

- **CycleTick**: One execution event per cycle — records cycle index, seconds remaining, BTC start price, BTC current price, realized volatility, estimated UP probability, market UP odds, detected edge, and the resulting trade signal (`BUY_UP` / `BUY_DOWN` / `NO_TRADE`).
- **EdgeDecision**: The output of the edge detection step — trade side (`BUY_UP` / `BUY_DOWN` / `NO_TRADE`), bot-estimated probability, market-implied probability, edge size, and Kelly-sized position fraction.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: A complete probability estimate (feed → volatility → Monte Carlo) is produced within 500 ms of the trigger window opening in every cycle.
- **SC-002**: The cycle scheduler fires the callback within 1 second of the configured trigger threshold in every observed cycle.
- **SC-003**: The full pipeline runs for 3 or more consecutive 5-minute cycles without crashing, hanging, or requiring any operator intervention.
- **SC-004**: Every triggered cycle produces logged output for all five stages: feed health check, volatility, probability, edge decision, and trade signal.
- **SC-005**: The automated test suite achieves 100% pass rate with no live network calls (all external dependencies mocked or stubbed).
- **SC-006**: When the price feed is unhealthy at trigger time, the pipeline skips the trade automatically and resumes in the next cycle — no restart required.

---

## Assumptions

- Polymarket BTC Up/Down markets resolve at each 5-minute wall-clock boundary, so wall-clock grid alignment is the correct cycle anchor.
- "Start price" for a cycle is captured at the moment the cycle boundary is crossed, and held fixed for the duration of that cycle's probability computation.
- A Monte Carlo path count of 1,000 provides sufficient signal-to-noise for a binary trade decision; a minimum edge threshold of 0.03 prevents acting on noise.
- The feature does not modify order placement or position management logic; it validates and extends the signal pipeline only.
- The `btc_feed.py` feed implementation is the authoritative one for this feature; `ws_feed.py` is treated as a lower-level building block and is out of scope unless a conflict arises.
- The test suite runs in paper-trading / dry-run mode with no live credentials required.
