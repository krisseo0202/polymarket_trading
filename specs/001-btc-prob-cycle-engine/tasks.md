# Tasks: End-to-End BTC Probability Engine + Cycle-Aligned Execution

**Input**: Design documents from `/specs/001-btc-prob-cycle-engine/`
**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ quickstart.md ✅

**Delivery Strategy**: Minimal first — tests for existing components → new EdgeDetector → pipeline wiring.
Each phase leaves the repo in a passing state.

**New production files**: `src/utils/edge_detector.py`, `src/pipeline.py`
**New test files**: 6 test files + integration suite
**Existing files touched**: `src/utils/__init__.py` (exports only)

---

## Phase 1: Setup

**Purpose**: Create test directory infrastructure needed before any test files can be added.

- [X] T001 Create `tests/integration/__init__.py` (empty file, makes directory a Python package for pytest discovery)

---

## Phase 2: Foundational

**Purpose**: Confirm the existing test baseline passes before adding new tests. Any failure here must be fixed before proceeding.

**⚠️ CRITICAL**: If this step fails, stop and fix before starting user story phases.

- [X] T002 Run `pytest tests/utils/test_kelly.py tests/utils/test_volatility.py -v` and confirm all tests pass (baseline health check — no code changes needed unless a test is broken)

**Checkpoint**: Existing tests green → user story phases can begin.

---

## Phase 3: User Story 1 — Reliable Real-Time BTC Price Signal (Priority: P1) 🎯 MVP

**Goal**: Prove `BtcPriceFeed` delivers a valid, healthy price feed and handles staleness/reconnect correctly.

**Independent Test**: `pytest tests/utils/test_btc_feed.py -v` — all pass with no network connection.

- [X] T003 [P] [US1] Write `tests/utils/test_btc_feed.py` — test `BtcPriceFeed` using direct `feed._handle_message(raw_json)` injection (no real WebSocket). Cover: (1) `is_healthy()` is False before first message; (2) after valid bookTicker JSON injection `get_latest_mid()` returns correct mid = (bid+ask)/2 and `is_healthy()` is True; (3) after `stale_warn_s` elapses with no new message `is_healthy()` returns False; (4) `get_recent_prices(n)` returns only entries within the last n seconds; (5) malformed JSON passed to `_handle_message` is silently skipped (no exception raised); (6) missing "b"/"a" keys in JSON are silently skipped. Use `time.sleep` sparingly — prefer directly setting `feed._latest.local_ts` to a past value to simulate staleness without sleeping.

**Checkpoint**: `pytest tests/utils/test_btc_feed.py -v` passes → BTC feed behaviour is proven.

---

## Phase 4: User Story 2 — Accurate Probability Estimate (Priority: P2)

**Goal**: Prove `simulate_up_prob` and `estimate_realized_vol` compute valid outputs across boundary conditions.

**Independent Test**: `pytest tests/utils/test_probability.py -v` — all pass with no network connection.

- [X] T004 [P] [US2] Write `tests/utils/test_probability.py` — test `simulate_up_prob` (imported from `quant_desk`) and `estimate_realized_vol` (imported from `src.utils.volatility`). Cover: (1) at-the-money with 60s remaining returns value in [0.45, 0.55] — use `n_paths=50000` for this assertion to reduce MC variance; (2) price far above start (e.g. current=110, start=100) with 1s remaining returns value > 0.95; (3) price far below start (e.g. current=90, start=100) with 1s remaining returns value < 0.05; (4) `time_left_sec=0.001` does not raise; (5) return value is Python `float`, not `numpy.float64` (use `isinstance(result, float)`); (6) `estimate_realized_vol([100.0]*60, 60)` returns 0.0 (flat prices → zero vol); (7) increasing-price series returns positive vol.

**Checkpoint**: `pytest tests/utils/test_probability.py -v` passes → probability engine is proven.

---

## Phase 5: User Story 3 — Edge Detection Against Polymarket Odds (Priority: P3)

**Goal**: Create the missing `EdgeDetector` component and prove `fetch_market_odds` handles network failures correctly.

**Independent Test**: `pytest tests/utils/test_edge_detector.py tests/utils/test_market_utils.py -v` — all pass with no network connection.

- [X] T005 [US3] Create `src/utils/edge_detector.py` — implement `EdgeDecision` dataclass and `detect_edge()` pure function exactly as specified in `data-model.md`. `EdgeDecision` fields: `side` (str), `bot_up_prob`, `market_up_odds`, `market_down_odds`, `up_edge`, `down_edge`, `edge`, `kelly_fraction`, `min_edge_threshold` (all float except `side`). `detect_edge(bot_up_prob, market_up_odds, market_down_odds, min_edge=0.03, max_kelly=0.05)`: compute `up_edge = bot_up_prob - market_up_odds` and `down_edge = (1 - bot_up_prob) - market_down_odds`; if both ≥ min_edge pick the larger; if only one ≥ min_edge pick that side; else `side="NO_TRADE"`. Kelly fraction for the selected side uses `kelly_fraction(p, q, odds)` from `src.utils.kelly` where for UP: `p=bot_up_prob, q=1-bot_up_prob, odds=(1/market_up_odds)-1`; for NO_TRADE: `kelly_fraction=0.0`.

- [X] T006 [US3] Update `src/utils/__init__.py` — add `from .edge_detector import EdgeDecision, detect_edge` import and add both names to `__all__` list (depends on T005).

- [X] T007 [P] [US3] Write `tests/utils/test_edge_detector.py` — test `detect_edge` exhaustively (pure function, no mocking needed). Cover: (1) UP edge detected: `bot_up_prob=0.65, market_up=0.55` → `side="BUY_UP", edge≈0.10`; (2) DOWN edge detected: `bot_up_prob=0.35, market_up=0.55, market_down=0.45` → `side="BUY_DOWN"`; (3) no edge: `bot_up_prob=0.52, market_up=0.55` → `side="NO_TRADE", kelly_fraction=0.0`; (4) both edges positive, UP larger → `side="BUY_UP"`; (5) both edges positive, DOWN larger → `side="BUY_DOWN"`; (6) `kelly_fraction > 0` when side is not NO_TRADE; (7) `kelly_fraction` is capped at `max_kelly=0.05` for extreme edge; (8) `up_edge` and `down_edge` fields reflect raw computed values even for NO_TRADE. (depends on T005)

- [X] T008 [P] [US3] Write `tests/utils/test_market_utils.py` — test `fetch_market_odds` using `unittest.mock.patch("requests.get")`. Cover: (1) valid two-token market response returns `(up_prob, down_prob)` both in [0,1]; (2) empty list response from Gamma API raises `ValueError`; (3) market with fewer than 2 clobTokenIds raises `ValueError`; (4) `requests.exceptions.Timeout` on first call retries, succeeds on second → no exception raised; (5) three consecutive timeouts exhaust retries and raise `RuntimeError`; (6) HTTP 500 response is retried; (7) HTTP 404 response raises `RuntimeError` immediately without retrying. Build minimal mock response objects with `.raise_for_status()`, `.json()`, `.status_code`.

**Checkpoint**: `pytest tests/utils/test_edge_detector.py tests/utils/test_market_utils.py -v` passes → edge detector proven, odds fetcher resilience proven.

---

## Phase 6: User Story 4 — Cycle-Aligned Execution Gate (Priority: P4)

**Goal**: Prove the cycle scheduler fires exactly once per cycle within the trigger window and handles errors gracefully.

**Independent Test**: `pytest tests/utils/test_cycle_scheduler.py -v` — all pass, total runtime < 10 seconds.

- [X] T009 [P] [US4] Write `tests/utils/test_cycle_scheduler.py` — test `detect_current_cycle`, `cycle_index`, `aligned_cycle_anchor`, and `run_last_second_strategy` from `src.utils.cycle_scheduler`. Cover: (1) `detect_current_cycle(anchor, cycle_len=10)` returns correct seconds-remaining for known elapsed values (e.g. elapsed=3s → remaining=7s); (2) `detect_current_cycle` at exact boundary returns `float(cycle_len)` not 0; (3) `cycle_index` increments at the right boundary; (4) `aligned_cycle_anchor(300)` returns a timestamp divisible by 300; (5) `run_last_second_strategy` with `cycle_len=1, trigger_window_s=0.3, poll_interval_s=0.05` fires the callback exactly once in the first cycle and once in the second cycle (run for 2.2s then set stop_event); use a `threading.Event` + counter list `[0]` in the callback; (6) callback that raises `Exception` does not crash the loop — next cycle still fires; (7) setting `stop_event` causes the loop to exit within 0.2 seconds.

**Checkpoint**: `pytest tests/utils/test_cycle_scheduler.py -v` passes → scheduler timing proven.

---

## Phase 7: User Story 5 — End-to-End Pipeline Integration (Priority: P5)

**Goal**: Wire all five components into a single runnable pipeline and prove it works end-to-end without any live network calls.

**Independent Test**: `pytest tests/integration/test_pipeline.py -v` — all pass with all I/O mocked.

- [X] T010 [US5] Create `src/pipeline.py` — implement `PipelineConfig` dataclass (fields per `data-model.md`) and `run_pipeline(config: PipelineConfig, client=None, stop_event=None, logger=None)` per plan.md Phase 3.1. Inside `run_pipeline`: (1) start `BtcPriceFeed(symbol=config.btc_symbol)`; (2) compute `anchor = aligned_cycle_anchor(config.cycle_len)`; (3) track `current_cycle_idx = -1` and `btc_start_price = None`; (4) define inner callback that: checks `cycle_index(anchor)` to detect cycle rollover and capture a new `btc_start_price`; checks `feed.is_healthy()` and skips with warning log if False; calls `feed.get_recent_prices(config.vol_window_s)` to get price list; calls `estimate_realized_vol`; calls `simulate_up_prob`; calls `fetch_market_odds(config.market_id)`; calls `detect_edge`; logs a `CycleTick`-equivalent dict; if `not config.dry_run and client and decision.side != "NO_TRADE"` calls `client.place_order()`; (5) call `run_last_second_strategy(anchor, callback, cycle_len=config.cycle_len, trigger_window_s=config.trigger_window_s, stop_event=stop_event)`. Import all utilities from `src.utils` and `quant_desk`. (depends on T005, T006)

- [X] T011 [US5] Write `tests/integration/test_pipeline.py` — test `run_pipeline` with all external I/O mocked. Setup: `unittest.mock.patch` `BtcPriceFeed` to return a mock with `is_healthy()=True`, `get_latest_mid()=50000.0`, `get_recent_prices(n)=[(time.time()-i, 50000.0+i) for i in range(60)]`; patch `fetch_market_odds` to return `(0.60, 0.40)`; use `config = PipelineConfig(market_id="test-id", cycle_len=2, trigger_window_s=0.5, n_paths=100, dry_run=True)`; run `run_pipeline` in a daemon thread for 4.5 seconds then set stop_event. Assert: (1) stop_event terminates the thread within 1 second; (2) `fetch_market_odds` was called at least once; (3) no `client.place_order` call was made (dry_run=True, and client=None); (4) no exceptions were raised. Also test unhealthy feed path: mock `is_healthy()=False`; assert `fetch_market_odds` is NOT called (cycle skipped). (depends on T010)

**Checkpoint**: `pytest tests/integration/test_pipeline.py -v` passes → full pipeline proven end-to-end.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T012 Run `pytest tests/ -v` from repo root and confirm all new and existing tests pass (depends on T001–T011)

- [X] T013 [P] Validate `quickstart.md` steps 2–4 against implemented code — run each code snippet offline (steps 2 and 4 require no network), confirm output matches expected values in the quickstart guide; update any snippet that no longer matches the final implementation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — MUST PASS before user story phases
- **Phase 3 (US1)**: Depends on Phase 2
- **Phase 4 (US2)**: Depends on Phase 2 — can run in parallel with Phase 3
- **Phase 5 (US3)**: Depends on Phase 2 — can run in parallel with Phases 3 & 4; T005 must finish before T007, T006, T008 (T007/T008 are parallel to each other)
- **Phase 6 (US4)**: Depends on Phase 2 — can run in parallel with Phases 3, 4, 5
- **Phase 7 (US5)**: Depends on Phase 5 (T005/T006 must exist); T011 depends on T010
- **Phase 8 (Polish)**: Depends on all prior phases

### User Story Dependencies

| Story | Depends On | Can Parallelize With |
|-------|-----------|---------------------|
| US1 (P1) | Phase 2 complete | US2, US4 |
| US2 (P2) | Phase 2 complete | US1, US4 |
| US3 (P3) | Phase 2 complete | US1, US2, US4 |
| US4 (P4) | Phase 2 complete | US1, US2, US3 |
| US5 (P5) | T005, T006 complete | nothing (final integrator) |

### Task-Level Dependencies

```
T001 → T002 → T003, T004, T005, T009  (fan-out: all US phases)
T005 → T006, T007, T008               (T007 and T008 parallel)
T005, T006 → T010 → T011
T001–T011 → T012 → T013
```

---

## Parallel Execution Examples

### Maximum Parallelism (after T002 passes)

```
Parallel group A (all independent):
  Task: T003 — Write tests/utils/test_btc_feed.py
  Task: T004 — Write tests/utils/test_probability.py
  Task: T005 — Create src/utils/edge_detector.py
  Task: T009 — Write tests/utils/test_cycle_scheduler.py

After T005 completes, parallel group B:
  Task: T006 — Update src/utils/__init__.py
  Task: T007 — Write tests/utils/test_edge_detector.py
  Task: T008 — Write tests/utils/test_market_utils.py

After T006 completes:
  Task: T010 — Create src/pipeline.py

After T010 completes:
  Task: T011 — Write tests/integration/test_pipeline.py

After all tasks:
  Task: T012 — Run full test suite
  Task: T013 — Validate quickstart
```

---

## Implementation Strategy

### MVP (User Story 1 Only — BTC Feed Proven)

1. Complete Phase 1: T001
2. Complete Phase 2: T002 (existing tests pass)
3. Complete Phase 3: T003 (BTC feed tests)
4. **STOP and VALIDATE**: `pytest tests/utils/test_btc_feed.py -v` all green
5. BTC feed is now proven reliable — US1 delivered ✅

### Full Minimal Pipeline (All 5 User Stories)

1. T001 → T002 (foundation)
2. T003, T004, T005, T009 in parallel (tests + edge detector)
3. T006, T007, T008 (complete US3)
4. T010 (pipeline), T011 (integration tests)
5. T012 → T013 (polish)

### Single-Developer Sequential Path

```
T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010 → T011 → T012 → T013
```

---

## Notes

- `[P]` = parallelizable (different files, no shared dependencies at time of execution)
- `[USN]` = maps to User Story N in spec.md for traceability
- T003, T004, T007, T008, T009 are pure test files — zero risk to existing production code
- T005 and T010 are the only new production files — review carefully before marking complete
- All tests must pass with no live network calls (no PRIVATE_KEY, no real WebSocket needed)
- Commit after each phase checkpoint — leaves repo in known-good state
