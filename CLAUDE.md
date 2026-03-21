# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration (How to Work in This Repo)

### 1) Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, STOP and re-plan immediately — don't keep pushing.
- Use plan mode for verification steps, not just building.
- Write detailed specs upfront to reduce ambiguity.

### 2) Subagent Strategy
- Use subagents liberally to keep the main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- For complex problems, throw more compute at it via subagents.
- One tack per subagent for focused execution.

### 3) Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern.
- Write rules for yourself that prevent the same mistake.
- Ruthlessly iterate on these lessons until mistake rate drops.
- Review relevant lessons at session start for the current project/task.

### 4) Verification Before Done
- Never mark a task complete without proving it works.
- Diff behavior between main and your changes when relevant.
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, and demonstrate correctness.

### 5) Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution."
- Skip this for simple, obvious fixes — don't over-engineer.
- Challenge your own work before presenting it.

### 6) Autonomous Bug Fixing
- When given a bug report: fix it. Don't ask for hand-holding.
- Point at logs, errors, failing tests — then resolve them.
- Zero context switching required from the user.
- Go fix failing CI tests without being told how.

## Task Management (How to Execute Work)
1. **Plan First:** Write plan to `tasks/todo.md` with checkable items.
2. **Verify Plan:** Check in before starting implementation (when scope/approach is non-obvious).
3. **Track Progress:** Mark items complete as you go.
4. **Explain Changes:** High-level summary at each step.
5. **Document Results:** Add review section to `tasks/todo.md`.
6. **Capture Lessons:** Update `tasks/lessons.md` after corrections.

## Commands

### Running the System
```bash
python bot.py              # live/paper trading (reads config/config.yaml)
python bot.py --paper      # force paper mode
python dashboard.py        # monitoring TUI (Rich)
python signal_diagnostic.py  # debug signals offline
python live_diagnostic.py    # debug live BTC feed
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/engine/test_trading_engine.py

# Run a single test
pytest tests/engine/test_trading_engine.py::test_trading_engine_execute_signal

# Run live tests (requires PRIVATE_KEY, PROXY_FUNDER env vars)
pytest tests/ --live --yes-token <TOKEN_ID> --no-token <TOKEN_ID>
# Or via env: POLY_LIVE_TEST=true pytest tests/
```

### Dependencies
Install manually (no requirements.txt present):
```bash
pip install py-clob-client requests pandas numpy pyyaml pytest rich websockets
```

## Git Workflow
- Remote URL must NOT embed tokens: `git remote set-url origin https://github.com/...`
- Token lives in osxkeychain only — clear stale credentials with:
  ```bash
  printf "protocol=https\nhost=github.com\n" | git credential-osxkeychain erase
  ```
- PR scope: fine-grained PAT needs "Contents: Read and write" + "Pull requests: Read and write"

## Architecture

### Core Flow
`bot.py` is the main entry point. It aligns to wall-clock 5-minute cycles via `CycleScheduler`, fetches BTC price from `BtcFeed` (WebSocket), runs each registered `Strategy.analyze()`, evaluates edge via `EdgeDetector` (Kelly sizing), and calls `PolymarketClient.place_order()`. State is crash-persisted to `bot_state.json` via `BotState`.

### Key Modules

**`bot.py`** — Main loop, cycle alignment, `_execute_signals`, `_snapshot_strategy_state`

**`dashboard.py`** — Rich TUI: BTC price chart, live order book, market cycle panel, strategy state panels

**`src/api/`**
- `client.py` — `PolymarketClient`: wraps `py-clob-client`'s `ClobClient`. In paper trading mode, `self.client = None` and all methods return mock data. Prices are always in `[0.0, 1.0]` probability space. Positions are fetched from `data-api.polymarket.com` (not the CLOB), using `PROXY_FUNDER` env var.
- `types.py` — All shared dataclasses: `MarketData`, `OrderBook`, `OrderBookEntry`, `Position`, `Order`, `Fill`, and the `Side` literal type.

**`src/strategies/`**
- `base.py` — Abstract `Strategy` class with `Signal` dataclass. Implement `analyze(market_data) -> List[Signal]` and `should_enter(signal) -> bool`. The `market_data` dict always has keys: `markets`, `order_books`, `price_history`, `positions`, `balance`.
- `btc_updown.py` — Momentum bias strategy: confirmation window, YES/NO entries based on BTC direction.
- `btc_vol_reversion.py` — Z-score mean reversion on BTC volatility.

**`src/engine/`**
- `risk_manager.py` — `RiskManager` + `RiskLimits`: validates signals against position/exposure/daily-loss limits and circuit breakers.
- `inventory.py` — `InventoryState`: average-cost inventory tracking with correct flip/partial-close math. Position `> 0` = long, `< 0` = short.
- `pnl.py` — `PnLTracker`: realized PnL from fills + mark-to-market unrealized PnL.
- `execution.py` — `ExecutionTracker`: polls open orders and positions, reconciles disappeared orders as FILLED/PARTIALLY_FILLED/CANCELED by comparing position deltas.
- `state_store.py` — `BotState`: crash-safe JSON persistence of bot state to `bot_state.json`.

**`src/utils/`**
- `btc_feed.py` — WebSocket BTC price feed.
- `edge_detector.py` — `EdgeDecision` + Kelly sizing: evaluates whether a signal has sufficient edge to trade.
- `cycle_scheduler.py` — Wall-clock-aligned 5-minute cycle trigger.
- `config.py` — `load_config()`: reads `config/config.yaml`, then overrides with env vars (`PRIVATE_KEY`, `FUNDER_ADDRESS`, `PAPER_TRADING`, `LOG_LEVEL`).
- `logger.py` — `setup_logger()`.

**`src/pipeline.py`** — Dry-run end-to-end pipeline harness for offline testing.

### Environment Variables
Required for live trading (store in `.env`, which is gitignored):
- `PRIVATE_KEY` — wallet private key for signing orders
- `PROXY_FUNDER` — funder/proxy address (used for positions API lookup)
- `CHAIN_ID` — defaults to `137` (Polygon)

Optional:
- `PAPER_TRADING=true` — skips real order submission
- `POLY_LIVE_TEST=true` / `YES_TOKEN_ID` / `NO_TOKEN_ID` — for live test suite

### Important Invariants
- All prices are in `[0.0, 1.0]` (probability/share price, not USD).
- Binary markets always have `YES` and `NO` outcome tokens; `market.outcome_tokens` maps `"YES"/"NO"` → `token_id`.
- `PolymarketClient` defaults `paper_trading=True`; `TradingEngine` defaults `paper_trading=False` (intentional asymmetry—client must be configured first).
- `ExecutionTracker.reconcile()` is the preferred method over `reconcile_open_orders()` (which is a thin wrapper); pass `inventories` dict to get position state synced automatically.
