# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration (How to Work in This Repo)

### 1) Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, STOP and re-plan immediately — don’t keep pushing.
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
- Ask yourself: “Would a staff engineer approve this?”
- Run tests, check logs, and demonstrate correctness.

### 5) Demand Elegance (Balanced)
- For non-trivial changes: pause and ask “is there a more elegant way?”
- If a fix feels hacky: “Knowing everything I know now, implement the elegant solution.”
- Skip this for simple, obvious fixes — don’t over-engineer.
- Challenge your own work before presenting it.

### 6) Autonomous Bug Fixing
- When given a bug report: fix it. Don’t ask for hand-holding.
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

### Running the System
```bash
# Live trading (reads config/config.yaml, defaults to paper trading)
python examples/live_trading.py

# Backtesting
python examples/backtest_example.py
```

### Dependencies
Install manually (no requirements.txt present):
```bash
pip install py-clob-client requests pandas numpy pyyaml pytest
```

## Architecture

### Core Flow
`TradingEngine` orchestrates everything: it fetches markets/order books from `PolymarketClient`, passes the data dict to each registered `Strategy.analyze()`, collects `Signal` objects, validates them through `RiskManager`, and calls `PolymarketClient.place_order()`. The engine also polls positions for exit signals each cycle.

### Key Modules

**`src/api/`**
- `client.py` — `PolymarketClient`: wraps `py-clob-client`'s `ClobClient`. In paper trading mode, `self.client = None` and all methods return mock data. Prices are always in `[0.0, 1.0]` probability space. Positions are fetched from `data-api.polymarket.com` (not the CLOB), using `PROXY_FUNDER` env var.
- `types.py` — All shared dataclasses: `MarketData`, `OrderBook`, `OrderBookEntry`, `Position`, `Order`, `Fill`, and the `Side` literal type.

**`src/strategies/`**
- `base.py` — Abstract `Strategy` class with `Signal` dataclass. Implement `analyze(market_data) -> List[Signal]` and `should_enter(signal) -> bool`. The `market_data` dict always has keys: `markets`, `order_books`, `price_history`, `positions`, `balance`.
- `arbitrage.py` — Example: three-way arbitrage (YES+NO prices should sum to ~1.0) and spread arbitrage.

**`src/engine/`**
- `trading_engine.py` — `TradingEngine`: main loop calling strategies and executing signals.
- `risk_manager.py` — `RiskManager` + `RiskLimits`: validates signals against position/exposure/daily-loss limits and circuit breakers.
- `inventory.py` — `InventoryState`: average-cost inventory tracking with correct flip/partial-close math. Position `> 0` = long, `< 0` = short.
- `pnl.py` — `PnLTracker`: realized PnL from fills + mark-to-market unrealized PnL.
- `execution.py` — `ExecutionTracker`: polls open orders and positions, reconciles disappeared orders as FILLED/PARTIALLY_FILLED/CANCELED by comparing position deltas. Used for live execution tracking.

**`src/backtest/`**
- `backtester.py` — `Backtester` + `BacktestResult`: simulates strategy on historical price data, computes Sharpe, max drawdown, win rate.
- `data_loader.py` — `DataLoader`: loads historical market data for backtesting.

**`src/utils/`**
- `config.py` — `load_config()`: reads `config/config.yaml`, then overrides with env vars (`PRIVATE_KEY`, `FUNDER_ADDRESS`, `PAPER_TRADING`, `LOG_LEVEL`).
- `logger.py` — `setup_logger()`.

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

## Active Technologies
- Python 3.10+ + `numpy` (MC simulation), `websockets` (BTC feed), `requests` (Polymarket API), `pytest` + `unittest.mock` (tests) (001-btc-prob-cycle-engine)
- None (in-memory; no DB required) (001-btc-prob-cycle-engine)

## Recent Changes
- 001-btc-prob-cycle-engine: Added Python 3.10+ + `numpy` (MC simulation), `websockets` (BTC feed), `requests` (Polymarket API), `pytest` + `unittest.mock` (tests)
