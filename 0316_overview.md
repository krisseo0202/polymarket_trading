 Plan: Core Components Overview

 Context

 User wants a simple, clear explanation of what each core component does in
 this Polymarket trading system.

 ---
 Core Components (Simple Terms)

 The Big Picture

 Market Data → Strategies → Risk Check → Place Order → Track Fills → Update P&L

 ---
 1. src/api/client.py — The Broker

 What it does: Talks to Polymarket. Fetches prices, submits orders, checks
 positions.
 Key concept: Has a "paper trading" mode where it fakes everything — no real
 money, no real API calls.

 2. src/api/types.py — The Vocabulary

 What it does: Defines all the shared data shapes used everywhere.
 MarketData = info about a market (question, tokens, volume)
 OrderBook = the current bids and asks
 Position = what you currently hold
 Order = a placed order
 Fill = a completed trade

 3. src/strategies/base.py — The Strategy Template

 What it does: Defines what every strategy must do: take in market data, return
  a list of Signal objects (buy/sell recommendations). Anyone building a
 strategy inherits from this.

 4. src/strategies/btc_updown.py — The BTC Momentum Strategy

 What it does: Watches BTC 5-minute Up/Down markets. If price is moving up, buy
  YES. If moving down, buy NO. Exits on profit target, stop loss, or time
 limit.

 5. src/strategies/arbitrage.py — The Arbitrage Finder

 What it does: Looks for pricing errors. On binary markets, YES + NO should =
 ~$1.00. If they don't, there's a trade opportunity.

 6. src/engine/trading_engine.py — The Conductor

 What it does: Runs the main loop. Fetches market data every N seconds, feeds
 it to all strategies, collects their signals, validates with risk manager, and
  fires off orders.

 7. src/engine/risk_manager.py — The Safety Net

 What it does: Checks every signal before it becomes an order. Enforces: max
 position size, max daily loss, max total exposure. Triggers a "circuit
 breaker" on large drawdowns.

 8. src/engine/inventory.py — The Position Ledger

 What it does: Tracks how much you're long or short per market, and your
 average entry price. Handles the math when you partially close or flip a
 position.

 9. src/engine/pnl.py — The Scoreboard

 What it does: Calculates profit/loss. Realized P&L comes from completed
 trades; unrealized P&L is mark-to-market on open positions.

 10. src/engine/execution.py — The Fill Tracker

 What it does: Polls the API for completed fills. Reconciles orders that
 disappeared (inferring if they were filled, partially filled, or cancelled) by
  comparing position changes.

 11. src/engine/state_store.py — The Memory

 What it does: Saves bot state (open orders, daily P&L, position snapshots) to
 disk atomically. If the bot crashes and restarts, it picks up exactly where it
  left off.

 12. src/backtest/backtester.py — The Time Machine

 What it does: Runs a strategy against historical data to see how it would have
  performed. Outputs Sharpe ratio, max drawdown, win rate, total return.

 13. src/utils/config.py — The Settings File Reader

 What it does: Loads config/config.yaml and overrides with environment
 variables (PRIVATE_KEY, PAPER_TRADING, etc.).

 14. src/utils/market_utils.py — The Toolkit

 What it does: Small helper functions: get mid price from an order book, round
 a price to tick size, fetch active events from the Gamma API.

 15. bot.py — The Production Bot

 What it does: The actual running bot. Aligns to wall-clock 5-minute
 boundaries, runs one full cycle (fetch → analyze → trade → track), shows a
 live price ticker between cycles, and handles graceful shutdown.

 ---
 How They Fit Together

 bot.py
   └── PolymarketClient   (fetch data, place orders)
   └── BTCUpDownStrategy  (generate signals)
   └── RiskManager        (validate signals)
   └── ExecutionTracker   (poll for fills)
   └── InventoryState     (track positions)
   └── PnLTracker         (compute profit/loss)
   └── StateStore         (persist state to disk)

 TradingEngine is the generic version of bot.py — it does the same
 orchestration loop but is designed for plugging in multiple strategies at
 once.