# Plan: Live Signal Quality & Risk Visibility

## Context

The bot is live-trading with the btc_updown / btc_vol_reversion strategy. The dashboard now
shows the market cycle countdown and strategy internals. The next gap is **observability of
signal quality and risk state**: we know what signals fired but not whether they were profitable,
and the risk manager's circuit-breaker / daily-loss state is invisible to the dashboard.

This sprint adds three things:
1. A **trade log** (append-only JSON Lines file) so every signal → fill → outcome is recorded
2. A **Risk panel** in the dashboard showing daily loss %, circuit-breaker state, and position sizing
3. A **signal quality report** script (`signal_report.py`) that reads the trade log and prints
   win rate, avg PnL per trade, Sharpe-equivalent, and per-strategy breakdown

---

## Files to Create / Modify

| File | Change |
|---|---|
| `src/engine/trade_log.py` | New — `TradeLog` class, append-only JSON Lines writer |
| `bot.py` | Write a `TradeLog` entry on every BUY fill and on every SELL fill with matched entry |
| `src/engine/state_store.py` | Add `risk_daily_loss_pct`, `risk_circuit_open` fields to `BotState` |
| `bot.py` | Snapshot risk manager state into `BotState` each cycle |
| `dashboard.py` | Add Risk panel (daily loss %, limit, circuit breaker, position size) |
| `signal_report.py` | New — standalone CLI report reader |

---

## Step 1 — `src/engine/trade_log.py`: Append-only trade journal

```python
@dataclass
class TradeEntry:
    ts: float           # wall-clock timestamp
    action: str         # "BUY" | "SELL"
    outcome: str        # "YES" | "NO"
    price: float
    size: float
    reason: str
    strategy: str
    market_id: str
    token_id: str
    entry_price: float = 0.0   # filled on SELL (matched entry price)
    realized_pnl: float = 0.0  # filled on SELL
```

`TradeLog(path)`:
- `log(entry: TradeEntry)` — atomic append: write JSON line to `<path>.tmp`, then `os.rename` appends to main file (open in append mode, no rename needed — just lock-free append)
- `load_all() -> List[TradeEntry]` — read all lines, skip corrupt

---

## Step 2 — `bot.py`: Write trade log entries

On every successful order in `_execute_signals()`:
- BUY fill → `TradeLog.log(TradeEntry(action="BUY", ...))`
- SELL fill → `TradeLog.log(TradeEntry(action="SELL", ..., entry_price=..., realized_pnl=...))`

Pass `trade_log` as a parameter to `_execute_signals()`.

---

## Step 3 — `state_store.py`: Add risk snapshot fields

```python
risk_daily_loss_pct: float = 0.0     # daily_realized_pnl / starting_balance
risk_circuit_open: bool = False       # True when circuit breaker triggered
risk_position_size: float = 0.0      # current total open position size
```

Update `load()` with `.get()` defaults (backward-compatible).

---

## Step 4 — `bot.py`: Snapshot risk state each cycle

After `_snapshot_strategy_state()` call:

```python
def _snapshot_risk_state(risk_manager, state, balance):
    limits = risk_manager.limits
    state.risk_daily_loss_pct = (
        state.daily_realized_pnl / balance if balance > 0 else 0.0
    )
    state.risk_circuit_open = getattr(risk_manager, "_circuit_open", False)
    total_pos = sum(
        abs(inv.get("position", 0))
        for inv in state.inventories.values()
    )
    state.risk_position_size = total_pos
```

---

## Step 5 — `dashboard.py`: Add Risk panel

New `_build_risk_panel(bot_state)` — 5 rows:

| Row | Content | Colour |
|---|---|---|
| Daily PnL% | `"-1.23%"` | green / red |
| Loss Limit | `"5.00%"` | dim |
| Circuit | `"OPEN"` / `"CLOSED"` | red / green |
| Position | `"12.50 sh"` | white |
| Risk | `"OK"` / `"NEAR LIMIT"` / `"LIMIT HIT"` | green / yellow / red |

Wire into `bottom_row` as a third column alongside Market Cycle and Strategy panels.

---

## Step 6 — `signal_report.py`: Standalone quality report

CLI tool reading `trades.jsonl`:

```
$ python signal_report.py
Strategy: btc_updown   Trades: 47   Win%: 59.6%   Avg PnL: +$0.23   Sharpe: 1.42
Strategy: btc_vol_rev  Trades: 12   Win%: 50.0%   Avg PnL: -$0.04   Sharpe: 0.31
```

Filters: `--since 24h`, `--strategy btc_updown`, `--csv`

---

## Step 7 — Tests

| Test file | Coverage |
|---|---|
| `tests/engine/test_trade_log.py` | append idempotency, corrupt-line skip, load_all round-trip |
| `tests/engine/test_risk_snapshot.py` | `_snapshot_risk_state` with mocked risk manager |

---

## Implementation Order

1. `trade_log.py` — no dependencies
2. `state_store.py` — add 3 risk fields
3. `bot.py` — wire trade log + risk snapshot (both in `_execute_signals` and cycle end)
4. `dashboard.py` — Risk panel + third bottom column
5. `signal_report.py` — standalone, depends on trade log format only
6. Tests

---

## Verification

1. Run `python bot.py --paper` — after first BUY signal, `trades.jsonl` contains one line
2. Run `python signal_report.py` — prints a summary table
3. Run `python dashboard.py` — Risk panel appears as third column in bottom row
4. Manually set `risk_circuit_open=true` in `bot_state.json` — dashboard shows `OPEN` in red
5. Run `pytest tests/engine/` — all green
