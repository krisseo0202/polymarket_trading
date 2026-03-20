# Data Model: End-to-End BTC Probability Engine + Cycle-Aligned Execution

**Branch**: `001-btc-prob-cycle-engine` | **Date**: 2026-03-18

---

## Entities

### EdgeDecision

Represents the output of the edge-detection step for one cycle.

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `side` | `str` | Trade side: `"BUY_UP"`, `"BUY_DOWN"`, or `"NO_TRADE"` |
| `bot_up_prob` | `float` | Bot-estimated probability BTC finishes above start price (0–1) |
| `market_up_odds` | `float` | Polymarket-implied probability for the UP outcome (0–1) |
| `market_down_odds` | `float` | Polymarket-implied probability for the DOWN outcome (0–1) |
| `up_edge` | `float` | `bot_up_prob - market_up_odds` (can be negative) |
| `down_edge` | `float` | `(1 - bot_up_prob) - market_down_odds` (can be negative) |
| `edge` | `float` | Edge on the selected side (0.0 if NO_TRADE) |
| `kelly_fraction` | `float` | Kelly-sized position fraction (0–max_fraction; 0.0 if NO_TRADE) |
| `min_edge_threshold` | `float` | Threshold used to make the decision |

**Validation rules**:
- `bot_up_prob` ∈ [0.0, 1.0]
- `market_up_odds` ∈ [0.0, 1.0], `market_down_odds` ∈ [0.0, 1.0]
- `kelly_fraction` ≥ 0.0

---

### CycleTick

A structured record of one complete cycle execution event. Produced once per cycle at trigger time.

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `cycle_index` | `int` | Zero-based index of the cycle (from `cycle_index()`) |
| `triggered_at` | `float` | Unix timestamp when the callback fired |
| `remaining_s` | `float` | Seconds remaining in the cycle when fired |
| `btc_start_price` | `float` | BTC mid-price at the start of this cycle (reference level) |
| `btc_current_price` | `float` | BTC mid-price at trigger time |
| `price_move_pct` | `float` | `(current - start) / start` |
| `realized_vol` | `float` | Annualized volatility computed from the cycle's price buffer |
| `up_probability` | `float` | Monte Carlo probability BTC finishes ≥ start price |
| `edge_decision` | `EdgeDecision` | Full edge detection result |
| `feed_healthy` | `bool` | Whether the BTC feed was healthy at trigger time |
| `skipped` | `bool` | True if the cycle was skipped (e.g., unhealthy feed) |
| `skip_reason` | `str \| None` | Reason for skip, if applicable |

---

### PipelineConfig

Configuration for the end-to-end pipeline. Passed at startup.

**Fields**:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `market_id` | `str` | required | Polymarket Gamma market ID for the BTC Up/Down market |
| `cycle_len` | `int` | `300` | Cycle duration in seconds |
| `trigger_window_s` | `float` | `30.0` | Fire callback when this many seconds remain |
| `vol_window_s` | `float` | `300.0` | Lookback window for vol estimation (seconds) |
| `vol_method` | `str` | `"ema"` | Volatility method: `"std"` or `"ema"` |
| `n_paths` | `int` | `1000` | Monte Carlo paths for probability estimation |
| `min_edge` | `float` | `0.03` | Minimum edge to produce a trade signal |
| `max_kelly` | `float` | `0.05` | Maximum Kelly fraction (5% of bankroll cap) |
| `dry_run` | `bool` | `True` | If True, skip order placement |
| `btc_symbol` | `str` | `"btcusdt"` | Binance symbol for BTC feed |

---

## State Transitions

### Cycle State Machine (per cycle, managed by `run_last_second_strategy`)

```
START
  │
  ▼
WAITING (remaining > trigger_window_s)
  │  cycle_index increments → capture btc_start_price for new cycle
  │  remaining ≤ trigger_window_s
  ▼
ACTIVE (callback fires once)
  │  1. check feed health
  │  2. get recent prices
  │  3. compute vol
  │  4. estimate probability
  │  5. fetch market odds
  │  6. detect edge
  │  7. log CycleTick
  │  8. if not dry_run and edge.side != NO_TRADE → place order
  ▼
FIRED (countdown logging only)
  │  cycle_index increments
  ▼
WAITING (new cycle)
```

### Feed Health Gate

```
trigger fires
  │
  ├── feed.is_healthy() == False ──► skip cycle, log warning, CycleTick(skipped=True)
  │
  └── feed.is_healthy() == True ──► proceed to vol/prob/edge computation
```

---

## Entity Relationships

```
PipelineConfig
    │
    ├── creates → BtcPriceFeed (1 instance, lifecycle = pipeline lifetime)
    ├── creates → run_last_second_strategy loop (1 daemon thread)
    └── each cycle produces → CycleTick
                                  └── contains → EdgeDecision
```
