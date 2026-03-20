# Quickstart: BTC Probability Engine + Cycle-Aligned Execution

**Branch**: `001-btc-prob-cycle-engine` | **Date**: 2026-03-18

---

## Prerequisites

```bash
pip install numpy websockets requests pytest
```

---

## Step 1: Verify BTC Feed (Live, ~10 seconds)

```python
# Run from repo root
import time
from src.utils.btc_feed import BtcPriceFeed

feed = BtcPriceFeed()
feed.start()
time.sleep(3)

print(f"Healthy: {feed.is_healthy()}")          # expect: True
print(f"Mid:     {feed.get_latest_mid():.2f}")   # expect: ~current BTC price
print(f"Age ms:  {feed.get_feed_age_ms():.0f}")  # expect: < 2000

recent = feed.get_recent_prices(60)
print(f"60s buf: {len(recent)} ticks")           # expect: > 0

feed.stop()
```

---

## Step 2: Verify Probability Estimate (Offline)

```python
import sys; sys.path.insert(0, ".")
from quant_desk import simulate_up_prob
from src.utils.volatility import estimate_realized_vol

# Simulate some prices
prices = [50000 + i * 10 for i in range(60)]  # flat-ish trend
vol = estimate_realized_vol(prices, window_sec=60, method="ema")
print(f"Realized vol: {vol:.6f}")              # expect: small positive number

# At-the-money, 60s left → should be ~50%
p = simulate_up_prob(
    start_price=50000,
    current_price=50000,
    time_left_sec=60,
    vol=vol if vol > 0 else 0.5,
)
print(f"UP probability: {p:.3f}")              # expect: ~0.50
```

---

## Step 3: Verify Market Odds (Live)

```python
from src.utils.market_utils import get_active_events_with_token_ids, fetch_market_odds

# Find a BTC Up/Down market
events = get_active_events_with_token_ids(
    event_keywords=["btc"],
    market_keywords=["up", "5", "minute"],
    min_volume=1000,
)
for e in events[:2]:
    print(e["title"], "—", len(e["markets"]), "markets")
    for m in e["markets"][:1]:
        print(f"  market_id={m['market_id']}  q={m['question'][:50]}")

# Fetch odds for the first match
if events:
    mid = events[0]["markets"][0]["market_id"]
    up, down = fetch_market_odds(str(mid))
    print(f"UP={up:.3f}  DOWN={down:.3f}  sum={up+down:.3f}")
```

---

## Step 4: Verify Edge Detection (after Phase 2 complete)

```python
from src.utils.edge_detector import detect_edge

# Scenario: bot thinks UP is 65%, market says 55%
d = detect_edge(bot_up_prob=0.65, market_up_odds=0.55, market_down_odds=0.45)
print(f"Side: {d.side}")           # expect: BUY_UP
print(f"Edge: {d.edge:.3f}")       # expect: ~0.10
print(f"Kelly: {d.kelly_fraction:.4f}")  # expect: small positive

# Scenario: no edge
d2 = detect_edge(bot_up_prob=0.52, market_up_odds=0.55, market_down_odds=0.45)
print(f"Side: {d2.side}")          # expect: NO_TRADE
```

---

## Step 5: Run Automated Tests

```bash
# From repo root
pytest tests/utils/ -v

# After Phase 3 — integration test
pytest tests/integration/ -v

# Full suite
pytest tests/ -v
```

Expected: all tests pass, no network calls required.

---

## Step 6: Run Full Pipeline in Dry-Run (after Phase 3 complete)

```python
import threading, time
from src.pipeline import PipelineConfig, run_pipeline

# Find a market ID first (Step 3 above)
config = PipelineConfig(
    market_id="YOUR_MARKET_ID_HERE",
    dry_run=True,
    trigger_window_s=30,   # fire in last 30s of each 5-min cycle
)

stop = threading.Event()
t = threading.Thread(target=run_pipeline, args=(config,), kwargs={"stop_event": stop})
t.daemon = True
t.start()

print("Pipeline running in dry-run mode. Ctrl-C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop.set()
    t.join(timeout=2)
    print("Stopped.")
```

Watch the logs for:
- `BtcPriceFeed started`
- Periodic heartbeats: `[cycle N] Waiting — Xs until cycle end`
- At trigger: `[cycle N] *** Trigger window open ***`
- `UP prob=0.XXX  UP odds=0.XXX  edge=+0.0XX  → BUY_UP / NO_TRADE`
