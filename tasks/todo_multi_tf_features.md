# Multi-Timeframe Macro Features — Plan

## Goal

Add contextual macro features derived from BTC price movement across 7 timeframes
(1m, 3m, 5m, 15m, 30m, 60m, 240m/4h) using TD Sequential, UT Bot, and RSI.

Features must be available in:
1. **Training pipeline** — batch computation on historical snapshots.
2. **Live inference** — computed per-tick in `feature_builder.build_live_features`.

## Design decisions (confirmed with user 2026-04-18)

- **Warmup**: always load ≥2 days of BTC 1s history before the training/session start.
- **Warmup source**: Binance REST (`/api/v3/klines` with `interval=1s`) on demand.
- **RSI**: add `indicators/rsi.py` with classic Wilder 14-period smoothing.
- **Schema**: extend the shared `FEATURE_COLUMNS` in `src/models/schema.py` (no separate v5 schema).

## Feature naming convention

Timeframe tokens: `1m 3m 5m 15m 30m 60m 240m`.

- **RSI** (7 cols): `rsi_{tf}`
- **UT Bot** (4 × 7 = 28 cols): `ut_{tf}_{trend | distance_pct | buy_signal | sell_signal}`
- **TD Sequential** (8 × 7 = 56 cols): `td_{tf}_{bull_setup | bear_setup | buy_cd | sell_cd | buy_9 | sell_9 | buy_13 | sell_13}`

**Total: 91 new feature columns.**

## Acceptance tests

- [ ] `rsi.py` produces values in `[0, 100]` and matches a known golden series within 1e-6.
- [ ] Multi-TF aggregator builds correct OHLC bars from a 1s tick stream with gaps.
- [ ] `build_live_features` returns all 91 new columns populated when given ≥2 days of BTC history.
- [ ] Training dataset builder loads Binance warmup data when CSV starts less than 2 days after the first snapshot.
- [ ] Live bot warmup at startup fetches ≥2 days of Binance 1s klines into the feed buffer.

## Milestones

### M1: Isolated feature function (no wiring)
- [ ] Implement `indicators/rsi.py` (+ unit tests).
- [ ] Implement `src/utils/ohlc_aggregator.py` — `aggregate_to_ohlc(ticks, interval_s)` (+ tests).
- [ ] Implement `src/models/multi_tf_features.py` — `compute_multi_tf_features(btc_prices, now_ts) -> dict` (+ tests).

### M2: Schema + live feature builder
- [ ] Extend `FEATURE_COLUMNS` and `DEFAULT_FEATURE_VALUES` in `src/models/schema.py`.
- [ ] Wire `compute_multi_tf_features` into `feature_builder.build_live_features`.
- [ ] Update feature builder tests.

### M3: Binance REST historical client
- [ ] `src/utils/binance_historical.py` — `fetch_btc_klines(start_ts, end_ts, interval) -> DataFrame`.
- [ ] Chunked pagination, retries, rate-limit handling.
- [ ] Tests (mocked HTTP).

### M4: Training pipeline integration
- [ ] `scripts/build_probability_snapshot_dataset.py` loads ≥2-day warmup from Binance before first snapshot.
- [ ] Multi-TF features computed per snapshot, merged into dataset.

### M5: Live bot warmup
- [ ] `BtcPriceFeed` or startup warms its rolling buffer with Binance klines at boot.
- [ ] Buffer retains ≥2 days of history for the 4-hour timeframe to stay warm.

## Open risks / notes

- **Buffer memory**: 2 days of 1s BTC = ~172k ticks. Manageable (~5MB) but non-trivial. 4h timeframe at 2 days gives only 12 bars — TD Sequential needs 9–13 bars so 2 days is marginal. Consider bumping to 3 days for the 4h timeframe.
- **Binance 1s klines** endpoint has per-request cap (typically 1000 bars). For 2 days of 1s = 172,800 bars → 173 paged requests. ~1 request per second rate limit, so startup warmup is ~3 minutes. Could use 1m klines for warmup (only needs 2880 bars for 2 days) if 1s granularity isn't essential for indicators computed on ≥1m bars.
- Actually for 1m+ timeframes we only need 1m-resolution data, not 1s. Warmup should fetch Binance **1m klines** (2880 bars = 3 requests, seconds).
- **Recompute cost at live tick**: 7 timeframes × 3 indicators per snapshot is heavy. The aggregation from 1s should be cached — only re-aggregate the last open bar each tick.
