"""
Backtest the TD Sequential + RSI decide() rule against resolved Polymarket
BTC 5-minute market history.

For each resolved slot:
  1. Pull 90 minutes of Binance 1-minute klines ending at slot_ts.
  2. Resample to 1m / 3m / 5m OHLC.
  3. Compute signed TD setup count and RSI-14 at each timeframe.
  4. Call decide(snapshot) with TTE = 150 s (mid-slot assumption).
  5. Compare decision against actual outcome (Up → BUY_YES, Down → BUY_NO).

Weights baked into decide():  30s=4  1m=3  3m=2  5m=1  min_score=4
(30s is unavailable from 1m klines — effective max score = 6)

Usage
-----
    python scripts/backtest_td_rsi.py
    python scripts/backtest_td_rsi.py --csv data/btc_updown_5m.csv
    python scripts/backtest_td_rsi.py --no-cache   # force re-fetch from Binance
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from indicators.base import IndicatorConfig                      # noqa: E402
from indicators.td_sequential import TDSequentialIndicator       # noqa: E402
from src.strategies.td_rsi_decide import decide                  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── constants ─────────────────────────────────────────────────────────────────
COINBASE_URL  = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
LOOKBACK_SEC  = 90 * 60          # 90 minutes of 1m bars before slot open
CACHE_PATH    = Path(_ROOT) / "data" / "btc_1m_cache.csv"
CSV_PATH      = Path(_ROOT) / "data" / "btc_updown_5m.csv"
TTE_ASSUMED   = 150.0            # mid-slot: 150 s remaining
RSI_PERIOD    = 14

_tds = TDSequentialIndicator(IndicatorConfig("TDSeq", {}))


# ── Coinbase helpers ──────────────────────────────────────────────────────────
# Coinbase Exchange candles: granularity=60 (1-minute), max 300 candles/request.
# Response columns (oldest-first after sort): [timestamp, low, high, open, close, volume]

def _fetch_1m_chunk(start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch up to 300 1-minute candles from Coinbase for [start_ts, end_ts]."""
    from datetime import datetime, timezone
    start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = datetime.fromtimestamp(end_ts,   tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"granularity": 60, "start": start_iso, "end": end_iso}
    r = requests.get(COINBASE_URL, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close"]]


def fetch_all_1m(start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch all 1m candles in [start_ts, end_ts] (Unix seconds), paging 300 at a time."""
    CHUNK_SEC = 300 * 60   # 300 bars × 60 s each
    chunks: list = []
    cursor = start_ts
    print("  Fetching Coinbase 1m candles …", end="", flush=True)
    while cursor < end_ts:
        chunk_end = min(cursor + CHUNK_SEC, end_ts)
        chunk = _fetch_1m_chunk(cursor, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end
        if cursor < end_ts:
            time.sleep(0.35)   # ~3 req/s, well under Coinbase public limit
        print(".", end="", flush=True)
    print()
    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    combined = pd.concat(chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


# ── RSI ───────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI via EWM (adjust=False → recursive, same as most platforms)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ── TD signed count ───────────────────────────────────────────────────────────

def signed_td(ohlc: pd.DataFrame) -> int:
    """
    Return signed TD setup count for the latest bar:
      negative  →  buy  setup (bullish_setup_count, TD Down)
      positive  →  sell setup (bearish_setup_count, TD Up)
      0         →  flat / no active setup
    """
    if len(ohlc) < 6:
        return 0
    result = _tds.compute(ohlc)
    vals = result.values
    bull = int(vals["bullish_setup_count"][-1])
    bear = int(vals["bearish_setup_count"][-1])
    if bull > 0 and bull >= bear:
        return -bull
    if bear > 0:
        return bear
    return 0


# ── per-slot signal ───────────────────────────────────────────────────────────

def build_snapshot(ohlc_1m: pd.DataFrame) -> dict:
    """Resample to 1m/3m/5m, compute TD + RSI, return snapshot dict."""
    snap: dict = {"time_to_expiry_seconds": TTE_ASSUMED}

    def _resample(rule: str) -> pd.DataFrame:
        r = ohlc_1m.resample(rule)
        return pd.DataFrame({
            "open":  r["open"].first(),
            "high":  r["high"].max(),
            "low":   r["low"].min(),
            "close": r["close"].last(),
        }).dropna(how="all")

    for tf_label, ohlc in [
        ("1m", ohlc_1m),
        ("3m", _resample("3min")),
        ("5m", _resample("5min")),
    ]:
        if len(ohlc) < 6:
            continue
        td  = signed_td(ohlc)
        rsi = compute_rsi(ohlc["close"]).iloc[-1]
        snap[f"td_{tf_label}"]  = td
        snap[f"rsi_{tf_label}"] = float(rsi) if not np.isnan(rsi) else None

    return snap


# ── PnL simulation ────────────────────────────────────────────────────────────

# Assumptions for signal-only backtest:
#   - Entry at 0.50 (fair coin-flip price) since we can't know actual fill
#   - Binary payout: win → receive $1/share, lose → receive $0
#   - Fractional Kelly sizing
PNL_ENTRY_PRICE       = 0.50
PNL_INITIAL_BANKROLL  = 1000.0
PNL_KELLY_FRACTION    = 0.15    # conservative fractional Kelly
PNL_MAX_BET_FRACTION  = 0.03    # hard cap: 3% of bankroll per trade
SIGNAL_CONFIDENCE     = 0.60    # assumed p for chosen side when signal fires


def _kelly_size(p_model: float, price: float, bankroll: float,
                kelly_frac: float = PNL_KELLY_FRACTION,
                max_frac: float = PNL_MAX_BET_FRACTION) -> float:
    """Fractional Kelly bet size for a binary contract.

    f_kelly = (p - price) / (1 - price)
    Actual bet = kelly_frac * f_kelly * bankroll, capped at max_frac * bankroll.
    """
    if p_model <= price or price >= 1.0:
        return 0.0
    f_full = (p_model - price) / (1.0 - price)
    f = kelly_frac * f_full
    f = min(f, max_frac)
    return max(0.0, f * bankroll)


def _simulate_pnl(results: pd.DataFrame) -> pd.DataFrame:
    """Add pnl, bet_size, cum_bankroll columns to results DataFrame.

    Uses fractional Kelly sizing with a fixed assumed confidence when a
    signal fires (TD >= 8 + RSI confirm historically implies ~55-65% edge).
    Win = payout - cost, Lose = -cost.
    """
    SIGNAL_CONFIDENCE = 0.60

    # Sequential loop required because bankroll depends on prior trades
    pnl_arr = np.zeros(len(results))
    bet_arr = np.zeros(len(results))
    bank_arr = np.zeros(len(results))
    bankroll = PNL_INITIAL_BANKROLL

    for i, (traded, correct) in enumerate(
        zip(results["traded"].to_numpy(), results["correct"].to_numpy())
    ):
        if not traded:
            bank_arr[i] = bankroll
            continue

        bet = _kelly_size(SIGNAL_CONFIDENCE, PNL_ENTRY_PRICE, bankroll)
        if bet <= 0:
            bank_arr[i] = bankroll
            continue

        shares = bet / PNL_ENTRY_PRICE
        trade_pnl = shares * (1.0 - PNL_ENTRY_PRICE) if correct else -bet

        bankroll += trade_pnl
        pnl_arr[i] = trade_pnl
        bet_arr[i] = bet
        bank_arr[i] = bankroll

    results["pnl"] = pnl_arr
    results["bet_size"] = bet_arr
    results["cum_bankroll"] = bank_arr
    return results


def _brier_score(results: pd.DataFrame) -> float | None:
    """Brier score treating the decision as an implicit probability forecast.

    This is a rough proxy — the TD+RSI rule is binary, not probabilistic.
    Signal fire → p=0.60 for chosen side, no signal → p=0.50 (no view).
    """
    if results.empty:
        return None
    p_up = np.where(
        results["decision"].to_numpy() == "BUY_YES", SIGNAL_CONFIDENCE,
        np.where(results["decision"].to_numpy() == "BUY_NO", 1.0 - SIGNAL_CONFIDENCE, 0.50),
    )
    actual = (results["outcome"].to_numpy() == "Up").astype(float)
    return float(np.mean((p_up - actual) ** 2))


# ── main backtest ─────────────────────────────────────────────────────────────

def run(csv_path: Path, use_cache: bool) -> None:
    markets = pd.read_csv(csv_path)
    markets = markets.dropna(subset=["outcome"]).copy()
    markets["outcome"] = markets["outcome"].str.strip()
    markets = markets[markets["outcome"].isin(["Up", "Down"])].reset_index(drop=True)
    n_slots = len(markets)
    print(f"Loaded {n_slots} resolved slots from {csv_path.name}")

    # ── load or fetch BTC 1m data ─────────────────────────────────────────────
    min_ts = int(markets["slot_ts"].min()) - LOOKBACK_SEC
    max_ts = int(markets["slot_ts"].max()) + 300   # include last slot window

    if use_cache and CACHE_PATH.exists():
        print(f"  Loading cached 1m candles from {CACHE_PATH.name}")
        btc_1m_all = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        btc_1m_all.index = pd.to_datetime(btc_1m_all.index, utc=True)
    else:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        btc_1m_all = fetch_all_1m(min_ts, max_ts)
        btc_1m_all.to_csv(CACHE_PATH)
        print(f"  Cached {len(btc_1m_all)} 1m bars to {CACHE_PATH.name}")

    if btc_1m_all.empty:
        print("ERROR: no BTC price data fetched. Check Coinbase connectivity.")
        return

    # ── iterate slots ─────────────────────────────────────────────────────────
    rows = []
    for _, mkt in markets.iterrows():
        slot_ts  = int(mkt["slot_ts"])
        outcome  = mkt["outcome"]   # "Up" or "Down"

        t_start = pd.Timestamp(slot_ts - LOOKBACK_SEC, unit="s", tz="UTC")
        t_end   = pd.Timestamp(slot_ts,                unit="s", tz="UTC")

        ohlc_window = btc_1m_all.loc[
            (btc_1m_all.index >= t_start) & (btc_1m_all.index < t_end)
        ]

        snap     = build_snapshot(ohlc_window)
        decision = decide(snap)
        correct_decision = "BUY_YES" if outcome == "Up" else "BUY_NO"
        traded   = decision != "NO_TRADE"
        correct  = traded and (decision == correct_decision)

        rows.append({
            "slot_ts":           slot_ts,
            "outcome":           outcome,
            "decision":          decision,
            "correct_direction": correct_decision,
            "traded":            traded,
            "correct":           correct,
            "td_1m":             snap.get("td_1m"),
            "rsi_1m":            snap.get("rsi_1m"),
            "td_3m":             snap.get("td_3m"),
            "rsi_3m":            snap.get("rsi_3m"),
            "td_5m":             snap.get("td_5m"),
            "rsi_5m":            snap.get("rsi_5m"),
        })

    results = pd.DataFrame(rows)

    # ── simulate PnL ─────────────────────────────────────────────────────────
    results = _simulate_pnl(results)

    # ── print report ──────────────────────────────────────────────────────────
    traded   = results[results["traded"]]
    n_traded = len(traded)

    print()
    print("=" * 60)
    print("  TD Sequential + RSI  —  Backtest Results")
    print("=" * 60)
    print(f"  Total slots evaluated   : {n_slots}")
    print(f"  Slots with signal       : {n_traded}  ({n_traded/n_slots*100:.1f}%)")

    if n_traded == 0:
        print("  No signals generated — nothing to evaluate.")
        print("=" * 60)
        return

    by_side = traded.groupby("decision")
    for side in ("BUY_YES", "BUY_NO"):
        if side not in by_side.groups:
            continue
        grp  = by_side.get_group(side)
        hits = grp["correct"].sum()
        print(f"  {side:8s} signals       : {len(grp):4d}   "
              f"correct = {hits:4d}  ({hits/len(grp)*100:.1f}%)")

    overall_acc = traded["correct"].mean()
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Overall accuracy        : {traded['correct'].sum()}/{n_traded}"
          f"  = {overall_acc*100:.1f}%")

    # baseline: always pick the majority outcome
    majority_frac = max(
        (results["outcome"] == "Up").mean(),
        (results["outcome"] == "Down").mean(),
    )
    print(f"  Majority-class baseline : {majority_frac*100:.1f}%")
    print(f"  Up / Down split         : "
          f"{(results['outcome']=='Up').sum()} / "
          f"{(results['outcome']=='Down').sum()}")

    # ── PnL summary ──────────────────────────────────────────────────────────
    print()
    print("  PnL Simulation (entry @ 0.50, fractional Kelly)")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Initial bankroll        : ${PNL_INITIAL_BANKROLL:,.2f}")
    final = traded["cum_bankroll"].iloc[-1] if n_traded else PNL_INITIAL_BANKROLL
    total_ret = (final - PNL_INITIAL_BANKROLL) / PNL_INITIAL_BANKROLL
    print(f"  Final bankroll          : ${final:,.2f}")
    print(f"  Total return            : {total_ret:+.2%}")
    print(f"  Total PnL              : ${final - PNL_INITIAL_BANKROLL:+,.2f}")
    if n_traded:
        avg_pnl = traded["pnl"].mean()
        win_pnl = traded[traded["correct"]]["pnl"].mean() if traded["correct"].any() else 0
        loss_pnl = traded[~traded["correct"]]["pnl"].mean() if (~traded["correct"]).any() else 0
        print(f"  Avg PnL per trade       : ${avg_pnl:+,.2f}")
        print(f"  Avg win                 : ${win_pnl:+,.2f}")
        print(f"  Avg loss                : ${loss_pnl:+,.2f}")
        # Max drawdown
        peak = traded["cum_bankroll"].expanding().max()
        dd = (traded["cum_bankroll"] - peak) / peak
        print(f"  Max drawdown            : {dd.min():.2%}")
        # Brier score (p_up prediction quality)
        brier = _brier_score(results)
        if brier is not None:
            print(f"  Brier score (all slots) : {brier:.4f}  (0=perfect, 0.25=coin flip)")

    print("=" * 60)

    # ── outcome breakdown ─────────────────────────────────────────────────────
    print()
    print("Signal distribution")
    print(results["decision"].value_counts().to_string())

    # ── per-slot CSV export ───────────────────────────────────────────────────
    out_path = Path(_ROOT) / "data" / "backtest_td_rsi_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nPer-slot results saved → {out_path.name}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest TD + RSI decide() rule")
    parser.add_argument("--csv",      default=str(CSV_PATH), help="resolved market CSV")
    parser.add_argument("--no-cache", action="store_true",   help="re-fetch from Binance")
    args = parser.parse_args()

    run(Path(args.csv), use_cache=not args.no_cache)
