"""
Multi-timeframe BTC signal dashboard using FVG + TD Sequential.

Fetches BTC/USD OHLC from Binance or Coinbase and prints a per-timeframe
signal table with a simple decision and confidence score.

Usage:
    python scripts/signal_fvg_td.py
"""

import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests

API = "binance"
LIMIT = 100
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]

BINANCE_SYMBOL = "BTCUSDT"
COINBASE_SYMBOL = "BTC-USD"

from indicators.fvg import FVGIndicator
from indicators.td_sequential import TDSequentialIndicator
from indicators.base import IndicatorConfig, SignalType

COINBASE_GRANULARITY = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def fetch_binance(interval: str, limit: int) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": BINANCE_SYMBOL, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "num_trades",
            "taker_base_vol",
            "taker_quote_vol",
            "ignore",
        ],
    )
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close"]]


def fetch_coinbase(interval: str, limit: int) -> pd.DataFrame:
    granularity = COINBASE_GRANULARITY.get(interval)
    if granularity is None:
        raise ValueError(f"Coinbase does not support interval '{interval}'")

    url = f"https://api.exchange.coinbase.com/products/{COINBASE_SYMBOL}/candles"
    params = {"granularity": granularity, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close"]]


def fetch_ohlc(interval: str, limit: int) -> pd.DataFrame:
    if API == "binance":
        return fetch_binance(interval, limit)
    if API == "coinbase":
        return fetch_coinbase(interval, limit)
    raise ValueError(f"Unknown API: {API!r}. Use 'binance' or 'coinbase'.")


_fvg = FVGIndicator(IndicatorConfig("FVG", {"threshold_percent": 0.0, "auto": False}))
_tds = TDSequentialIndicator(IndicatorConfig("TDSeq", {}))


def _latest_fvg_signal(fvg_result) -> Optional[str]:
    for sig in reversed(fvg_result.signals):
        if sig.type == SignalType.BULL_FVG:
            return "BULL_FVG"
        if sig.type == SignalType.BEAR_FVG:
            return "BEAR_FVG"
    return None


def _has_signal(fvg_result, signal_type: SignalType) -> bool:
    return any(sig.type == signal_type for sig in fvg_result.signals)


def decide_from_fvg_td(fvg_result, tds_result, latest_close: float) -> Dict[str, Any]:
    vals = tds_result.values

    bull_setup = int(vals["bullish_setup_count"][-1])
    bear_setup = int(vals["bearish_setup_count"][-1])
    buy_cd = int(vals["buy_cd_count"][-1])
    sell_cd = int(vals["sell_cd_count"][-1])
    buy_9 = bool(vals["buy_9"][-1])
    sell_9 = bool(vals["sell_9"][-1])
    buy_13 = bool(vals["buy_13"][-1])
    sell_13 = bool(vals["sell_13"][-1])

    latest_gap = fvg_result.values.get("latest_gap")
    latest_fvg_signal = _latest_fvg_signal(fvg_result)
    bull_mitigated_now = _has_signal(fvg_result, SignalType.BULL_FVG_MITIGATED)
    bear_mitigated_now = _has_signal(fvg_result, SignalType.BEAR_FVG_MITIGATED)

    active_bull_gap = bool(
        latest_gap is not None
        and getattr(latest_gap, "is_bullish", False)
        and not getattr(latest_gap, "mitigated", True)
    )
    active_bear_gap = bool(
        latest_gap is not None
        and not getattr(latest_gap, "is_bullish", True)
        and not getattr(latest_gap, "mitigated", True)
    )

    gap_distance_pct = None
    if latest_gap is not None and latest_close > 0:
        gap_mid = (float(latest_gap.min_level) + float(latest_gap.max_level)) / 2.0
        gap_distance_pct = abs(latest_close - gap_mid) / latest_close

    buy_exhaustion = 1.0 if buy_13 else 0.7 if buy_9 else min(max(buy_cd / 13.0, bull_setup / 9.0), 0.45)
    sell_exhaustion = 1.0 if sell_13 else 0.7 if sell_9 else min(max(sell_cd / 13.0, bear_setup / 9.0), 0.45)

    buy_score = 0.0
    sell_score = 0.0

    if active_bull_gap:
        buy_score += 0.35
    elif latest_fvg_signal == "BULL_FVG":
        buy_score += 0.20

    if active_bear_gap:
        sell_score += 0.35
    elif latest_fvg_signal == "BEAR_FVG":
        sell_score += 0.20

    buy_score += 0.45 * buy_exhaustion
    sell_score += 0.45 * sell_exhaustion

    if gap_distance_pct is not None:
        if gap_distance_pct <= 0.0025:
            if active_bull_gap:
                buy_score += 0.15
            if active_bear_gap:
                sell_score += 0.15
        elif gap_distance_pct >= 0.01:
            if active_bull_gap:
                buy_score -= 0.10
            if active_bear_gap:
                sell_score -= 0.10

    if bull_mitigated_now:
        buy_score -= 0.20
    if bear_mitigated_now:
        sell_score -= 0.20

    if active_bull_gap or latest_fvg_signal == "BULL_FVG":
        sell_score -= 0.10
    if active_bear_gap or latest_fvg_signal == "BEAR_FVG":
        buy_score -= 0.10

    buy_score = max(0.0, min(1.0, buy_score))
    sell_score = max(0.0, min(1.0, sell_score))

    edge = abs(buy_score - sell_score)
    best_score = max(buy_score, sell_score)

    if best_score < 0.55 or edge < 0.15:
        decision = "NO TRADE"
        confidence = max(0.05, min(0.60, 0.20 + edge))
    elif buy_score > sell_score:
        decision = "BUY"
        confidence = max(0.0, min(1.0, 0.45 + 0.40 * buy_score + 0.20 * edge))
    else:
        decision = "SELL"
        confidence = max(0.0, min(1.0, 0.45 + 0.40 * sell_score + 0.20 * edge))

    return {
        "decision": decision,
        "confidence": round(confidence, 2),
        "buy_score": round(buy_score, 2),
        "sell_score": round(sell_score, 2),
        "bull_setup": bull_setup,
        "bear_setup": bear_setup,
        "buy_cd": buy_cd,
        "sell_cd": sell_cd,
        "buy_9": buy_9,
        "sell_9": sell_9,
        "buy_13": buy_13,
        "sell_13": sell_13,
        "latest_fvg_signal": latest_fvg_signal or "-",
    }


def analyze(ohlc: pd.DataFrame, tf: str) -> Dict[str, Any]:
    fvg_result = _fvg.compute(ohlc, timeframe=tf)
    tds_result = _tds.compute(ohlc, timeframe=tf)
    decision_ctx = decide_from_fvg_td(fvg_result, tds_result, float(ohlc["close"].iloc[-1]))

    bull_setup = decision_ctx["bull_setup"]
    bear_setup = decision_ctx["bear_setup"]
    buy_cd = decision_ctx["buy_cd"]
    sell_cd = decision_ctx["sell_cd"]
    buy_9 = decision_ctx["buy_9"]
    sell_9 = decision_ctx["sell_9"]
    buy_13 = decision_ctx["buy_13"]
    sell_13 = decision_ctx["sell_13"]

    if bull_setup > 0:
        td_setup = f"buy={bull_setup}" + ("*" if buy_9 else "")
    elif bear_setup > 0:
        td_setup = f"sell={bear_setup}" + ("*" if sell_9 else "")
    else:
        td_setup = "-"

    if buy_cd > 0:
        td_cd = f"bcd={buy_cd}" + ("*" if buy_13 else "")
    elif sell_cd > 0:
        td_cd = f"scd={sell_cd}" + ("*" if sell_13 else "")
    else:
        td_cd = "-"

    return {
        "fvg": decision_ctx["latest_fvg_signal"],
        "td_setup": td_setup,
        "td_cd": td_cd,
        "decision": decision_ctx["decision"],
        "confidence": decision_ctx["confidence"],
        "buy_score": decision_ctx["buy_score"],
        "sell_score": decision_ctx["sell_score"],
        "latest_close": float(ohlc["close"].iloc[-1]),
    }


def main() -> None:
    source = API.capitalize()
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    rows = []
    latest_close = None

    for tf in TIMEFRAMES:
        try:
            ohlc = fetch_ohlc(tf, LIMIT)
            result = analyze(ohlc, tf)
            if latest_close is None:
                latest_close = result["latest_close"]
            rows.append(
                (
                    tf,
                    result["fvg"],
                    result["td_setup"],
                    result["td_cd"],
                    result["decision"],
                    result["confidence"],
                )
            )
        except Exception as exc:
            rows.append((tf, "ERROR", str(exc)[:20], "-", "ERROR", "-"))

    print("\n=== BTC/USD Multi-Timeframe Signal Dashboard ===")
    if latest_close is None:
        latest_close = 0.0
    print(f"Source: {source}  |  {now}  |  Latest close: ${latest_close:,.2f}")
    print()

    header = f"{'TF':<6} | {'FVG':<10} | {'TD Setup':<11} | {'TD CD':<8} | {'Decision':<8} | Conf"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for tf, fvg, setup, cd, decision, conf in rows:
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        print(f"{tf:<6} | {fvg:<10} | {setup:<11} | {cd:<8} | {decision:<8} | {conf_str}")
    print()


if __name__ == "__main__":
    main()
