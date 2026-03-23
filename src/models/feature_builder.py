"""Feature engineering for the BTC Up/Down XGBoost model."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from indicators.base import IndicatorConfig
from indicators.fvg import FVGIndicator
from indicators.td_sequential import TDSequentialIndicator
from src.api.types import OrderBook
from .schema import DEFAULT_FEATURE_VALUES, FEATURE_COLUMNS


_STRIKE_RE = re.compile(r"\$([0-9,]+(?:\.[0-9]+)?)")

_FVG = FVGIndicator(IndicatorConfig("FVG", {"threshold_percent": 0.0, "auto": False}))
_TDS = TDSequentialIndicator(IndicatorConfig("TDSeq", {}))


@dataclass
class FeatureBuildResult:
    """Structured feature-building result."""

    features: Dict[str, float]
    ready: bool
    status: str


def parse_strike_price(question: str) -> Optional[float]:
    """Extract the BTC strike level from a market question."""
    match = _STRIKE_RE.search(question or "")
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def build_live_features(snapshot: Mapping[str, object]) -> FeatureBuildResult:
    """Build a fixed live feature vector from BTC feed and order books."""
    features = dict(DEFAULT_FEATURE_VALUES)

    btc_prices = list(snapshot.get("btc_prices") or [])
    if len(btc_prices) < 2:
        return FeatureBuildResult(features=features, ready=False, status="insufficient_btc_history")

    yes_book = snapshot.get("yes_book")
    no_book = snapshot.get("no_book")
    if not isinstance(yes_book, OrderBook) or not isinstance(no_book, OrderBook):
        return FeatureBuildResult(features=features, ready=False, status="missing_order_books")

    strike_price = snapshot.get("strike_price")
    question = str(snapshot.get("question") or "")
    if strike_price is None:
        strike_price = parse_strike_price(question)
    if strike_price is None or float(strike_price) <= 0:
        return FeatureBuildResult(features=features, ready=False, status="missing_strike")
    strike_price = float(strike_price)

    now_ts = float(snapshot.get("now_ts") or btc_prices[-1][0])
    slot_expiry_ts = snapshot.get("slot_expiry_ts")
    if slot_expiry_ts is None:
        return FeatureBuildResult(features=features, ready=False, status="missing_expiry")
    seconds_to_expiry = max(0.0, float(slot_expiry_ts) - now_ts)

    btc_mid = float(btc_prices[-1][1])
    features["btc_mid"] = btc_mid
    features["strike_price"] = strike_price
    features["seconds_to_expiry"] = seconds_to_expiry
    features["moneyness"] = (btc_mid - strike_price) / strike_price
    features["distance_to_strike_bps"] = (btc_mid - strike_price) / strike_price * 10_000.0

    features["btc_ret_15s"] = _safe_return(btc_prices, now_ts, 15)
    features["btc_ret_30s"] = _safe_return(btc_prices, now_ts, 30)
    features["btc_ret_60s"] = _safe_return(btc_prices, now_ts, 60)
    features["btc_vol_30s"] = _realized_vol(btc_prices, now_ts, 30)
    features["btc_vol_60s"] = _realized_vol(btc_prices, now_ts, 60)

    yes_history = list(snapshot.get("yes_history") or [])
    no_history = list(snapshot.get("no_history") or [])
    _add_book_features(features, yes_book, yes_history, now_ts, prefix="yes")
    _add_book_features(features, no_book, no_history, now_ts, prefix="no")

    indicator_status = _add_indicator_features(features, btc_prices, btc_mid)
    status = "ready" if indicator_status == "ready" else indicator_status
    return FeatureBuildResult(features=features, ready=True, status=status)


def _safe_return(prices: Sequence[Tuple[float, float]], now_ts: float, lookback_s: int) -> float:
    current = _latest_value(prices)
    previous = _lookback_value(prices, now_ts, lookback_s)
    if current is None or previous is None or previous <= 0:
        return 0.0
    return (current - previous) / previous


def _realized_vol(prices: Sequence[Tuple[float, float]], now_ts: float, window_s: int) -> float:
    cutoff = now_ts - window_s
    window = [float(price) for ts, price in prices if float(ts) >= cutoff]
    if len(window) < 3:
        return 0.0
    arr = np.asarray(window, dtype=float)
    returns = np.diff(np.log(arr))
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns))


def _latest_value(prices: Sequence[Tuple[float, float]]) -> Optional[float]:
    if not prices:
        return None
    return float(prices[-1][1])


def _lookback_value(
    prices: Sequence[Tuple[float, float]], now_ts: float, lookback_s: int
) -> Optional[float]:
    target = now_ts - lookback_s
    candidate = None
    for ts, price in prices:
        if float(ts) <= target:
            candidate = float(price)
        else:
            break
    return candidate


def _add_book_features(
    features: Dict[str, float],
    book: OrderBook,
    history: Sequence[Tuple[float, float]],
    now_ts: float,
    prefix: str,
) -> None:
    bid = float(book.bids[0].price) if book.bids else 0.0
    ask = float(book.asks[0].price) if book.asks else 0.0
    bid_sz = float(book.bids[0].size) if book.bids else 0.0
    ask_sz = float(book.asks[0].size) if book.asks else 0.0
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bid or ask
    spread = max(0.0, ask - bid) if bid > 0 and ask > 0 else 0.0
    spread_pct = spread / mid if mid > 0 else 0.0
    denom = bid_sz + ask_sz
    imbalance = (bid_sz - ask_sz) / denom if denom > 0 else 0.0

    features[f"{prefix}_bid"] = bid
    features[f"{prefix}_ask"] = ask
    features[f"{prefix}_mid"] = mid
    features[f"{prefix}_spread"] = spread
    features[f"{prefix}_spread_pct"] = spread_pct
    features[f"{prefix}_book_imbalance"] = imbalance
    features[f"{prefix}_ret_30s"] = _safe_return(history, now_ts, 30)


def _add_indicator_features(
    features: Dict[str, float],
    btc_prices: Sequence[Tuple[float, float]],
    btc_mid: float,
) -> str:
    ohlc = _prices_to_ohlc(btc_prices, bar_seconds=5)
    if len(ohlc) < 20:
        return "ready_indicator_warmup"

    try:
        fvg_result = _FVG.compute(ohlc, timeframe="5s")
        tds_result = _TDS.compute(ohlc, timeframe="5s")
    except Exception:
        return "ready_indicator_error"

    latest_gap = fvg_result.values.get("latest_gap")
    features["active_bull_gap"] = float(
        bool(latest_gap is not None and getattr(latest_gap, "is_bullish", False) and not getattr(latest_gap, "mitigated", True))
    )
    features["active_bear_gap"] = float(
        bool(latest_gap is not None and not getattr(latest_gap, "is_bullish", True) and not getattr(latest_gap, "mitigated", True))
    )

    if latest_gap is not None and btc_mid > 0:
        gap_mid = (float(latest_gap.min_level) + float(latest_gap.max_level)) / 2.0
        features["latest_gap_distance_pct"] = abs(btc_mid - gap_mid) / btc_mid

    vals = tds_result.values
    features["bull_setup"] = float(_last_scalar(vals.get("bullish_setup_count")))
    features["bear_setup"] = float(_last_scalar(vals.get("bearish_setup_count")))
    features["buy_cd"] = float(_last_scalar(vals.get("buy_cd_count")))
    features["sell_cd"] = float(_last_scalar(vals.get("sell_cd_count")))
    features["buy_9"] = float(bool(_last_scalar(vals.get("buy_9"))))
    features["sell_9"] = float(bool(_last_scalar(vals.get("sell_9"))))
    features["buy_13"] = float(bool(_last_scalar(vals.get("buy_13"))))
    features["sell_13"] = float(bool(_last_scalar(vals.get("sell_13"))))
    return "ready"


def _last_scalar(values: object) -> float:
    if values is None:
        return 0.0
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        if len(values) == 0:
            return 0.0
        value = values[-1]
    else:
        value = values
    if isinstance(value, (np.bool_, bool)):
        return 1.0 if bool(value) else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _prices_to_ohlc(
    prices: Sequence[Tuple[float, float]], bar_seconds: int
) -> pd.DataFrame:
    if not prices:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["bucket"] = df["timestamp"].dt.floor(f"{bar_seconds}s")

    grouped = df.groupby("bucket")["price"]
    ohlc = grouped.agg(open="first", high="max", low="min", close="last")
    return ohlc[["open", "high", "low", "close"]]


def coerce_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an input frame into the fixed feature schema."""
    out = pd.DataFrame(index=df.index)
    for name in FEATURE_COLUMNS:
        if name in df.columns:
            out[name] = pd.to_numeric(df[name], errors="coerce").fillna(DEFAULT_FEATURE_VALUES[name])
        else:
            out[name] = DEFAULT_FEATURE_VALUES[name]
    return out

