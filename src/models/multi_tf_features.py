"""Multi-timeframe technical indicator features for BTC.

For each timeframe in {1m, 3m, 5m, 15m, 30m, 60m, 4h} we compute:
  * RSI(14)              → 1 feature (rsi_{tf})
  * UT Bot (ATR 10)      → 4 features: trend, distance_pct, signal,
                            bars_since_signal (signed)
  * TD Sequential        → 7 features: setup, cd, signal_9, signal_13,
                            display, tdst_distance_pct, tdst_side
                            (all signed: + bullish / − bearish)

Total: 7 TFs × 12 features = 84 output columns.

Missing-data policy
-------------------
Each timeframe needs a minimum number of warm bars before its indicators are
meaningful. When the supplied history is too short for a given TF, that TF's
features stay at 0.0 (the schema default) and a per-TF status flag is emitted
in the returned ``status`` dict.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from indicators.base import IndicatorConfig
from indicators.rsi import RSIIndicator
from indicators.td_sequential import TDSequentialIndicator
from indicators.ut_bot import UTBotIndicator
from src.utils.ohlc_aggregator import aggregate_to_ohlc

from ._indicator_utils import signed_bars_since, signed_flag, tdst_distance_and_side


# (interval_seconds, short_token). Short token is the suffix used in feature
# names, e.g. rsi_5m, ut_240m_trend.
TIMEFRAMES: Tuple[Tuple[int, str], ...] = (
    (60, "1m"),
    (180, "3m"),
    (300, "5m"),
    (900, "15m"),
    (1800, "30m"),
    (3600, "60m"),
    (14400, "240m"),
)

# Minimum bars needed for each indicator family to warm up.
# RSI(14) needs 15 bars. UT Bot ATR(10) needs ~11 bars. TD Sequential needs
# at least 9 bars for a setup; 13 more for a countdown. We use 25 as a
# conservative floor that lets all three produce a non-trivial output.
MIN_BARS_PER_TF = 25

_RSI = RSIIndicator(IndicatorConfig(name="RSI", params={"period": 14}))
_UTB = UTBotIndicator(IndicatorConfig(name="UTBot", params={"atr_period": 10, "key_value": 1.0}))
_TDS = TDSequentialIndicator(IndicatorConfig(name="TDSeq", params={}))


def multi_tf_feature_names() -> List[str]:
    """Return the full list of feature column names — matches schema order."""
    names: List[str] = []
    for _, tf in TIMEFRAMES:
        names.append(f"rsi_{tf}")
        for suffix in ("trend", "distance_pct", "signal", "bars_since_signal"):
            names.append(f"ut_{tf}_{suffix}")
        for suffix in (
            "setup", "cd", "signal_9", "signal_13", "display",
            "tdst_distance_pct", "tdst_side",
        ):
            names.append(f"td_{tf}_{suffix}")
    return names


def compute_multi_tf_features(
    btc_prices: Sequence,
    now_ts: float,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Compute the multi-timeframe features at ``now_ts``.

    Args:
        btc_prices: sequence of ``(ts, price[, volume])`` tuples, ascending by
            timestamp. Should include at least 2 days of history for the 4h
            timeframe to warm up properly.
        now_ts: reference time (UTC seconds). Used to trim bars strictly
            after this moment so features reflect state "as of now".

    Returns:
        (features, status) where:
          * ``features`` is a dict of column → value. Missing values are 0.0.
          * ``status`` is a dict of ``{tf: "ready" | "insufficient_bars_N"}``.
    """
    features: Dict[str, float] = {name: 0.0 for name in multi_tf_feature_names()}
    status: Dict[str, str] = {}
    if not btc_prices:
        for _, tf in TIMEFRAMES:
            status[tf] = "no_history"
        return features, status

    for interval_s, tf in TIMEFRAMES:
        ohlc = aggregate_to_ohlc(btc_prices, interval_s=interval_s)
        if not ohlc.empty:
            bar_floor = int(now_ts // interval_s) * interval_s
            bar_floor_dt = pd.Timestamp(bar_floor, unit="s", tz="UTC")
            ohlc = ohlc[ohlc.index <= bar_floor_dt]

        if len(ohlc) < MIN_BARS_PER_TF:
            status[tf] = f"insufficient_bars_{len(ohlc)}"
            continue

        _populate_tf_features(features, ohlc, tf)
        status[tf] = "ready"

    return features, status


def _populate_tf_features(
    features: Dict[str, float],
    ohlc: pd.DataFrame,
    tf: str,
) -> None:
    close_last = float(ohlc["close"].iloc[-1])

    rsi_vals = _RSI.compute(ohlc, timeframe=tf).values["rsi"]
    rsi_last = _last_finite(rsi_vals)
    if rsi_last is not None:
        features[f"rsi_{tf}"] = rsi_last

    ut_result = _UTB.compute(ohlc, timeframe=tf).values
    trail_last = float(ut_result["trail"][-1])
    features[f"ut_{tf}_trend"] = 1.0 if close_last > trail_last else -1.0
    features[f"ut_{tf}_distance_pct"] = (
        (close_last - trail_last) / close_last if close_last > 0 else 0.0
    )
    features[f"ut_{tf}_signal"] = signed_flag(
        bool(ut_result["buy"][-1]), bool(ut_result["sell"][-1]),
    )
    features[f"ut_{tf}_bars_since_signal"] = signed_bars_since(
        ut_result["buy"], ut_result["sell"],
    )

    td_vals = _TDS.compute(ohlc, timeframe=tf).values
    # td_dn carries the bullish chart digit, td_up the bearish — signed subtraction
    # keeps the + bullish / − bearish convention.
    features[f"td_{tf}_setup"] = (
        float(td_vals["bullish_setup_count"][-1]) - float(td_vals["bearish_setup_count"][-1])
    )
    features[f"td_{tf}_cd"] = (
        float(td_vals["buy_cd_count"][-1]) - float(td_vals["sell_cd_count"][-1])
    )
    features[f"td_{tf}_signal_9"] = signed_flag(
        bool(td_vals["buy_9"][-1]), bool(td_vals["sell_9"][-1]),
    )
    features[f"td_{tf}_signal_13"] = signed_flag(
        bool(td_vals["buy_13"][-1]), bool(td_vals["sell_13"][-1]),
    )
    features[f"td_{tf}_display"] = float(td_vals["td_dn"][-1]) - float(td_vals["td_up"][-1])

    tdst_dist, tdst_side = tdst_distance_and_side(td_vals, close_last)
    features[f"td_{tf}_tdst_distance_pct"] = tdst_dist
    features[f"td_{tf}_tdst_side"] = tdst_side


def _last_finite(arr: np.ndarray) -> float | None:
    if len(arr) == 0:
        return None
    for v in reversed(arr):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return None
