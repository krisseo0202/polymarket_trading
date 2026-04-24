"""Shared helpers for signed indicator features.

Used by both ``feature_builder`` (single-TF 5s block) and
``multi_tf_features`` (7-TF macro bank) to avoid two copies of the same
sign-collapse logic.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np


def signed_flag(buy: bool, sell: bool) -> float:
    """+1 if buy, -1 if sell, 0 otherwise. Mutually exclusive per bar."""
    return 1.0 if buy else (-1.0 if sell else 0.0)


def signed_bars_since(buy, sell) -> float:
    """Signed bars since last signal: + since last buy, − since last sell.

    Returns 0.0 when no signal has been seen — indistinguishable from a
    signal firing on the current bar, but the companion ``..._signal``
    column disambiguates.
    """
    if buy is None or sell is None:
        return 0.0
    buy_arr = np.asarray(buy, dtype=bool)
    sell_arr = np.asarray(sell, dtype=bool)
    n = len(buy_arr)
    if n == 0:
        return 0.0
    buy_idxs = np.flatnonzero(buy_arr)
    sell_idxs = np.flatnonzero(sell_arr)
    last_buy = int(buy_idxs[-1]) if len(buy_idxs) else -1
    last_sell = int(sell_idxs[-1]) if len(sell_idxs) else -1
    if last_buy < 0 and last_sell < 0:
        return 0.0
    if last_buy >= last_sell:
        return float(n - 1 - last_buy)
    return -float(n - 1 - last_sell)


def tdst_distance_and_side(
    td_vals: Mapping[str, np.ndarray], close_last: float,
) -> Tuple[float, float]:
    """Signed (close − active TDST level) / close, plus +1/0/−1 side flag.

    Sign convention: support active → distance typically positive (goes
    negative on a breakdown); resistance active → distance typically
    negative (goes positive on a breakout). Both polarities are
    informative, so the magnitude-and-sign pair is preferred over
    absolute-value.
    """
    support = td_vals.get("tdst_support")
    resistance = td_vals.get("tdst_resistance")
    side_arr = td_vals.get("tdst_side")
    if support is None or resistance is None or side_arr is None or close_last <= 0:
        return 0.0, 0.0
    if len(side_arr) == 0:
        return 0.0, 0.0
    side_last = int(side_arr[-1])
    if side_last > 0 and not np.isnan(support[-1]):
        return (close_last - float(support[-1])) / close_last, 1.0
    if side_last < 0 and not np.isnan(resistance[-1]):
        return (close_last - float(resistance[-1])) / close_last, -1.0
    return 0.0, 0.0
