"""Feature family map for ablation studies.

Groups every column in ``FEATURE_COLUMNS`` into one of 12 families. Used by
``scripts/feature_ablation.py`` to drop one family at a time and measure the
P&L / Brier delta vs the baseline model.

INVARIANT (validated at import time): every name in ``FEATURE_COLUMNS`` appears
in exactly one group. If you add a new feature to ``schema.py``, you must add
it to exactly one group here.

The multi-TF feature names are recomputed locally from ``MULTI_TF_TIMEFRAMES``
so this module stays importable even when the project's API client (which
``schema.py`` transitively pulls) is unavailable in the current env.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# Mirror of src.models.multi_tf_features.TIMEFRAMES — keep in sync.
MULTI_TF_TIMEFRAMES: Tuple[str, ...] = (
    "1m", "3m", "5m", "15m", "30m", "60m", "240m",
)


def _multi_tf_names() -> List[str]:
    names: List[str] = []
    for tf in MULTI_TF_TIMEFRAMES:
        names.append(f"rsi_{tf}")
        for suf in ("trend", "distance_pct", "signal", "bars_since_signal"):
            names.append(f"ut_{tf}_{suf}")
        for suf in (
            "setup", "cd", "signal_9", "signal_13", "display",
            "tdst_distance_pct", "tdst_side",
        ):
            names.append(f"td_{tf}_{suf}")
    return names


FEATURE_GROUPS: Dict[str, List[str]] = {
    "btc_momentum": [
        "btc_ret_5s", "btc_ret_15s", "btc_ret_30s", "btc_ret_60s",
        "btc_ret_180s", "btc_ret_600s", "btc_ret_1800s", "btc_ret_3600s",
        "btc_vol_15s", "btc_vol_30s", "btc_vol_60s", "vol_ratio_15_60",
    ],
    "btc_structure": [
        "btc_mid", "strike_price", "seconds_to_expiry",
        "moneyness", "distance_to_strike_bps",
    ],
    "btc_microstructure": [
        "volume_surge_ratio", "btc_vwap_deviation",
        "cumulative_volume_delta_60s",
    ],
    "ob_basic": [
        "yes_bid", "yes_ask", "yes_mid", "yes_spread", "yes_spread_pct",
        "yes_book_imbalance", "yes_ret_30s",
        "no_bid", "no_ask", "no_mid", "no_spread", "no_spread_pct",
        "no_book_imbalance", "no_ret_30s",
    ],
    "ob_depth": [
        "yes_microprice", "yes_depth_slope", "yes_depth_concentration",
        "no_microprice", "no_depth_slope", "no_depth_concentration",
        "yes_top3_bid_depth", "yes_top3_ask_depth", "yes_top3_imbalance",
        "yes_bid_slope", "yes_ask_slope",
        "no_top3_bid_depth", "no_top3_ask_depth", "no_top3_imbalance",
        "no_bid_slope", "no_ask_slope",
    ],
    "ob_coherence": [
        "mid_sum_residual", "mid_sum_residual_abs",
        "spread_asymmetry", "depth_asymmetry",
    ],
    "slot_path": [
        "slot_high_excursion_bps", "slot_low_excursion_bps",
        "slot_drift_bps", "slot_time_above_strike_pct", "slot_strike_crosses",
    ],
    "derived": [
        "moneyness_x_tte", "microprice_x_tte", "strike_crosses_x_vol",
        "fair_value_p_up", "yes_bid_residual", "microprice_residual",
    ],
    "indicators": [
        # FVG
        "active_bull_gap", "active_bear_gap", "latest_gap_distance_pct",
        # TD Sequential
        "td_setup", "td_cd", "td_signal_9", "td_signal_13", "td_display",
        "td_tdst_distance_pct", "td_tdst_side",
        # UT Bot
        "ut_bot_trend", "ut_bot_distance_pct", "ut_bot_signal",
        "ut_bot_bars_since_signal",
        # RSI
        "rsi_14",
    ],
    "calendar": [
        "hour_sin", "hour_cos", "is_weekend",
    ],
    "recent_outcomes": [
        "recent_up_rate_5", "recent_up_rate_10", "recent_up_rate_20",
    ],
    "multi_tf": _multi_tf_names() + ["ut_trend_disagreement"],
}


def group_of(feature_name: str) -> str:
    """Return the group name a feature belongs to. Raises KeyError if unmapped."""
    for group, members in FEATURE_GROUPS.items():
        if feature_name in members:
            return group
    raise KeyError(f"feature {feature_name!r} not assigned to any group")


def features_in_group(group_name: str) -> List[str]:
    """Return the feature list for a group. Raises KeyError if no such group."""
    return list(FEATURE_GROUPS[group_name])


def all_grouped_features() -> List[str]:
    """Flat list of every feature in any group."""
    out: List[str] = []
    for members in FEATURE_GROUPS.values():
        out.extend(members)
    return out


def validate_against(feature_columns: List[str]) -> None:
    """Verify every name in ``feature_columns`` is in exactly one group, and
    no group contains a name absent from ``feature_columns``.

    Call from a unit test or at startup with ``schema.FEATURE_COLUMNS``.
    """
    grouped = all_grouped_features()
    grouped_set = set(grouped)
    cols_set = set(feature_columns)

    if len(grouped) != len(grouped_set):
        seen: Dict[str, int] = {}
        for n in grouped:
            seen[n] = seen.get(n, 0) + 1
        dups = sorted([n for n, c in seen.items() if c > 1])
        raise ValueError(f"feature_group_map: duplicate assignments: {dups}")

    missing = sorted(cols_set - grouped_set)
    extra = sorted(grouped_set - cols_set)
    if missing or extra:
        raise ValueError(
            f"feature_group_map mismatch with FEATURE_COLUMNS — "
            f"missing from groups: {missing}; in groups but not in schema: {extra}"
        )
