"""Shared feature schema and default thresholds for BTC Up/Down models."""

from __future__ import annotations

from typing import Dict, List

# Timeframe tokens used by multi-TF indicator features. Source of truth:
# src/models/multi_tf_features.TIMEFRAMES. Kept in sync manually — if you add
# a timeframe there, append it here too (order preserves column ordering).
_MULTI_TF_TOKENS = ("1m", "3m", "5m", "15m", "30m", "60m", "240m")


def _multi_tf_columns() -> List[str]:
    cols: List[str] = []
    for tf in _MULTI_TF_TOKENS:
        cols.append(f"rsi_{tf}")
        for suffix in ("trend", "distance_pct", "buy_signal", "sell_signal"):
            cols.append(f"ut_{tf}_{suffix}")
        for suffix in (
            "bull_setup", "bear_setup", "buy_cd", "sell_cd",
            "buy_9", "sell_9", "buy_13", "sell_13",
        ):
            cols.append(f"td_{tf}_{suffix}")
    return cols


FEATURE_COLUMNS: List[str] = [
    "btc_mid",
    "btc_ret_5s",
    "btc_ret_15s",
    "btc_ret_30s",
    "btc_ret_60s",
    "btc_ret_180s",
    "btc_vol_15s",
    "btc_vol_30s",
    "btc_vol_60s",
    "vol_ratio_15_60",
    "strike_price",
    "seconds_to_expiry",
    "moneyness",
    "distance_to_strike_bps",
    # Volume features (tick count proxy; 0.0 when live feed lacks volume)
    "volume_surge_ratio",
    "btc_vwap_deviation",
    "cumulative_volume_delta_60s",
    # Orderbook features
    "yes_bid",
    "yes_ask",
    "yes_mid",
    "yes_spread",
    "yes_spread_pct",
    "yes_book_imbalance",
    "yes_ret_30s",
    "no_bid",
    "no_ask",
    "no_mid",
    "no_spread",
    "no_spread_pct",
    "no_book_imbalance",
    "no_ret_30s",
    # Family A — full-depth book features (from collected bids/asks dicts)
    "yes_microprice",
    "yes_depth_slope",
    "yes_depth_concentration",
    "no_microprice",
    "no_depth_slope",
    "no_depth_concentration",
    # Family B — YES/NO coherence (cross-book residuals and asymmetries)
    "mid_sum_residual",
    "mid_sum_residual_abs",
    "spread_asymmetry",
    "depth_asymmetry",
    # Family C — within-slot path (stateful across the 5-min window)
    "slot_high_excursion_bps",
    "slot_low_excursion_bps",
    "slot_drift_bps",
    "slot_time_above_strike_pct",
    "slot_strike_crosses",
    # FVG indicator
    "active_bull_gap",
    "active_bear_gap",
    "latest_gap_distance_pct",
    # TD Sequential indicator
    "bull_setup",
    "bear_setup",
    "buy_cd",
    "sell_cd",
    "buy_9",
    "sell_9",
    "buy_13",
    "sell_13",
    # UT Bot indicator
    "ut_bot_trend",
    "ut_bot_distance_pct",
    "ut_bot_buy_signal",
    "ut_bot_sell_signal",
    # Multi-timeframe macro features (7 TFs × 13 per TF = 91 cols).
    # Order: 1m, 3m, 5m, 15m, 30m, 60m, 240m; within each TF:
    # rsi_{tf}, ut_{tf}_{trend,distance_pct,buy_signal,sell_signal},
    # td_{tf}_{bull_setup,bear_setup,buy_cd,sell_cd,buy_9,sell_9,buy_13,sell_13}.
    *_multi_tf_columns(),
]

DEFAULT_FEATURE_VALUES: Dict[str, float] = {name: 0.0 for name in FEATURE_COLUMNS}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_edge": 0.03,
    "min_prob_yes": 0.54,
    "max_prob_yes_for_no": 0.46,
    "max_spread_pct": 0.06,
    "exit_edge": -0.01,
    "min_seconds_to_expiry": 20.0,
    "max_seconds_to_expiry": 240.0,
}
