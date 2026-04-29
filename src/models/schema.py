"""Shared feature schema and default thresholds for BTC Up/Down models."""

from __future__ import annotations

from typing import Dict, List

from .multi_tf_features import multi_tf_feature_names


FEATURE_COLUMNS: List[str] = [
    "btc_mid",
    "btc_ret_5s",
    "btc_ret_15s",
    "btc_ret_30s",
    "btc_ret_60s",
    "btc_ret_180s",
    "btc_ret_600s",
    "btc_ret_1800s",
    "btc_ret_3600s",
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
    # Family A+ — top-N cumulative depth & side-split slopes
    "yes_top3_bid_depth",
    "yes_top3_ask_depth",
    "yes_top3_imbalance",
    "yes_bid_slope",
    "yes_ask_slope",
    "no_top3_bid_depth",
    "no_top3_ask_depth",
    "no_top3_imbalance",
    "no_bid_slope",
    "no_ask_slope",
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
    # Family D — interaction terms (LogReg cannot synthesize these internally)
    "moneyness_x_tte",
    "microprice_x_tte",
    "strike_crosses_x_vol",
    # Family E — fair-value residuals (closed-form Brownian proxy)
    "fair_value_p_up",
    "yes_bid_residual",
    "microprice_residual",
    # FVG indicator
    "active_bull_gap",
    "active_bear_gap",
    "latest_gap_distance_pct",
    # TD Sequential (5s bars) — signed: + bullish / − bearish / 0 inactive.
    # td_cd is only meaningful in classic mode; in pine_simple it still
    # computes but the chart ignores it and the model should too.
    "td_setup",
    "td_cd",
    "td_signal_9",
    "td_signal_13",
    "td_display",
    "td_tdst_distance_pct",
    "td_tdst_side",
    # UT Bot (5s bars) — signed.
    "ut_bot_trend",
    "ut_bot_distance_pct",
    "ut_bot_signal",
    "ut_bot_bars_since_signal",
    # Single-TF RSI-14 on ticks (warm in ~15 ticks; the multi-TF RSI bank
    # needs bar-level history and takes longer).
    "rsi_14",
    # Calendar regime — cyclical hour + weekend flag.
    "hour_sin",
    "hour_cos",
    "is_weekend",
    # Recent slot outcome history (Up rate over last N resolved slots).
    # Default to 0.5 (uninformative) until enough history is supplied.
    "recent_up_rate_5",
    "recent_up_rate_10",
    "recent_up_rate_20",
    *multi_tf_feature_names(),
    # Multi-TF cross-TF disagreement — must come after multi_tf_feature_names()
    # because it derives from those columns.
    "ut_trend_disagreement",
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
