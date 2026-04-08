"""Shared schema and defaults for the BTC Up/Down XGBoost model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List


MODEL_NAME = "btc_updown_xgb"

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


def build_default_metadata() -> Dict[str, object]:
    """Return default model metadata used before training artifacts exist."""
    return {
        "model_name": MODEL_NAME,
        "model_version": f"{MODEL_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "feature_columns": FEATURE_COLUMNS,
        "thresholds": DEFAULT_THRESHOLDS,
        "calibration": {
            "bin_edges": [0.0, 1.0],
            "bin_values": [0.5],
        },
        "metrics": {},
    }

