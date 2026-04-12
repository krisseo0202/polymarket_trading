"""Human-readable one-liners for every feature column produced by
`train_and_compare_v5.build_dataset`. Used by the LLM experiment suggester
so it understands what each column means when picking subsets.

The LLM is told: `feature_subset` MUST be a subset of `AVAILABLE_FEATURES`
(defined below). Anything else is rejected by the parser.
"""

FEATURE_DOCS: dict[str, str] = {
    # time / context
    "time_to_expiry":        "Seconds remaining in the 5-minute slot (300..0).",

    # BTC momentum
    "ret_5s":                "BTC log-return over last 5s.",
    "ret_15s":               "BTC log-return over last 15s.",
    "ret_30s":               "BTC log-return over last 30s.",
    "ret_60s":               "BTC log-return over last 60s.",
    "ret_180s":              "BTC log-return over last 180s.",

    # BTC volatility
    "vol_15s":               "BTC realized vol (std of 1s log-returns) over 15s.",
    "vol_30s":               "BTC realized vol over 30s.",
    "vol_60s":               "BTC realized vol over 60s.",
    "vol_ratio_15_60":       "vol_15s / vol_60s — regime-change detector.",

    # BTC volume / flow
    "volume_surge_ratio":    "15s/60s average BTC tick-volume ratio.",
    "vwap_deviation":        "(BTC close − VWAP_60s) / close.",
    "cvd_60s":               "Cumulative volume delta over 60s (buy−sell)/total.",

    # BTC classic indicators
    "rsi_14":                "14-period BTC RSI (0..100).",
    "td_setup_net":          "TD Sequential net setup count (bearish − bullish).",

    # Polymarket orderbook — up-side
    "spread":                "Up-side ask − bid (quote width in probability).",
    "ob_imbalance":          "(up_bid_d3 − up_ask_d3)/(up_bid_d3 + up_ask_d3). [alias of depth_skew]",
    "ob_cross_imbalance":    "up_bid_d3 / (up_bid_d3 + down_bid_d3). [alias of imbalance]",

    # v5 microstructure
    "depth_ratio":           "log((up_bid_d3+1)/(up_ask_d3+1)) — signed depth tilt.",
    "depth_skew":            "(up_bid_d3 − up_ask_d3)/(up_bid_d3 + up_ask_d3).",
    "imbalance":             "up_bid_d3 / (up_bid_d3 + down_bid_d3) — cross-side lean.",
    "wall_flag":             "1 if current max side depth > 2× causal mean of all depths in slot.",
    "depth_change":          "(up_bid_d3+up_ask_d3)_now − (…)_5s_ago — depth-flow proxy.",
    "mid_return_5s":         "log(up_mid_now / up_mid_5s_ago) — market-mid momentum.",
    "acceleration":          "Δ(5s log-return of up_mid) — 2nd-derivative of mid.",
    "rolling_std_30s":       "Std of up_mid over last 30s of slot (intra-slot vol).",
    "range_30s":             "Max−min of up_mid over last 30s of slot.",
}

# Order matters here — anything present in the dataset is allowed. The
# run_experiment function will KeyError if the LLM picks something not in
# this list, which is the intended fail-loud behavior.
AVAILABLE_FEATURES: list = list(FEATURE_DOCS.keys())
