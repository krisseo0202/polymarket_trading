import json

from src.backtest.snapshot_dataset import (
    build_btc_decision_dataset,
    build_snapshot_dataset,
    load_btc_prices,
    load_market_history,
    load_probability_ticks,
)


def test_build_snapshot_dataset_without_btc(tmp_path):
    market_path = tmp_path / "markets.csv"
    tick_path = tmp_path / "ticks.jsonl"

    market_path.write_text(
        "\n".join(
            [
                "slot_ts,slot_utc,question,outcome,strike_price",
                "1000,1970-01-01T00:16:40Z,Bitcoin Up or Down,Up,",
            ]
        ),
        encoding="utf-8",
    )
    tick_rows = [
        {"ts": 1010.0, "slot_ts": 1000, "yes_bid": 0.48, "yes_ask": 0.50, "no_bid": 0.50, "no_ask": 0.52, "yes_mid": 0.49, "no_mid": 0.51},
        {"ts": 1040.0, "slot_ts": 1000, "yes_bid": 0.53, "yes_ask": 0.55, "no_bid": 0.45, "no_ask": 0.47, "yes_mid": 0.54, "no_mid": 0.46},
    ]
    tick_path.write_text("\n".join(json.dumps(row) for row in tick_rows), encoding="utf-8")

    markets_df = load_market_history(str(market_path))
    prob_df = load_probability_ticks(str(tick_path))
    dataset = build_snapshot_dataset(markets_df, prob_df, btc_df=None)

    assert len(dataset) == 2
    assert set(["snapshot_ts", "label", "sim_yes_ask", "sim_no_ask", "feature_status",
                "edge_yes", "edge_no", "profitable_yes", "profitable_no"]).issubset(dataset.columns)
    assert int(dataset.iloc[0]["label"]) == 1
    assert float(dataset.iloc[1]["yes_ask"]) == 0.55
    assert float(dataset.iloc[1]["yes_ret_30s"]) > 0
    assert str(dataset.iloc[0]["feature_status"]).startswith("missing_")

    # Outcome=Up (label=1): edge_yes = 1 - ask > 0, edge_no = 0 - ask < 0
    row0 = dataset.iloc[0]
    assert float(row0["edge_yes"]) == 1.0 - float(row0["sim_yes_ask"])
    assert float(row0["edge_no"]) == 0.0 - float(row0["sim_no_ask"])
    assert int(row0["profitable_yes"]) == 1
    assert int(row0["profitable_no"]) == 0


def test_build_snapshot_dataset_with_local_btc_and_slot_open_strike(tmp_path):
    market_path = tmp_path / "markets.csv"
    tick_path = tmp_path / "ticks.jsonl"
    btc_path = tmp_path / "btc.csv"

    market_path.write_text(
        "\n".join(
            [
                "slot_ts,slot_utc,question,outcome,strike_price",
                "1000,1970-01-01T00:16:40Z,Bitcoin Up or Down,Down,",
            ]
        ),
        encoding="utf-8",
    )
    tick_rows = [
        {"ts": 1010.0, "slot_ts": 1000, "yes_bid": 0.51, "yes_ask": 0.53, "no_bid": 0.47, "no_ask": 0.49, "yes_mid": 0.52, "no_mid": 0.48},
        {"ts": 1030.0, "slot_ts": 1000, "yes_bid": 0.54, "yes_ask": 0.56, "no_bid": 0.44, "no_ask": 0.46, "yes_mid": 0.55, "no_mid": 0.45},
    ]
    tick_path.write_text("\n".join(json.dumps(row) for row in tick_rows), encoding="utf-8")
    btc_path.write_text(
        "\n".join(
            [
                "ts,price",
                "995,100000",
                "1000,100010",
                "1005,100020",
                "1010,100040",
                "1020,100030",
                "1030,100060",
            ]
        ),
        encoding="utf-8",
    )

    markets_df = load_market_history(str(market_path))
    prob_df = load_probability_ticks(str(tick_path))
    btc_df = load_btc_prices(str(btc_path))
    dataset = build_snapshot_dataset(markets_df, prob_df, btc_df=btc_df, btc_window_seconds=60)

    assert len(dataset) == 2
    assert float(dataset.iloc[0]["strike_price"]) == 100010.0
    assert float(dataset.iloc[1]["btc_mid"]) == 100060.0
    assert float(dataset.iloc[1]["seconds_to_expiry"]) == 270.0
    assert str(dataset.iloc[1]["feature_status"]).startswith("ready") or str(dataset.iloc[1]["feature_status"]) == "insufficient_btc_history"

    # Outcome=Down (label=0): edge_yes = 0 - ask < 0, edge_no = 1 - ask > 0
    row0 = dataset.iloc[0]
    assert float(row0["edge_yes"]) == 0.0 - float(row0["sim_yes_ask"])
    assert float(row0["edge_no"]) == 1.0 - float(row0["sim_no_ask"])
    assert int(row0["profitable_yes"]) == 0
    assert int(row0["profitable_no"]) == 1


def test_build_snapshot_dataset_with_cost_per_share(tmp_path):
    """Verify cost_per_share parameter reduces edge and can flip profitability."""
    market_path = tmp_path / "markets.csv"
    tick_path = tmp_path / "ticks.jsonl"

    market_path.write_text(
        "\n".join(
            [
                "slot_ts,slot_utc,question,outcome,strike_price",
                "1000,1970-01-01T00:16:40Z,Bitcoin Up or Down,Up,",
            ]
        ),
        encoding="utf-8",
    )
    tick_rows = [
        {"ts": 1010.0, "slot_ts": 1000, "yes_bid": 0.48, "yes_ask": 0.50, "no_bid": 0.50, "no_ask": 0.52, "yes_mid": 0.49, "no_mid": 0.51},
    ]
    tick_path.write_text(json.dumps(tick_rows[0]), encoding="utf-8")

    markets_df = load_market_history(str(market_path))
    prob_df = load_probability_ticks(str(tick_path))

    # Without cost: edge_yes = 1.0 - 0.50 = 0.50 (profitable)
    ds_free = build_snapshot_dataset(markets_df, prob_df, cost_per_share=0.0)
    assert float(ds_free.iloc[0]["edge_yes"]) == 0.50
    assert int(ds_free.iloc[0]["profitable_yes"]) == 1

    # With high cost: edge_yes = 1.0 - 0.50 - 0.60 = -0.10 (not profitable)
    ds_costly = build_snapshot_dataset(markets_df, prob_df, cost_per_share=0.60)
    assert abs(float(ds_costly.iloc[0]["edge_yes"]) - (-0.10)) < 1e-9
    assert int(ds_costly.iloc[0]["profitable_yes"]) == 0


def test_build_btc_decision_dataset_basic(tmp_path):
    """Step 3-5: expand contracts into decision rows with BTC features."""
    market_path = tmp_path / "markets.csv"
    btc_path = tmp_path / "btc.csv"

    market_path.write_text(
        "\n".join([
            "slot_ts,slot_utc,question,outcome,strike_price",
            "1000,1970-01-01T00:16:40Z,Bitcoin Up or Down,Up,",
        ]),
        encoding="utf-8",
    )

    # 1-second BTC prices: slot starts at 1000, expires at 1300
    btc_rows = ["ts,price"]
    price = 50000.0
    for t in range(900, 1301):
        price += (t % 3 - 1) * 0.5  # small deterministic walk
        btc_rows.append(f"{t},{price:.2f}")
    btc_path.write_text("\n".join(btc_rows), encoding="utf-8")

    markets_df = load_market_history(str(market_path))
    btc_df = load_btc_prices(str(btc_path))
    ds = build_btc_decision_dataset(markets_df, btc_df, row_interval_sec=15)

    # Step 3 checks: 0, 15, 30, ... 300 → 21 rows
    assert len(ds) == 21
    assert list(ds.columns) == [
        "contract_id", "timestamp", "expiry_ts", "time_to_expiry_sec",
        "reference_price", "target_up",
        "btc_spot", "ret_15s", "ret_30s", "ret_60s",
        "rolling_vol_60s", "ma_12", "rsi_14",
        "dist_to_strike", "ma_12_gap",
    ]
    assert int(ds.iloc[0]["contract_id"]) == 1000
    assert int(ds.iloc[0]["timestamp"]) == 1000
    assert int(ds.iloc[0]["expiry_ts"]) == 1300
    assert int(ds.iloc[0]["time_to_expiry_sec"]) == 300
    assert int(ds.iloc[-1]["time_to_expiry_sec"]) == 0
    assert all(ds["target_up"] == 1)

    # Step 4 checks: btc_spot is positive, returns are finite
    assert all(ds["btc_spot"] > 0)
    assert all(ds["ret_15s"].apply(lambda v: abs(v) < 1))
    assert all(ds["rolling_vol_60s"] >= 0)
    assert all(ds["rsi_14"].between(0, 100))
    assert all(ds["ma_12"] > 0)

    # Step 5 checks: normalised features
    row = ds.iloc[10]
    expected_dist = (float(row["btc_spot"]) - float(row["reference_price"])) / float(row["reference_price"])
    assert abs(float(row["dist_to_strike"]) - expected_dist) < 1e-9
    expected_gap = (float(row["btc_spot"]) - float(row["ma_12"])) / float(row["ma_12"])
    assert abs(float(row["ma_12_gap"]) - expected_gap) < 1e-9


def test_build_btc_decision_dataset_down_outcome(tmp_path):
    """Verify target_up=0 for Down outcomes and strike_price fallback."""
    market_path = tmp_path / "markets.csv"
    btc_path = tmp_path / "btc.csv"

    market_path.write_text(
        "\n".join([
            "slot_ts,slot_utc,question,outcome,strike_price",
            "2000,1970-01-01T00:33:20Z,Bitcoin Up or Down,Down,50005.0",
        ]),
        encoding="utf-8",
    )

    btc_rows = ["ts,price"]
    for t in range(1900, 2301):
        btc_rows.append(f"{t},50000.0")
    btc_path.write_text("\n".join(btc_rows), encoding="utf-8")

    markets_df = load_market_history(str(market_path))
    btc_df = load_btc_prices(str(btc_path))
    ds = build_btc_decision_dataset(markets_df, btc_df, row_interval_sec=60)

    # 0, 60, 120, 180, 240, 300 → 6 rows
    assert len(ds) == 6
    assert all(ds["target_up"] == 0)
    assert float(ds.iloc[0]["reference_price"]) == 50005.0
