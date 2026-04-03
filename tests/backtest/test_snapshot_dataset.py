import json

from src.backtest.snapshot_dataset import (
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
    assert set(["snapshot_ts", "label", "sim_yes_ask", "sim_no_ask", "feature_status"]).issubset(dataset.columns)
    assert int(dataset.iloc[0]["label"]) == 1
    assert float(dataset.iloc[1]["yes_ask"]) == 0.55
    assert float(dataset.iloc[1]["yes_ret_30s"]) > 0
    assert str(dataset.iloc[0]["feature_status"]).startswith("missing_")


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
