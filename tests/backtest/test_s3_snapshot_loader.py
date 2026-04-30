"""Tests for src.backtest.s3_snapshot_loader parsing + frame building.

No network access; feeds synthetic snapshot lines through the parsing layer.
"""

import json

import pandas as pd

from src.backtest.s3_snapshot_loader import (
    SnapshotBatch,
    build_features_frame,
    parse_book_dict,
    snapshot_to_order_books,
)


def _synthetic_snapshot(slot_ts: int, snapshot_ts: float, btc: float) -> dict:
    return {
        "slot_ts": slot_ts,
        "snapshot_ts": snapshot_ts,
        "up_token": "up-tok",
        "down_token": "down-tok",
        "strike": btc,  # strike = spot keeps moneyness == 0
        "strike_source": "chainlink",
        "btc_now": btc,
        "btc_source": "binance",
        "yes_bid": 0.52,
        "yes_ask": 0.54,
        "yes_mid": 0.53,
        "yes_spread": 0.02,
        "yes_imbalance": 0.0,
        "yes_bids": {"0.52": "100", "0.51": "50", "0.50": "50"},
        "yes_asks": {"0.54": "90", "0.55": "50", "0.56": "50"},
        "yes_bid_depth": 200.0,
        "yes_ask_depth": 190.0,
        "yes_n_levels": 6,
        "no_bid": 0.46,
        "no_ask": 0.48,
        "no_mid": 0.47,
        "no_spread": 0.02,
        "no_imbalance": 0.0,
        "no_bids": {"0.46": "80"},
        "no_asks": {"0.48": "80"},
        "no_bid_depth": 80.0,
        "no_ask_depth": 80.0,
        "no_n_levels": 2,
        "realized_vol_30s": 0.0001,
        "realized_vol_60s": 0.00015,
    }


def test_parse_book_dict_sorts_bids_descending_and_asks_ascending():
    bids = parse_book_dict({"0.50": "10", "0.52": "20", "0.51": "15"}, side="bids")
    assert [e.price for e in bids] == [0.52, 0.51, 0.50]
    assert [e.size for e in bids] == [20.0, 15.0, 10.0]

    asks = parse_book_dict({"0.56": "5", "0.54": "10", "0.55": "8"}, side="asks")
    assert [e.price for e in asks] == [0.54, 0.55, 0.56]


def test_parse_book_dict_drops_non_numeric_and_non_positive_entries():
    bids = parse_book_dict(
        {"0.50": "10", "garbage": "x", "0.45": "0", "-0.01": "5"},
        side="bids",
    )
    assert [e.price for e in bids] == [0.50]


def test_snapshot_to_order_books_full_depth():
    snap = _synthetic_snapshot(slot_ts=100, snapshot_ts=150.0, btc=75_000.0)
    yes_book, no_book = snapshot_to_order_books(snap)

    assert len(yes_book.bids) == 3
    assert len(yes_book.asks) == 3
    assert yes_book.bids[0].price == 0.52
    assert yes_book.asks[0].price == 0.54
    assert len(no_book.bids) == 1
    assert len(no_book.asks) == 1


def test_snapshot_batch_from_lines_splits_snapshots_and_outcomes():
    snap = _synthetic_snapshot(slot_ts=100, snapshot_ts=150.0, btc=75_000.0)
    outcome_sentinel = {"type": "outcome", "slot_ts": 100, "outcome": "Up"}
    junk = ""
    bad_json = "not json"

    lines = [json.dumps(snap), json.dumps(outcome_sentinel), junk, bad_json]
    batch = SnapshotBatch.from_lines(lines)

    assert len(batch.snapshots) == 1
    assert batch.outcomes == {100: "Up"}


def test_build_features_frame_emits_family_a_columns_with_real_values():
    snapshots = [
        _synthetic_snapshot(slot_ts=100, snapshot_ts=150.0, btc=75_000.0),
        _synthetic_snapshot(slot_ts=100, snapshot_ts=180.0, btc=75_010.0),
    ]
    outcomes = {100: "Up"}

    df = build_features_frame(snapshots, outcomes)

    assert not df.empty
    assert len(df) == 2
    assert set(df["label"]) == {1}

    for col in (
        "yes_microprice",
        "yes_depth_slope",
        "yes_depth_concentration",
        "no_microprice",
        "no_depth_concentration",
    ):
        assert col in df.columns, f"missing column {col}"

    # With full-depth bids/asks the depth features must be non-zero — this is
    # the parity fix vs the old synthetic-book backtest path that produced 0.
    assert (df["yes_microprice"] > 0).all()
    assert (df["yes_depth_concentration"] > 0).all()
    assert (df["yes_depth_concentration"] < 1.0).all()  # 3 levels, not concentrated at L1

    # With a single-level NO book, slope degenerates to 0.0 while concentration = 1.
    assert (df["no_depth_slope"] == 0.0).all()
    assert (df["no_depth_concentration"] == 1.0).all()


def test_build_features_frame_drops_slots_without_outcome():
    snapshots = [_synthetic_snapshot(slot_ts=100, snapshot_ts=150.0, btc=75_000.0)]
    df = build_features_frame(snapshots, outcomes={})
    assert df.empty


def test_build_features_frame_uses_btc_1s_when_available():
    snap = _synthetic_snapshot(slot_ts=1_000, snapshot_ts=1_200.0, btc=75_000.0)
    btc_df = pd.DataFrame(
        {
            "timestamp": [1_000.0 + i for i in range(201)],
            "open": [75_000.0 + i for i in range(201)],
            "high": [75_000.0 + i for i in range(201)],
            "low": [75_000.0 + i for i in range(201)],
            "close": [75_000.0 + i for i in range(201)],
            "volume": [1 for _ in range(201)],
        }
    )
    df = build_features_frame([snap], outcomes={1_000: "Down"}, btc_1s_df=btc_df)

    # Non-zero btc_ret_60s proves the 1s history (not the fallback) drove features.
    assert not df.empty
    assert df.iloc[0]["btc_ret_60s"] != 0.0
    assert df.iloc[0]["label"] == 0
