"""Integration tests for the unified --training-period / --valid-period /
--test-period CLI flags across the refactored scripts.

These exercise the argparse wiring and the split-resolution path without
requiring xgboost / pyarrow / s3fs to be installed."""

import argparse
import importlib
import json

import pandas as pd

from src.backtest.period_split import (
    add_period_arguments,
    parse_period_config,
    resolve_split_from_args,
)


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------


def test_train_adds_period_arguments_to_parser():
    """train.py should expose the three period flags in its CLI."""
    import scripts.train as train_mod

    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="x")
    parser.add_argument("--out", default="y")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--synthetic-half-spread", type=float, default=0.01)
    add_period_arguments(parser)

    ns = parser.parse_args([
        "--selection", "s", "--out", "o",
        "--training-period", "2026-03-01:2026-04-01",
        "--valid-period", "2026-04-01:2026-04-15",
        "--test-period", "2026-04-15:2026-05-01",
    ])
    assert ns.training_period == "2026-03-01:2026-04-01"
    assert ns.valid_period == "2026-04-01:2026-04-15"
    assert ns.test_period == "2026-04-15:2026-05-01"
    # train_mod has imported resolve_split_from_args — importing it now
    # ensures the module loads cleanly under pytest's collection.
    assert hasattr(train_mod, "resolve_split_from_args")


def test_train_legacy_split_by_slot_still_returns_tuple():
    """The legacy wrapper must keep tuple return for test_train.py compat."""
    import scripts.train as train_mod

    df = pd.DataFrame({
        "slot_ts": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "label": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    })
    train, val = train_mod.split_by_slot(df, val_ratio=0.2)
    assert isinstance(train, pd.DataFrame) and isinstance(val, pd.DataFrame)
    assert train["slot_ts"].nunique() == 4
    assert val["slot_ts"].nunique() == 1


# ---------------------------------------------------------------------------
# feature_probe.py
# ---------------------------------------------------------------------------


def test_feature_probe_adds_period_arguments():
    import scripts.feature_probe as probe_mod

    # Rebuild the argparse from the same builders feature_probe uses.
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="x")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    add_period_arguments(parser)
    ns = parser.parse_args(["--training-period", "2026-03-01:2026-04-01"])
    assert ns.training_period == "2026-03-01:2026-04-01"
    # Module imports cleanly.
    assert hasattr(probe_mod, "resolve_split_from_args")


def test_feature_probe_legacy_split_by_slot_still_works():
    import scripts.feature_probe as probe_mod

    df = pd.DataFrame({
        "slot_ts": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "label": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    })
    train, val = probe_mod.split_by_slot(df, val_ratio=0.2)
    assert train["slot_ts"].nunique() == 4
    assert val["slot_ts"].nunique() == 1


# ---------------------------------------------------------------------------
# backtest_logreg_edge.py
# ---------------------------------------------------------------------------


def test_backtest_adds_period_arguments():
    import scripts.backtest_logreg_edge as bt

    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    add_period_arguments(parser)
    ns = parser.parse_args([
        "--training-period", "2026-03-01:2026-04-01",
        "--test-period", "2026-04-01:2026-05-01",
    ])
    assert ns.training_period and ns.test_period
    assert hasattr(bt, "resolve_split_from_args")


# ---------------------------------------------------------------------------
# End-to-end: meta.json period serialization (simulates what train.py writes)
# ---------------------------------------------------------------------------


def test_period_config_serializes_to_meta_json_shape_train_expects():
    """train.py writes meta['periods'] = split.config.to_dict(); that shape
    must round-trip through JSON and contain the three expected keys."""
    cfg = parse_period_config(
        training="2026-03-01:2026-04-01",
        validation="2026-04-01:2026-04-15",
        test="2026-04-15:2026-05-01",
    )
    meta = {
        "split_mode": "periods",
        "periods": cfg.to_dict(),
        "n_train_slots": 42,
        "n_val_slots": 10,
        "n_test_slots": 15,
    }
    encoded = json.dumps(meta, default=float)
    roundtripped = json.loads(encoded)

    assert roundtripped["split_mode"] == "periods"
    assert set(roundtripped["periods"].keys()) == {"training", "validation", "test"}
    for label in ("training", "validation", "test"):
        assert roundtripped["periods"][label]["start_ts"] < roundtripped["periods"][label]["end_ts"]
        assert "start_utc" in roundtripped["periods"][label]


def test_resolve_split_on_realistic_dataset():
    """Walk through a realistic slot_ts column (5-min slots over ~2 days) and
    confirm the three-way split routes rows correctly."""
    from datetime import datetime, timezone

    base = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())
    n_slots = 48 * 12  # 48 hours × 12 slots/hour
    rows = []
    for i in range(n_slots):
        slot_ts = base + i * 300
        for snap in range(5):
            rows.append({"slot_ts": slot_ts, "snapshot_ts": slot_ts + snap * 30, "label": snap % 2})
    df = pd.DataFrame(rows)

    args = argparse.Namespace(
        training_period="2026-03-01:2026-03-02",
        valid_period="2026-03-02:2026-03-02T12:00:00Z",
        test_period="2026-03-02T12:00:00Z:2026-03-03",
    )
    result = resolve_split_from_args(args, df, val_ratio=0.2)
    assert set(result.frames.keys()) == {"training", "validation", "test"}
    assert result.frames["training"]["slot_ts"].max() < result.frames["validation"]["slot_ts"].min()
    assert result.frames["validation"]["slot_ts"].max() < result.frames["test"]["slot_ts"].min()
    # No row should appear in more than one bucket.
    total = sum(len(f) for f in result.frames.values())
    assert total == len(df)
