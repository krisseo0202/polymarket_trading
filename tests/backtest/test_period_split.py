"""Tests for the shared period-based train/val/test split module."""

import argparse
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.backtest.period_split import (
    PERIOD_LABELS,
    PeriodBounds,
    add_period_arguments,
    parse_period_config,
    parse_period_spec,
    resolve_split_from_args,
    split_by_periods,
    split_by_val_ratio,
)


# ---------------------------------------------------------------------------
# PeriodBounds
# ---------------------------------------------------------------------------


def test_period_bounds_rejects_invalid_label():
    with pytest.raises(ValueError):
        PeriodBounds(label="garbage", start_ts=0, end_ts=1)


def test_period_bounds_rejects_empty_window():
    with pytest.raises(ValueError, match="must be > start_ts"):
        PeriodBounds(label="training", start_ts=100, end_ts=100)


def test_period_bounds_contains_is_half_open():
    p = PeriodBounds(label="training", start_ts=100, end_ts=200)
    assert p.contains(100) is True      # inclusive start
    assert p.contains(150) is True
    assert p.contains(199.999) is True
    assert p.contains(200) is False     # exclusive end
    assert p.contains(99) is False


def test_period_bounds_mask_matches_contains():
    p = PeriodBounds(label="training", start_ts=100, end_ts=200)
    s = pd.Series([50, 100, 150, 199, 200, 250])
    assert p.mask(s).tolist() == [False, True, True, True, False, False]


def test_period_bounds_humanize_uses_utc():
    p = PeriodBounds(label="training", start_ts=1776556800, end_ts=1776643200)
    assert "2026-04-19T00:00:00+00:00" in p.humanize()
    assert "2026-04-20T00:00:00+00:00" in p.humanize()


def test_period_bounds_to_dict_roundtrip():
    p = PeriodBounds(label="training", start_ts=1776556800, end_ts=1776643200)
    d = p.to_dict()
    assert d["label"] == "training"
    assert d["start_ts"] == 1776556800.0
    assert "2026-04-19" in d["start_utc"]


# ---------------------------------------------------------------------------
# parse_period_spec
# ---------------------------------------------------------------------------


def test_parse_date_only_spec():
    p = parse_period_spec("2026-03-01:2026-04-01", "training")
    # Date-only is interpreted as 00:00 UTC start-of-day.
    assert p.start_ts == datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()
    assert p.end_ts == datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp()


def test_parse_iso_datetime_spec():
    spec = "2026-03-01T12:00:00Z:2026-03-02T12:00:00Z"
    p = parse_period_spec(spec, "validation")
    assert p.start_ts == datetime(2026, 3, 1, 12, tzinfo=timezone.utc).timestamp()
    assert p.end_ts == datetime(2026, 3, 2, 12, tzinfo=timezone.utc).timestamp()


def test_parse_unix_seconds_spec():
    p = parse_period_spec("1776000000:1776600000", "test")
    assert p.start_ts == 1776000000.0
    assert p.end_ts == 1776600000.0


def test_parse_dotted_separator_also_works():
    p = parse_period_spec("2026-03-01..2026-04-01", "training")
    assert p.start_ts < p.end_ts


def test_parse_rejects_malformed_spec():
    for bad in ("", "only-one-side", "a:b:c:d:e:f:g:h:i", "2026-03-01"):
        with pytest.raises(ValueError):
            parse_period_spec(bad, "training")


def test_parse_rejects_inverted_range():
    with pytest.raises(ValueError, match="must be > start_ts"):
        parse_period_spec("2026-04-01:2026-03-01", "training")


# ---------------------------------------------------------------------------
# parse_period_config
# ---------------------------------------------------------------------------


def test_parse_period_config_none_when_no_training():
    assert parse_period_config(training=None) is None


def test_parse_period_config_rejects_val_without_training():
    with pytest.raises(ValueError, match="require --training-period"):
        parse_period_config(training=None, validation="2026-03-01:2026-04-01")


def test_parse_period_config_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        parse_period_config(
            training="2026-03-01:2026-03-20",
            validation="2026-03-15:2026-04-01",  # overlaps training
        )


def test_parse_period_config_rejects_out_of_order():
    with pytest.raises(ValueError, match="chronological"):
        parse_period_config(
            training="2026-04-01:2026-05-01",
            validation="2026-03-01:2026-04-01",  # earlier than training
        )


def test_parse_period_config_full_three_way():
    cfg = parse_period_config(
        training="2026-03-01:2026-04-01",
        validation="2026-04-01:2026-04-15",
        test="2026-04-15:2026-05-01",
    )
    assert cfg is not None
    assert [p.label for p in cfg.periods()] == list(PERIOD_LABELS)


def test_parse_period_config_training_only():
    cfg = parse_period_config(training="2026-03-01:2026-04-01")
    assert cfg is not None
    assert cfg.validation is None
    assert cfg.test is None
    assert len(cfg.periods()) == 1


# ---------------------------------------------------------------------------
# split_by_periods
# ---------------------------------------------------------------------------


def _synthetic_df(n_slots=100, slot_width_s=300):
    """Build a toy dataset with `n_slots` unique slot_ts values, spanning
    ``n_slots * slot_width_s`` seconds starting at 2026-03-01T00:00 UTC."""
    base = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()
    rows = []
    for i in range(n_slots):
        for snap in range(3):
            rows.append({
                "slot_ts": int(base + i * slot_width_s),
                "snapshot_ts": base + i * slot_width_s + snap * 30,
                "label": (i + snap) % 2,
            })
    return pd.DataFrame(rows)


def test_split_by_periods_filters_correctly():
    # 24 daily slots starting 2026-03-01 → data spans 2026-03-01 ... 2026-03-24.
    df = _synthetic_df(n_slots=24, slot_width_s=86400)
    cfg = parse_period_config(
        training="2026-03-01:2026-03-11",
        validation="2026-03-11:2026-03-16",
        test="2026-03-16:2026-03-25",
    )
    assert cfg is not None
    frames = split_by_periods(df, cfg, ts_col="slot_ts")

    total = sum(len(sub) for sub in frames.values())
    assert total == len(df)
    for p in cfg.periods():
        assert not frames[p.label].empty
        assert (frames[p.label]["slot_ts"] >= p.start_ts).all()
        assert (frames[p.label]["slot_ts"] < p.end_ts).all()


def test_split_by_periods_empty_window_raises():
    df = _synthetic_df(n_slots=10)
    # Pick a training window far before the data.
    cfg = parse_period_config(training="2000-01-01:2000-01-02")
    assert cfg is not None
    with pytest.raises(ValueError, match="matched 0 rows"):
        split_by_periods(df, cfg, ts_col="slot_ts")


def test_split_by_periods_missing_column_raises():
    df = pd.DataFrame({"not_slot_ts": [1, 2, 3]})
    cfg = parse_period_config(training="2026-03-01:2026-04-01")
    assert cfg is not None
    with pytest.raises(KeyError, match="slot_ts"):
        split_by_periods(df, cfg, ts_col="slot_ts")


def test_split_by_periods_custom_ts_col():
    df = _synthetic_df(n_slots=50)
    cfg = parse_period_config(training="2026-03-01:2026-04-01")
    assert cfg is not None
    # Using snapshot_ts instead of slot_ts.
    frames = split_by_periods(df, cfg, ts_col="snapshot_ts")
    assert frames["training"]["snapshot_ts"].min() >= cfg.training.start_ts
    assert frames["training"]["snapshot_ts"].max() < cfg.training.end_ts


# ---------------------------------------------------------------------------
# split_by_val_ratio (legacy fallback)
# ---------------------------------------------------------------------------


def test_split_by_val_ratio_last_slots_go_to_val():
    df = _synthetic_df(n_slots=10)
    frames = split_by_val_ratio(df, val_ratio=0.2)
    # Last 20% of 10 unique slots = 2 slots into val.
    assert frames["validation"]["slot_ts"].nunique() == 2
    assert frames["training"]["slot_ts"].nunique() == 8
    # Val timestamps are strictly after train timestamps.
    assert frames["training"]["slot_ts"].max() < frames["validation"]["slot_ts"].min()


def test_split_by_val_ratio_rejects_invalid_ratio():
    df = _synthetic_df(n_slots=5)
    for bad in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError):
            split_by_val_ratio(df, val_ratio=bad)


# ---------------------------------------------------------------------------
# resolve_split_from_args — CLI integration
# ---------------------------------------------------------------------------


def _args(**kw):
    """Mock argparse namespace with sensible defaults."""
    ns = argparse.Namespace(training_period=None, valid_period=None, test_period=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_resolve_split_falls_back_to_val_ratio_when_no_periods():
    df = _synthetic_df(n_slots=10)
    result = resolve_split_from_args(_args(), df, val_ratio=0.2)
    assert result.used_fallback is True
    assert result.config is None
    assert set(result.frames.keys()) == {"training", "validation"}


def test_resolve_split_uses_periods_when_given():
    df = _synthetic_df(n_slots=10, slot_width_s=86400)  # 10 daily slots
    args = _args(training_period="2026-03-01:2026-03-05", valid_period="2026-03-05:2026-03-10")
    result = resolve_split_from_args(args, df, val_ratio=0.2)
    assert result.used_fallback is False
    assert result.config is not None
    assert set(result.frames.keys()) == {"training", "validation"}


def test_resolve_split_three_way():
    df = _synthetic_df(n_slots=24, slot_width_s=86400)
    args = _args(
        training_period="2026-03-01:2026-03-11",
        valid_period="2026-03-11:2026-03-16",
        test_period="2026-03-16:2026-03-25",
    )
    result = resolve_split_from_args(args, df, val_ratio=0.2)
    assert set(result.frames.keys()) == {"training", "validation", "test"}
    assert result.config is not None


def test_add_period_arguments_attaches_three_flags():
    parser = argparse.ArgumentParser()
    add_period_arguments(parser)
    parsed = parser.parse_args([
        "--training-period", "2026-03-01:2026-04-01",
        "--valid-period", "2026-04-01:2026-04-15",
        "--test-period", "2026-04-15:2026-05-01",
    ])
    assert parsed.training_period == "2026-03-01:2026-04-01"
    assert parsed.valid_period == "2026-04-01:2026-04-15"
    assert parsed.test_period == "2026-04-15:2026-05-01"


def test_period_config_to_dict_is_serializable_for_meta_json():
    import json
    cfg = parse_period_config(
        training="2026-03-01:2026-04-01",
        validation="2026-04-01:2026-04-15",
        test="2026-04-15:2026-05-01",
    )
    assert cfg is not None
    # meta.json round-trip: encode with default=float for numpy compatibility.
    payload = json.dumps(cfg.to_dict(), default=float)
    back = json.loads(payload)
    assert set(back.keys()) == {"training", "validation", "test"}
    assert back["training"]["start_ts"] == cfg.training.start_ts
