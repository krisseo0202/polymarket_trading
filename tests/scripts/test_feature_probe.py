"""Tests for scripts.feature_probe using a fully synthetic dataset.

Goal: ``random_value`` must rank below features we inject with real signal and
above features we inject as pure noise. Also verify redundancy detection.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

_PROBE_PATH = _ROOT / "scripts" / "feature_probe.py"
_spec = importlib.util.spec_from_file_location("feature_probe_mod", _PROBE_PATH)
assert _spec is not None and _spec.loader is not None
_probe_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _probe_mod  # dataclass inspects sys.modules during decoration
_spec.loader.exec_module(_probe_mod)


N_SLOTS = 400
N_SNAPS_PER_SLOT = 4


def _make_synthetic_dataset(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for slot_idx in range(N_SLOTS):
        # One true latent signal per slot determines the label.
        latent = rng.standard_normal()
        label = 1 if latent > 0 else 0
        for snap in range(N_SNAPS_PER_SLOT):
            true_signal = latent + rng.standard_normal() * 0.3
            redundant = true_signal * 0.95 + rng.standard_normal() * 0.05
            weak_signal = latent * 0.2 + rng.standard_normal()
            pure_noise_a = rng.standard_normal()
            pure_noise_b = rng.standard_normal()
            rows.append({
                "slot_ts": 1_000 + slot_idx,
                "snapshot_ts": 1_000 + slot_idx + snap * 0.25,
                "label": label,
                "true_signal": true_signal,
                "redundant_signal": redundant,
                "weak_signal": weak_signal,
                "pure_noise_a": pure_noise_a,
                "pure_noise_b": pure_noise_b,
                "yes_mid": 0.5,
                "no_mid": 0.5,
            })
    df = pd.DataFrame(rows)
    _probe_mod._append_sentinels(df, seed=seed, n=1)
    return df


def test_expand_date_range_inclusive():
    dates = _probe_mod.expand_date_range("2026-04-12", "2026-04-18")
    assert dates[0] == "2026-04-12"
    assert dates[-1] == "2026-04-18"
    assert len(dates) == 7


def test_expand_date_range_single_day():
    assert _probe_mod.expand_date_range("2026-04-18", "2026-04-18") == ["2026-04-18"]


def test_expand_date_range_rejects_reversed_range():
    with pytest.raises(SystemExit):
        _probe_mod.expand_date_range("2026-04-18", "2026-04-12")


def test_expand_date_range_rejects_bad_format():
    with pytest.raises(SystemExit):
        _probe_mod.expand_date_range("04/12/2026", "2026-04-18")


def test_split_by_slot_is_time_ordered():
    df = _make_synthetic_dataset()
    train, val = _probe_mod.split_by_slot(df, val_ratio=0.2)
    assert train["slot_ts"].max() < val["slot_ts"].min()
    assert len(train) + len(val) == len(df)


def test_feature_columns_excludes_metadata():
    df = _make_synthetic_dataset()
    feats = _probe_mod.feature_columns(df)
    assert "label" not in feats
    assert "slot_ts" not in feats
    assert "true_signal" in feats
    assert "random_value" in feats


def test_strong_signals_rank_above_random_sentinel():
    df = _make_synthetic_dataset()
    train, val = _probe_mod.split_by_slot(df, val_ratio=0.2)
    features = _probe_mod.feature_columns(df)

    models = _probe_mod.fit_probes(train, val, features)
    perm = _probe_mod.permutation_importance_xgb(models, val, n_repeats=8)
    stability = _probe_mod.time_stability(models, val, n_repeats=3)
    univar = _probe_mod.per_feature_univariate(
        train, val, features, fill_config=_probe_mod.FillConfig()
    )

    table = _probe_mod.build_importance_table(models, perm, stability, univar, train)

    # The strong signal MUST rank above the random floor — that's the core
    # guarantee the sentinel provides: real signal distinguishes itself from
    # noise. (Ordering among noise features is random by construction, so we
    # don't assert anything there.)
    floor = float(table.loc["random_value", "xgb_perm_importance"])
    true_imp = float(table.loc["true_signal", "xgb_perm_importance"])
    redundant_imp = float(table.loc["redundant_signal", "xgb_perm_importance"])

    assert true_imp > floor, (
        f"true_signal ({true_imp:+.4f}) must exceed random floor ({floor:+.4f})."
    )
    assert redundant_imp > floor, (
        f"redundant_signal ({redundant_imp:+.4f}) must exceed random floor ({floor:+.4f})."
    )
    assert bool(table.loc["true_signal", "above_random_sentinel"])
    assert bool(table.loc["redundant_signal", "above_random_sentinel"])


def test_redundancy_clusters_detect_near_duplicate_feature():
    df = _make_synthetic_dataset()
    features = _probe_mod.feature_columns(df)
    corr = _probe_mod.correlation_matrix(df, features)

    # Fake importance ranking: true_signal first, redundant_signal second.
    importance = pd.Series({f: 0.0 for f in features})
    importance["true_signal"] = 1.0
    importance["redundant_signal"] = 0.9
    importance["weak_signal"] = 0.2

    clusters = _probe_mod.find_redundancy_clusters(corr, importance, threshold=0.85)
    # At least one cluster should contain both true_signal and redundant_signal.
    found = False
    for cluster in clusters:
        names = {feat for feat, _ in cluster}
        if {"true_signal", "redundant_signal"}.issubset(names):
            found = True
            break
    assert found, f"expected true_signal+redundant_signal cluster, got {clusters}"


def test_report_and_importance_table_round_trip(tmp_path):
    df = _make_synthetic_dataset()
    out_dir = tmp_path / "probe"
    out_dir.mkdir()

    train, val = _probe_mod.split_by_slot(df, val_ratio=0.2)
    features = _probe_mod.feature_columns(df)
    models = _probe_mod.fit_probes(train, val, features)
    perm = _probe_mod.permutation_importance_xgb(models, val, n_repeats=5)
    stability = _probe_mod.time_stability(models, val, n_repeats=3)
    univar = _probe_mod.per_feature_univariate(
        train, val, features, fill_config=_probe_mod.FillConfig()
    )
    table = _probe_mod.build_importance_table(models, perm, stability, univar, train)
    corr = _probe_mod.correlation_matrix(train, features)

    table.to_csv(out_dir / "importance_table.csv")
    corr.to_csv(out_dir / "correlation_matrix.csv")
    _probe_mod.save_probe_models(out_dir, models)
    _probe_mod.write_report(
        out_dir, table, corr, models,
        dataset_stats={"n_rows": len(df), "n_train_slots": train["slot_ts"].nunique(),
                       "n_val_slots": val["slot_ts"].nunique()},
    )

    assert (out_dir / "report.md").exists()
    assert (out_dir / "importance_table.csv").exists()
    assert (out_dir / "probe_models" / "xgb.json").exists()
    report = (out_dir / "report.md").read_text()
    assert "Random sentinel floor" in report
    assert "true_signal" in report
