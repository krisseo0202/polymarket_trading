"""Round-trip test for scripts.train: probe -> selection.yaml -> train -> artifacts."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_probe_mod = _load_script_module("feature_probe_mod", _ROOT / "scripts" / "feature_probe.py")
_train_mod = _load_script_module("train_mod", _ROOT / "scripts" / "train.py")


def _synthetic(seed: int = 11, n_slots: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for slot_idx in range(n_slots):
        latent = rng.standard_normal()
        label = 1 if latent > 0 else 0
        for snap in range(3):
            rows.append({
                "slot_ts": 10_000 + slot_idx,
                "snapshot_ts": 10_000 + slot_idx + snap * 0.3,
                "label": label,
                "signal_a": latent + rng.standard_normal() * 0.3,
                "signal_b": latent * 0.5 + rng.standard_normal() * 0.5,
                "noise": rng.standard_normal(),
                "yes_mid": 0.5,
                "no_mid": 0.5,
            })
    df = pd.DataFrame(rows)
    _probe_mod._append_sentinels(df, seed=seed, n=1)
    return df


def test_probe_then_train_roundtrip(tmp_path):
    df = _synthetic()
    probe_out = tmp_path / "probe"
    probe_out.mkdir()

    # --- Run the probe exactly as feature_probe.main would, but inline. ---
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
    table.to_csv(probe_out / "importance_table.csv")
    corr.to_csv(probe_out / "correlation_matrix.csv")
    _probe_mod.save_probe_models(probe_out, models)
    _probe_mod.write_report(
        probe_out, table, corr, models,
        dataset_stats={
            "n_rows": len(df),
            "n_train_slots": train["slot_ts"].nunique(),
            "n_val_slots": val["slot_ts"].nunique(),
        },
    )

    # Persist dataset parquet at the location train.py expects.
    dataset_path = tmp_path / "dataset.parquet"
    df.to_parquet(dataset_path, index=False)

    # --- Write selection.yaml — simulating Claude's pick. ---
    above = table[table["above_random_sentinel"]]
    active = [f for f in above.index if not f.startswith("random_value")]
    assert "signal_a" in active, (
        f"signal_a should be above random floor, but above={list(above.index)}"
    )

    selection = {
        "run_id": "roundtrip-test",
        "probe_dataset": str(dataset_path),
        "active_features": active,
        "random_floor": {
            "xgb_perm_importance": float(table.loc["random_value", "xgb_perm_importance"]),
        },
        "rejected_features": [
            {"name": f, "reason": "below random floor"}
            for f in table.index
            if not table.loc[f, "above_random_sentinel"] and not f.startswith("random_value")
        ],
        "rationale": "Synthetic round-trip.",
    }
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(yaml.safe_dump(selection), encoding="utf-8")

    # --- Train via the same functions train.main uses. ---
    sel = _train_mod.load_selection(str(selection_path))
    out = tmp_path / "model_out"
    out.mkdir()

    tr, va = _train_mod.split_by_slot(df, val_ratio=0.2)
    X_train = tr[sel.active_features].to_numpy(dtype=float)
    y_train = tr["label"].to_numpy(dtype=float)
    X_val = va[sel.active_features].to_numpy(dtype=float)
    y_val = va["label"].to_numpy(dtype=float)

    lr_scaler, lr_model, lr_calibrator, lr_brier = _train_mod.train_logreg(
        X_train, y_train, X_val, y_val
    )
    assert 0.0 <= lr_brier <= 1.0
    assert hasattr(lr_model, "coef_")

    xgb_model, xgb_calibrator, xgb_brier = _train_mod.train_xgb(
        X_train, y_train, X_val, y_val
    )
    assert 0.0 <= xgb_brier <= 1.0

    # --- Slot-level PnL on both ---
    lr_p = lr_calibrator.transform(lr_model.predict_proba(lr_scaler.transform(X_val))[:, 1])
    metrics = _train_mod.slot_pnl(va, lr_p)
    assert "sharpe" in metrics
    assert metrics["n_trades"] >= 0

    # --- Promote gate against reloaded probe metrics ---
    probe_metrics = _train_mod.load_probe_metrics(probe_out, va)
    assert "logreg" in probe_metrics and "xgb" in probe_metrics
    passed, reason = _train_mod.promote_gate(
        {"brier": lr_brier, **metrics},
        probe_metrics["logreg"],
    )
    assert isinstance(passed, bool)
    assert isinstance(reason, str) and reason


def test_load_selection_parses_required_fields(tmp_path):
    path = tmp_path / "selection.yaml"
    path.write_text(
        yaml.safe_dump({
            "run_id": "r1",
            "probe_dataset": "x.parquet",
            "active_features": ["a", "b"],
            "random_floor": {"xgb_perm_importance": 0.001},
        }),
        encoding="utf-8",
    )
    sel = _train_mod.load_selection(str(path))
    assert sel.run_id == "r1"
    assert sel.active_features == ["a", "b"]
    assert sel.random_floor == {"xgb_perm_importance": 0.001}
    assert sel.rejected_features == []


def test_promote_gate_flags_brier_regression():
    passed, reason = _train_mod.promote_gate(
        main={"brier": 0.30, "sharpe": 1.0, "pnl": 5.0},
        probe={"brier": 0.20, "sharpe": 1.0, "pnl": 5.0},
    )
    assert passed is False
    assert "Brier" in reason


def test_promote_gate_passes_when_within_slack():
    passed, reason = _train_mod.promote_gate(
        main={"brier": 0.2005, "sharpe": 0.95, "pnl": 4.9},
        probe={"brier": 0.2000, "sharpe": 1.00, "pnl": 5.0},
    )
    assert passed is True
    assert "tolerance" in reason
