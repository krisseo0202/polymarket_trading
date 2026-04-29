"""Hyperparameter tuning for logreg_fb training.

Runs Optuna to tune the LogReg + XGB pair on a selection.yaml, then writes a
tuned model artifact in the same layout as scripts/train.py (so LogRegFBModel
can load it unchanged).

Objective: minimize calibrated validation Brier. The final gate also checks
the existing promote criteria (Brier ≤ probe + slack, Sharpe, PnL).

Usage:
    .venv/bin/python scripts/tune_logreg_fb.py \\
        --selection experiments/full-week/selection.yaml \\
        --out models/full_week_tuned \\
        --trials 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required; pip install xgboost") from exc

from src.backtest.fill_sim import FillConfig, as_dict
from src.backtest.fill_sim import slot_pnl as _slot_pnl_impl
from src.models.tte_weights import BUCKET_WEIGHTS

# Re-use train.py's promote gate + selection loader
from scripts.train import (
    Selection,
    compute_tte_sample_weights,
    load_selection,
    load_probe_metrics,
    promote_gate,
    split_by_slot,
)


def slot_pnl(val_df, p_val, config=None) -> Dict[str, float]:
    return as_dict(_slot_pnl_impl(val_df, p_val, config=config))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _train_tuned_logreg(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[StandardScaler, LogisticRegression, IsotonicRegression, float]:
    C = trial.suggest_float("lr_C", 0.01, 10.0, log=True)
    # l1 ratio only relevant if we pick elasticnet; stick with pure L1 for
    # interpretability + deterministic sparsity.
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        penalty="l1", C=C, solver="saga", max_iter=5_000, n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaler.transform(X_train), y_train)
    raw_p = model.predict_proba(scaler.transform(X_val))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y_val)
    calibrated = calibrator.transform(raw_p)
    brier = float(brier_score_loss(y_val, calibrated))
    return scaler, model, calibrator, brier


def _train_tuned_xgb(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[XGBClassifier, IsotonicRegression, float]:
    params: Dict[str, Any] = {
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
    }
    model = XGBClassifier(
        **params,
        eval_metric="logloss",
        early_stopping_rounds=40,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    raw_p = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y_val)
    calibrated = calibrator.transform(raw_p)
    brier = float(brier_score_loss(y_val, calibrated))
    return model, calibrator, brier


def make_objective(
    mode: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df: pd.DataFrame,
    fill_cfg: FillConfig,
):
    def objective(trial: optuna.Trial) -> float:
        if mode == "logreg":
            scaler, model, calibrator, brier = _train_tuned_logreg(
                trial, X_train, y_train, X_val, y_val
            )
            p_cal = calibrator.transform(model.predict_proba(scaler.transform(X_val))[:, 1])
        else:  # xgb
            model, calibrator, brier = _train_tuned_xgb(
                trial, X_train, y_train, X_val, y_val
            )
            p_cal = calibrator.transform(model.predict_proba(X_val)[:, 1])

        # Attach slot-PnL to trial user-attrs so the writer can pick a winner
        # that satisfies the promote gate, not just the lowest Brier.
        metrics = slot_pnl(val_df, p_cal, config=fill_cfg)
        trial.set_user_attr("brier", brier)
        trial.set_user_attr("pnl", metrics["pnl"])
        trial.set_user_attr("sharpe", metrics["sharpe"])
        trial.set_user_attr("n_trades", metrics["n_trades"])
        trial.set_user_attr("win_rate", metrics["win_rate"])
        trial.set_user_attr("mean_entry_cost_bps", metrics["mean_entry_cost_bps"])
        return brier
    return objective


# ---------------------------------------------------------------------------
# Retrain winner + write artifacts
# ---------------------------------------------------------------------------


def retrain_and_save_logreg(
    out_dir: Path,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, np.ndarray]:
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        penalty="l1",
        C=params["lr_C"],
        solver="saga",
        max_iter=5_000,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaler.transform(X_train), y_train)
    raw_p = model.predict_proba(scaler.transform(X_val))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y_val)
    calibrated = calibrator.transform(raw_p)
    brier = float(brier_score_loss(y_val, calibrated))

    with (out_dir / "logreg_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    with (out_dir / "logreg_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    with (out_dir / "logreg_calibrator.pkl").open("wb") as f:
        pickle.dump(calibrator, f)
    return brier, calibrated


def retrain_and_save_xgb(
    out_dir: Path,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, np.ndarray]:
    xgb_params = {k[len("xgb_"):]: v for k, v in params.items() if k.startswith("xgb_")}
    model = XGBClassifier(
        **xgb_params,
        eval_metric="logloss",
        early_stopping_rounds=40,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    raw_p = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y_val)
    calibrated = calibrator.transform(raw_p)
    brier = float(brier_score_loss(y_val, calibrated))

    model.save_model(str(out_dir / "xgb_model.json"))
    with (out_dir / "xgb_calibrator.pkl").open("wb") as f:
        pickle.dump(calibrator, f)
    return brier, calibrated


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna-tuned LogReg + XGB for logreg_fb")
    parser.add_argument("--selection", required=True, help="Path to selection.yaml")
    parser.add_argument("--out", required=True, help="Output model dir")
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials per mode")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--synthetic-half-spread", type=float, default=0.01)
    parser.add_argument("--skip-xgb", action="store_true", default=False,
                        help="Skip XGB tuning (LogReg only; faster).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    sel = load_selection(args.selection)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sel.probe_dataset)
    missing = [f for f in sel.active_features if f not in df.columns]
    if missing:
        raise SystemExit(f"Selected features missing from dataset: {missing}")

    train_df, val_df = split_by_slot(df, val_ratio=args.val_ratio)
    X_train = train_df[sel.active_features].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=float)
    X_val = val_df[sel.active_features].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=float)

    fill_cfg = FillConfig(
        synthetic_half_spread=args.synthetic_half_spread,
        fee_bps=args.fee_bps,
    )

    # ---- LogReg tune ----
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lr_study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        study_name="logreg_fb_lr",
    )
    lr_study.optimize(
        make_objective("logreg", X_train, y_train, X_val, y_val, val_df, fill_cfg),
        n_trials=args.trials,
        show_progress_bar=False,
    )
    lr_best = lr_study.best_trial
    logging.info(
        "LogReg best: brier=%.4f C=%.4g sharpe=%.2f pnl=%.2f",
        lr_best.value, lr_best.params["lr_C"],
        lr_best.user_attrs.get("sharpe", 0.0),
        lr_best.user_attrs.get("pnl", 0.0),
    )

    # ---- XGB tune (optional) ----
    xgb_best = None
    if not args.skip_xgb:
        xgb_study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=args.seed + 1),
            study_name="logreg_fb_xgb",
        )
        xgb_study.optimize(
            make_objective("xgb", X_train, y_train, X_val, y_val, val_df, fill_cfg),
            n_trials=args.trials,
            show_progress_bar=False,
        )
        xgb_best = xgb_study.best_trial
        logging.info(
            "XGB best:    brier=%.4f sharpe=%.2f pnl=%.2f",
            xgb_best.value,
            xgb_best.user_attrs.get("sharpe", 0.0),
            xgb_best.user_attrs.get("pnl", 0.0),
        )

    # ---- Retrain winners + persist ----
    lr_brier, lr_p = retrain_and_save_logreg(
        out_dir, lr_best.params, X_train, y_train, X_val, y_val
    )
    lr_metrics = {"brier": lr_brier, **slot_pnl(val_df, lr_p, config=fill_cfg)}

    xgb_metrics: Dict[str, float] = {}
    if xgb_best is not None:
        xgb_brier, xgb_p = retrain_and_save_xgb(
            out_dir, xgb_best.params, X_train, y_train, X_val, y_val
        )
        xgb_metrics = {"brier": xgb_brier, **slot_pnl(val_df, xgb_p, config=fill_cfg)}

    # ---- Schema + meta ----
    (out_dir / "schema.json").write_text(
        json.dumps({"features": sel.active_features}, indent=2), encoding="utf-8"
    )

    probe_dir = Path(sel.probe_dataset).parent / "probe"
    probe_metrics = load_probe_metrics(probe_dir, val_df, fill_config=fill_cfg)

    gate_results: Dict[str, Dict[str, object]] = {}
    for name, main_m in (("logreg", lr_metrics), ("xgb", xgb_metrics)):
        if not main_m:
            continue
        probe_m = probe_metrics.get(name)
        if probe_m:
            passed, reason = promote_gate(main_m, probe_m)
            gate_results[name] = {"passed": passed, "reason": reason, "probe": probe_m}
        else:
            gate_results[name] = {"passed": None, "reason": "probe metrics unavailable"}

    meta = {
        "selection_path": args.selection,
        "run_id": sel.run_id,
        "feature_count": len(sel.active_features),
        "n_train_slots": int(train_df["slot_ts"].nunique()),
        "n_val_slots": int(val_df["slot_ts"].nunique()),
        "trials_per_mode": args.trials,
        "fill_config": {
            "synthetic_half_spread": fill_cfg.synthetic_half_spread,
            "fee_bps": fill_cfg.fee_bps,
            "entry_threshold": fill_cfg.entry_threshold,
        },
        "logreg": {
            "metrics": lr_metrics,
            "best_params": lr_best.params,
            "trials": len(lr_study.trials),
        },
        "xgb": None if xgb_best is None else {
            "metrics": xgb_metrics,
            "best_params": xgb_best.params,
            "trials": len(xgb_study.trials),
        },
        "gate": gate_results,
        "rejected_features": sel.rejected_features,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    print(f"\nTuning complete → {out_dir}")
    print(f"  LogReg  brier={lr_metrics['brier']:.4f}  sharpe={lr_metrics['sharpe']:.2f}  pnl={lr_metrics['pnl']:.2f}  C={lr_best.params['lr_C']:.4g}")
    if xgb_best is not None:
        print(f"  XGB     brier={xgb_metrics['brier']:.4f}  sharpe={xgb_metrics['sharpe']:.2f}  pnl={xgb_metrics['pnl']:.2f}")
    for name, res in gate_results.items():
        if res.get("passed") is True:
            print(f"  {name}: PROMOTE — {res['reason']}")
        elif res.get("passed") is False:
            print(f"  {name}: REJECT — {res['reason']}")
        else:
            print(f"  {name}: N/A — {res['reason']}")


if __name__ == "__main__":
    main()
