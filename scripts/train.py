"""Unified main-model training driven by a selection.yaml.

Consumes the decision record Claude writes after reading a probe report:

    experiments/<run>/selection.yaml

Produces a model artifact directory with:

    {out_dir}/
      schema.json         # exact feature list the artifact expects (meta-driven)
      logreg_model.pkl
      logreg_scaler.pkl
      logreg_calibrator.pkl
      xgb_model.json
      xgb_calibrator.pkl
      meta.json           # probe comparison, Brier, Sharpe, PnL, gate verdict

Usage:
    python scripts/train.py \\
        --selection experiments/2026-04-18-familyC-probe/selection.yaml \\
        --out models/<model_name>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.backtest.fill_sim import FillConfig, as_dict
from src.backtest.fill_sim import slot_pnl as _slot_pnl_impl
from src.backtest.period_split import (
    ResolvedSplit,
    add_period_arguments,
    resolve_split_from_args,
    split_by_val_ratio,
)
from src.models.tte_weights import (
    BUCKET_WEIGHTS,
    tte_series_to_weights,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required; pip install xgboost") from exc


_DEFAULT_VAL_RATIO = 0.20


@dataclass
class Selection:
    run_id: str
    probe_dataset: str
    active_features: List[str]
    rejected_features: List[Dict[str, str]]
    random_floor: Dict[str, float]
    rationale: str


def load_selection(path: str) -> Selection:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Selection(
        run_id=str(raw.get("run_id", "unnamed")),
        probe_dataset=str(raw["probe_dataset"]),
        active_features=list(raw["active_features"]),
        rejected_features=list(raw.get("rejected_features") or []),
        random_floor=dict(raw.get("random_floor") or {}),
        rationale=str(raw.get("rationale", "")),
    )


def split_by_slot(df: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy walk-forward split. Kept for tests that import it directly;
    new code should use ``src.backtest.period_split.resolve_split_from_args``
    which also supports explicit ``--training-period/--valid-period/--test-period``.
    """
    frames = split_by_val_ratio(df, val_ratio=val_ratio, ts_col="slot_ts")
    return frames["training"], frames["validation"]


# ---------------------------------------------------------------------------
# Main-model training
# ---------------------------------------------------------------------------


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[StandardScaler, LogisticRegression, IsotonicRegression, float]:
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        penalty="l1", C=0.1, solver="saga", max_iter=5_000, n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            scaler.transform(X_train),
            y_train,
            sample_weight=sample_weight,
        )
    raw_p = model.predict_proba(scaler.transform(X_val))[:, 1]
    # Calibrator learns on unweighted val rows: the calibration map should
    # reflect the true outcome distribution, not the training emphasis.
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y_val)
    calibrated = calibrator.transform(raw_p)
    brier = float(brier_score_loss(y_val, calibrated))
    return scaler, model, calibrator, brier


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[XGBClassifier, IsotonicRegression, float]:
    model = XGBClassifier(
        max_depth=5,
        n_estimators=800,
        learning_rate=0.03,
        eval_metric="logloss",
        early_stopping_rounds=40,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    raw_p = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p, y_val)
    calibrated = calibrator.transform(raw_p)
    brier = float(brier_score_loss(y_val, calibrated))
    return model, calibrator, brier


def compute_tte_sample_weights(
    train_df: pd.DataFrame, enabled: bool,
) -> Optional[np.ndarray]:
    """Return sample weights from TTE buckets, or None if disabled.

    Weights emphasize the core 60–180s trading window. See
    src/models/tte_weights.py for the rationale and per-bucket values.
    """
    if not enabled:
        return None
    if "slot_expiry_ts" not in train_df.columns or "snapshot_ts" not in train_df.columns:
        return None
    tte = np.maximum(
        0.0,
        train_df["slot_expiry_ts"].to_numpy(dtype=float)
        - train_df["snapshot_ts"].to_numpy(dtype=float),
    )
    return tte_series_to_weights(tte)


# ---------------------------------------------------------------------------
# Evaluation — delegates to src.backtest.fill_sim.slot_pnl so probe and train
# use the same fill rule (real ask preferred over mid+spread, configurable fee).
# ---------------------------------------------------------------------------


def slot_pnl(
    val_df: pd.DataFrame,
    p_val: np.ndarray,
    config: Optional[FillConfig] = None,
) -> Dict[str, float]:
    """Thin adapter over src.backtest.fill_sim.slot_pnl returning a dict."""
    metrics = _slot_pnl_impl(val_df, p_val, config=config)
    return as_dict(metrics)


# ---------------------------------------------------------------------------
# Promote gate
# ---------------------------------------------------------------------------


def promote_gate(
    main: Dict[str, float],
    probe: Dict[str, float],
    brier_slack: float = 0.002,
    sharpe_slack: float = 0.1,
    pnl_ratio: float = 0.95,
) -> Tuple[bool, str]:
    """Main must not regress much vs the probe (trained on ALL features)."""
    if main["brier"] > probe["brier"] + brier_slack:
        return False, f"Brier regression: {main['brier']:.4f} > {probe['brier']:.4f} + {brier_slack}"
    if main["sharpe"] < probe["sharpe"] - sharpe_slack:
        return False, f"Sharpe regression: {main['sharpe']:.2f} < {probe['sharpe']:.2f} - {sharpe_slack}"
    if probe["pnl"] > 0 and main["pnl"] < probe["pnl"] * pnl_ratio:
        return False, f"PnL regression: {main['pnl']:.2f} < {probe['pnl']:.2f} × {pnl_ratio}"
    return True, "Main model within tolerance of probe on Brier, Sharpe, PnL"


# ---------------------------------------------------------------------------
# Probe Brier reload (for the gate)
# ---------------------------------------------------------------------------


def load_probe_metrics(
    probe_dir: Path,
    val_df: pd.DataFrame,
    fill_config: Optional[FillConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """Re-score the probe models on the same val split for an apples-to-apples gate."""
    import xgboost as xgb_mod

    metrics: Dict[str, Dict[str, float]] = {}
    probe_models_dir = probe_dir / "probe_models"
    if not probe_models_dir.exists():
        logging.warning("No probe_models/ found at %s; skipping gate", probe_models_dir)
        return metrics

    feature_names_path = probe_models_dir / "feature_names.json"
    feature_names = json.loads(feature_names_path.read_text()) if feature_names_path.exists() else []
    if not feature_names:
        return metrics

    X = val_df[feature_names].to_numpy(dtype=float)
    y = val_df["label"].to_numpy(dtype=float)

    with (probe_models_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    with (probe_models_dir / "logreg.pkl").open("rb") as f:
        logreg = pickle.load(f)
    p_lr = logreg.predict_proba(scaler.transform(X))[:, 1]
    metrics["logreg"] = {
        "brier": float(brier_score_loss(y, p_lr)),
        **slot_pnl(val_df, p_lr, config=fill_config),
    }

    xgb = xgb_mod.XGBClassifier()
    xgb.load_model(str(probe_models_dir / "xgb.json"))
    p_xg = xgb.predict_proba(X)[:, 1]
    metrics["xgb"] = {
        "brier": float(brier_score_loss(y, p_xg)),
        **slot_pnl(val_df, p_xg, config=fill_config),
    }

    return metrics


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LogReg + XGB on selected features")
    parser.add_argument("--selection", required=True, help="Path to selection.yaml")
    parser.add_argument("--out", required=True, help="Output model artifact directory")
    parser.add_argument(
        "--val-ratio", type=float, default=_DEFAULT_VAL_RATIO,
        help="Legacy fallback when no --training-period is given. Defaults to 0.20.",
    )
    parser.add_argument(
        "--fee-bps", type=float, default=0.0,
        help="Flat per-trade fee in bps (for gas/protocol fees). Default 0.",
    )
    parser.add_argument(
        "--synthetic-half-spread", type=float, default=0.01,
        help="Fallback half-spread added to mid when real ask is missing.",
    )
    parser.add_argument(
        "--tte-weighted", action="store_true", default=True,
        help="Weight training loss by TTE bucket (emphasize core 60-180s window). "
             "Default on. See src/models/tte_weights.py.",
    )
    parser.add_argument(
        "--no-tte-weighted", dest="tte_weighted", action="store_false",
        help="Disable TTE-bucket weighting (use uniform sample weights).",
    )
    parser.add_argument("--log-level", default="INFO")
    add_period_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    sel = load_selection(args.selection)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sel.probe_dataset)
    logging.info("Loaded dataset: %d rows, %d slots",
                 len(df), df["slot_ts"].nunique())

    missing = [f for f in sel.active_features if f not in df.columns]
    if missing:
        raise SystemExit(f"Selected features missing from dataset: {missing}")

    split = resolve_split_from_args(args, df, val_ratio=args.val_ratio)
    train_df = split.frames["training"]
    val_df = split.frames.get("validation")
    test_df = split.frames.get("test")

    if val_df is None:
        raise SystemExit(
            "Training requires a validation set. Either provide --valid-period "
            "or rely on the default --val-ratio (non-zero)."
        )

    X_train = train_df[sel.active_features].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=float)
    X_val = val_df[sel.active_features].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=float)

    fill_cfg = FillConfig(
        synthetic_half_spread=args.synthetic_half_spread,
        fee_bps=args.fee_bps,
    )

    sample_weight = compute_tte_sample_weights(train_df, enabled=args.tte_weighted)
    if sample_weight is not None:
        logging.info(
            "TTE-weighted training ON — bucket weights: %s",
            BUCKET_WEIGHTS,
        )
    else:
        logging.info("TTE-weighted training OFF — uniform sample weights")

    # --- LogReg -----------------------------------------------------------------
    lr_scaler, lr_model, lr_calibrator, lr_brier = train_logreg(
        X_train, y_train, X_val, y_val, sample_weight=sample_weight,
    )
    lr_p = lr_calibrator.transform(lr_model.predict_proba(lr_scaler.transform(X_val))[:, 1])
    lr_metrics = {"brier": lr_brier, **slot_pnl(val_df, lr_p, config=fill_cfg)}
    logging.info("LogReg  brier=%.4f sharpe=%.2f pnl=%.2f trades=%d",
                 lr_metrics["brier"], lr_metrics["sharpe"], lr_metrics["pnl"], lr_metrics["n_trades"])

    with (out_dir / "logreg_scaler.pkl").open("wb") as f:
        pickle.dump(lr_scaler, f)
    with (out_dir / "logreg_model.pkl").open("wb") as f:
        pickle.dump(lr_model, f)
    with (out_dir / "logreg_calibrator.pkl").open("wb") as f:
        pickle.dump(lr_calibrator, f)

    # --- XGB --------------------------------------------------------------------
    xgb_model, xgb_calibrator, xgb_brier = train_xgb(
        X_train, y_train, X_val, y_val, sample_weight=sample_weight,
    )
    xgb_p = xgb_calibrator.transform(xgb_model.predict_proba(X_val)[:, 1])
    xgb_metrics = {"brier": xgb_brier, **slot_pnl(val_df, xgb_p, config=fill_cfg)}
    logging.info("XGB     brier=%.4f sharpe=%.2f pnl=%.2f trades=%d",
                 xgb_metrics["brier"], xgb_metrics["sharpe"], xgb_metrics["pnl"], xgb_metrics["n_trades"])

    xgb_model.save_model(str(out_dir / "xgb_model.json"))
    with (out_dir / "xgb_calibrator.pkl").open("wb") as f:
        pickle.dump(xgb_calibrator, f)

    # --- Schema + meta ---------------------------------------------------------
    (out_dir / "schema.json").write_text(
        json.dumps({"features": sel.active_features}, indent=2), encoding="utf-8"
    )

    # Held-out test evaluation (optional). Runs only when --test-period is
    # configured; gives a single final number that the val-set metrics
    # (used for calibration + gate) can no longer leak into.
    test_metrics: Dict[str, Dict[str, float]] = {}
    if test_df is not None:
        X_test = test_df[sel.active_features].to_numpy(dtype=float)
        y_test = test_df["label"].to_numpy(dtype=float)

        lr_p_test = lr_calibrator.transform(
            lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]
        )
        test_metrics["logreg"] = {
            "brier": float(brier_score_loss(y_test, lr_p_test)),
            **slot_pnl(test_df, lr_p_test, config=fill_cfg),
        }
        xgb_p_test = xgb_calibrator.transform(xgb_model.predict_proba(X_test)[:, 1])
        test_metrics["xgb"] = {
            "brier": float(brier_score_loss(y_test, xgb_p_test)),
            **slot_pnl(test_df, xgb_p_test, config=fill_cfg),
        }
        logging.info(
            "TEST    logreg brier=%.4f sharpe=%.2f pnl=%.2f trades=%d",
            test_metrics["logreg"]["brier"], test_metrics["logreg"]["sharpe"],
            test_metrics["logreg"]["pnl"], test_metrics["logreg"]["n_trades"],
        )
        logging.info(
            "TEST    xgb    brier=%.4f sharpe=%.2f pnl=%.2f trades=%d",
            test_metrics["xgb"]["brier"], test_metrics["xgb"]["sharpe"],
            test_metrics["xgb"]["pnl"], test_metrics["xgb"]["n_trades"],
        )

    probe_dir = Path(sel.probe_dataset).parent / "probe"
    probe_metrics = load_probe_metrics(probe_dir, val_df, fill_config=fill_cfg)

    gate_results: Dict[str, Dict[str, object]] = {}
    for name, main_m in (("logreg", lr_metrics), ("xgb", xgb_metrics)):
        probe_m = probe_metrics.get(name)
        if probe_m:
            passed, reason = promote_gate(main_m, probe_m)
            gate_results[name] = {
                "passed": passed,
                "reason": reason,
                "probe": probe_m,
            }
        else:
            gate_results[name] = {"passed": None, "reason": "probe metrics unavailable"}

    meta = {
        "selection_path": args.selection,
        "run_id": sel.run_id,
        "feature_count": len(sel.active_features),
        "n_train_slots": int(train_df["slot_ts"].nunique()),
        "n_val_slots": int(val_df["slot_ts"].nunique()),
        "split_mode": "val_ratio" if split.used_fallback else "periods",
        "val_ratio_fallback": args.val_ratio if split.used_fallback else None,
        "periods": split.config.to_dict() if split.config is not None else None,
        "fill_config": {
            "synthetic_half_spread": fill_cfg.synthetic_half_spread,
            "fee_bps": fill_cfg.fee_bps,
            "entry_threshold": fill_cfg.entry_threshold,
        },
        "tte_weighted": args.tte_weighted,
        "tte_bucket_weights": BUCKET_WEIGHTS if args.tte_weighted else None,
        "logreg": lr_metrics,
        "xgb": xgb_metrics,
        "gate": gate_results,
        "rejected_features": sel.rejected_features,
    }
    if test_df is not None:
        meta["n_test_slots"] = int(test_df["slot_ts"].nunique())
        meta["test"] = test_metrics
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    print(f"\nTraining complete → {out_dir}")
    for name, res in gate_results.items():
        if res.get("passed") is True:
            print(f"  {name}: PROMOTE — {res['reason']}")
        elif res.get("passed") is False:
            print(f"  {name}: REJECT — {res['reason']}")
        else:
            print(f"  {name}: N/A — {res['reason']}")


if __name__ == "__main__":
    main()
