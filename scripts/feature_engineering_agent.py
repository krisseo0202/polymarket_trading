"""LLM-driven feature engineering loop.

Three phases:
  1. ABLATION  — drop each feature family, measure ΔPNL/ΔBrier on the test window.
  2. PROPOSE   — feed the ablation table + diagnostics to Claude (role: senior quant
                 feature engineer). Get back JSON: drop_groups, proposals (with
                 executable Python), hypothesis.
  3. ADDITION  — for each proposal: validate code → exec in restricted namespace
                 → sentinel-gate via feature_probe → retrain → backtest →
                 promote_gate → write dated selection.yaml + history record.

Safety properties:
  • Fixed test window enforced (raises if --test-period missing).
  • Proposed Python is scanned for `import` and other risky tokens before exec;
    runs in `{"np": np, "pd": pd, "df": df}` only.
  • Output validated: pd.Series, float dtype, length == len(df), <20% NaN.
  • Sentinel gate runs BEFORE retrain. No retrain on noise.
  • selection.yaml never overwritten in-place — agent writes a dated copy.
  • Multi-proposal acceptance is gated by a COMBINED retrain. Proposals that
    pass alone but regress together are rejected.
  • Accepted proposals are materialized into a new parquet (dataset_with_fe.parquet)
    and their python_code is persisted to feature_definitions.json so future
    training runs are reproducible without re-executing the LLM.
  • --dry-run stops after Phase 2, makes zero file changes.

Usage:
    python scripts/feature_engineering_agent.py \\
        --selection experiments/signed-v1/selection.yaml \\
        --training-period 2026-04-15..2026-04-21 \\
        --valid-period   2026-04-21..2026-04-22 \\
        --test-period    2026-04-22..2026-04-29 \\
        --out experiments/fe_agent/$(date +%Y%m%d-%H%M%S) \\
        --max-proposals 3 \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env so ANTHROPIC_API_KEY (and any other secrets) are available before
# the anthropic SDK is constructed in _call_claude.
try:
    from dotenv import load_dotenv  # type: ignore
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except ImportError:
    pass

from src.backtest.fill_sim import FillConfig
from src.backtest.period_split import add_period_arguments, resolve_split_from_args
from src.models.feature_group_map import FEATURE_GROUPS

import importlib.util
_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "train.py")
_train_spec = importlib.util.spec_from_file_location("_train_mod", _TRAIN_PATH)
_train_mod = importlib.util.module_from_spec(_train_spec)
sys.modules["_train_mod"] = _train_mod
_train_spec.loader.exec_module(_train_mod)

load_selection = _train_mod.load_selection
promote_gate = _train_mod.promote_gate

_ABL_PATH = os.path.join(os.path.dirname(__file__), "feature_ablation.py")
_abl_spec = importlib.util.spec_from_file_location("_abl_mod", _ABL_PATH)
_abl_mod = importlib.util.module_from_spec(_abl_spec)
sys.modules["_abl_mod"] = _abl_mod
_abl_spec.loader.exec_module(_abl_mod)

run_ablation = _abl_mod.run_ablation
_fit_and_eval = _abl_mod._fit_and_eval


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are a senior quantitative feature engineer working on a
Polymarket BTC 5-minute Up/Down binary-outcome trading model. Your job is to
read ablation results and propose specific new features that will improve
P&L on a held-out test set.

CONTEXT
- The model predicts P(Up resolves) for the next 5-minute window.
- Resolution: Chainlink BTC/USD start vs end price for the window.
- Current model: XGBoost classifier with isotonic calibration.
- Features are organized into 12 family groups (see FEATURE_GROUPS).

OBJECTIVE
- Primary: maximize ΔPNL on the fixed test window.
- Hard gates: ΔBrier ≤ +0.005, ΔSharpe ≥ -0.10, |per-side calibration gap| ≤ 0.10.
- Brier-only winners that lose money are rejected.

INPUT (per turn)
- ABLATION TABLE: per-group ΔPNL/ΔBrier/ΔSharpe/Δwin_rate vs baseline.
- BASELINE METRICS: brier, sharpe, pnl, win_rate, per-side calibration gap.

OUTPUT (strict JSON, no prose, no markdown fences)
{
  "drop_groups": ["group_name", ...],     // groups whose ΔPNL was non-negative
                                          //   (we lose nothing or gain by dropping)
  "proposals": [
    {
      "name": "snake_case_feature_name",   // unique, not in any existing group
      "description": "1-2 sentence why",
      "python_code": "def compute(df):\\n    return ..."
                                          //   single function `compute(df) -> pd.Series`
                                          //   uses only df columns, np, pd
                                          //   NO imports, NO file/network I/O
                                          //   NO eval/exec/__import__/getattr
                                          //   returns float64 Series of len(df)
    }, ...
  ],
  "hypothesis": "1-3 sentences explaining the failure mode and why these
                 features address it"
}

CONSTRAINTS
- Up to 5 proposals per turn.
- Each python_code MUST reference only columns that exist in the schema (pass
  the schema column list).
- Prefer features that target the failure mode visible in the per-side
  calibration gap, not random new ratios.
- If a group is structurally important (e.g. ob_basic) but its ΔPNL is small,
  do NOT recommend dropping it — only drop groups with clear non-negative ΔPNL
  AND no critical per-side calibration regression.

KILL-TEST GATES (your proposals are auto-rejected if they fail any of these)
1. NaN fraction > 5% on any split → write features that always produce a
   number, even on warmup rows. Default to 0.0 / -1.0 sentinels for missing.
2. |Pearson corr| with any existing feature > 0.98 → don't propose features
   that are linear duplicates of existing columns (e.g. yes_mid - 0.5 when
   yes_mid is already a feature). Differentiate via nonlinear transforms,
   ratios across families, or interactions.
3. |Pearson corr| with label > 0.95 → forbidden, this is label leakage.
   Never use the label column or anything derived from a future timestamp.
4. KS-statistic between train and test > 0.20 → distribution shifted; the
   feature won't generalize. Avoid features that depend on absolute time
   (e.g. raw timestamps) or on regime flags that change between train and test.
5. Single value covers > 99% of train rows → the feature is effectively
   constant; useless to the model.

Aim for features that are dense (low NaN), distinct (low corr with existing),
honest (no future info), and stable (similar distribution train vs test).
"""


# ---------------------------------------------------------------------------
# Restricted code execution
# ---------------------------------------------------------------------------


_FORBIDDEN_TOKENS = (
    "import ", "from ", "__import__", "exec(", "eval(", "compile(",
    "open(", "globals(", "locals(", "getattr(", "setattr(",
    "delattr(", "subprocess", "os.", "sys.", "Path(",
)


def _scan_code(code: str) -> Optional[str]:
    """Return reason string if code contains forbidden tokens, else None."""
    if "def compute(" not in code:
        return "missing_def_compute"
    for tok in _FORBIDDEN_TOKENS:
        if tok in code:
            return f"forbidden_token:{tok.strip()}"
    return None


def _exec_compute(code: str, df: pd.DataFrame) -> pd.Series:
    """Run the proposal's compute(df) in a restricted namespace.

    Raises ValueError on any failure (compile, runtime, output validation).
    """
    ns: Dict[str, Any] = {"np": np, "pd": pd}
    try:
        compiled = compile(code, "<llm_proposal>", "exec")
    except SyntaxError as e:
        raise ValueError(f"syntax_error: {e}") from e
    exec(compiled, ns)  # noqa: S102 — gated upstream by _scan_code
    fn = ns.get("compute")
    if not callable(fn):
        raise ValueError("compute is not callable")
    try:
        out = fn(df)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"compute_raised: {type(e).__name__}: {e}") from e

    if not isinstance(out, pd.Series):
        raise ValueError(f"compute_returned_non_series: {type(out).__name__}")
    if len(out) != len(df):
        raise ValueError(f"length_mismatch: got {len(out)}, want {len(df)}")
    out = pd.to_numeric(out, errors="coerce").astype(float)
    nan_pct = float(out.isna().mean())
    if nan_pct > 0.20:
        raise ValueError(f"too_many_nans: {nan_pct:.2%} > 20%")
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _ablation_csv(rows: List[Dict[str, float]]) -> str:
    if not rows:
        return "(empty)"
    buf = io.StringIO()
    fields = ["group", "n_dropped", "n_remaining", "dpnl", "dbrier", "dsharpe", "dwin_rate", "n_trades"]
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    return buf.getvalue().rstrip()


def _build_user_message(
    ablation_rows: List[Dict[str, float]],
    baseline: Dict[str, float],
    schema_columns: List[str],
) -> str:
    return (
        "ABLATION TABLE (test window):\n"
        f"{_ablation_csv(ablation_rows)}\n\n"
        "BASELINE METRICS (test window):\n"
        f"{json.dumps(baseline, indent=2)}\n\n"
        f"SCHEMA COLUMNS ({len(schema_columns)} total) — your python_code may "
        "only reference these names:\n"
        f"{json.dumps(schema_columns)}\n\n"
        "Return strict JSON per the system schema."
    )


def _call_claude(
    system_prompt: str,
    user_message: str,
    model: str,
    max_proposals: int,
) -> Dict[str, Any]:
    """Call Claude with prompt caching on the system block."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.3,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Hard cap: at most {max_proposals} proposals per turn.",
            },
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    text_blocks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    raw = "".join(text_blocks).strip()
    # Strip code fences if model added them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"Claude returned non-JSON response (parse error: {e}):\n{raw[:2000]}"
        )


# ---------------------------------------------------------------------------
# History log
# ---------------------------------------------------------------------------


def _append_history(history_path: Path, record: Dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Phase 2.5 — Kill tests (cheap statistical gates BEFORE any retrain)
# ---------------------------------------------------------------------------


@dataclass
class KillThresholds:
    """Configurable thresholds for the kill-test phase.

    Defaults are calibrated to be tight enough that genuinely-noise features
    fail, but loose enough that legitimately-correlated features (e.g. a new
    momentum lag that's similar but not identical to an existing one) survive.
    """
    max_nan_frac: float = 0.05            # > 5% NaN on any split → kill
    max_corr_existing: float = 0.98       # |corr| with any existing feature
    max_corr_label: float = 0.95          # |corr| with label → leakage
    max_ks_stat: float = 0.20             # KS train vs test → distribution shift
    max_const_frac: float = 0.99          # > 99% identical values → useless


def _kill_tests(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    active_features: List[str],
    thresholds: KillThresholds,
) -> Tuple[bool, str, Dict[str, float]]:
    """Run cheap statistical gates BEFORE the sentinel/retrain phase.

    Returns ``(passed, reason, details)``. ``details`` always populated for
    audit logging (even on pass) so we can inspect what just barely cleared.

    Gates (in order, fail-fast):
      1. NaN frac > threshold on any split.
      2. Constant-or-near-constant values (single value covers > max_const_frac).
      3. |corr| with any existing active feature > max_corr_existing
         (computed on train_df; near-duplicates of existing features are useless).
      4. |corr| with label > max_corr_label (label leakage flag).
      5. KS-statistic between train and test distributions > max_ks_stat
         (the feature's distribution shifted; model trained on train won't
         generalize to test).
    """
    from scipy.stats import ks_2samp

    details: Dict[str, float] = {}
    train_arr = train_df[name].to_numpy(dtype=float)
    val_arr = val_df[name].to_numpy(dtype=float)
    test_arr = test_df[name].to_numpy(dtype=float)

    # --- 1. NaN frac on any split -----------------------------------------
    nan_train = float(np.isnan(train_arr).mean())
    nan_val = float(np.isnan(val_arr).mean())
    nan_test = float(np.isnan(test_arr).mean())
    details["nan_frac_train"] = nan_train
    details["nan_frac_val"] = nan_val
    details["nan_frac_test"] = nan_test
    worst_nan = max(nan_train, nan_val, nan_test)
    if worst_nan > thresholds.max_nan_frac:
        return False, (
            f"high_nan_frac: max={worst_nan:.3f} > {thresholds.max_nan_frac}"
        ), details

    # Replace NaN with 0 for downstream stats so single-NaN rows don't poison
    # corr / KS. Upstream `_exec_compute` already filled NaNs but be defensive.
    train_filled = np.where(np.isnan(train_arr), 0.0, train_arr)
    test_filled = np.where(np.isnan(test_arr), 0.0, test_arr)

    # --- 2. Near-constant values ------------------------------------------
    if len(train_filled) > 0:
        # Bin into ~1000 buckets and check if any single bucket has > threshold mass.
        # Robust to constant features and near-constant ones (e.g. all zeros except 5 rows).
        std = float(np.std(train_filled))
        if std == 0.0:
            details["const_frac"] = 1.0
            return False, "constant_feature: train std == 0", details
        # Check fraction at the modal value (rounded to 6 decimals to absorb fp noise).
        rounded = np.round(train_filled, 6)
        _, counts = np.unique(rounded, return_counts=True)
        modal_frac = float(counts.max()) / len(rounded)
        details["const_frac"] = modal_frac
        if modal_frac > thresholds.max_const_frac:
            return False, (
                f"near_constant: {modal_frac:.3f} of train rows share modal value"
            ), details

    # --- 3. |corr| with any existing active feature -----------------------
    # Skip features whose train slice has zero variance.
    max_existing_corr = 0.0
    worst_partner = ""
    train_var = float(np.var(train_filled))
    if train_var > 0:
        for f in active_features:
            if f == name:
                continue
            other = train_df[f].to_numpy(dtype=float)
            other = np.where(np.isnan(other), 0.0, other)
            if np.var(other) == 0.0:
                continue
            c = float(np.corrcoef(train_filled, other)[0, 1])
            if not np.isfinite(c):
                continue
            if abs(c) > abs(max_existing_corr):
                max_existing_corr = c
                worst_partner = f
    details["max_existing_corr"] = max_existing_corr
    details["max_existing_corr_partner"] = worst_partner  # type: ignore[assignment]
    if abs(max_existing_corr) > thresholds.max_corr_existing:
        return False, (
            f"correlated_with_existing: |corr({name},{worst_partner})|="
            f"{abs(max_existing_corr):.3f} > {thresholds.max_corr_existing}"
        ), details

    # --- 4. |corr| with label (leakage flag) ------------------------------
    label_corr = 0.0
    if "label" in train_df.columns and train_var > 0:
        y = train_df["label"].to_numpy(dtype=float)
        if np.var(y) > 0:
            c = float(np.corrcoef(train_filled, y)[0, 1])
            if np.isfinite(c):
                label_corr = c
    details["label_corr"] = label_corr
    if abs(label_corr) > thresholds.max_corr_label:
        return False, (
            f"label_leakage: |corr({name},label)|={abs(label_corr):.3f} "
            f"> {thresholds.max_corr_label}"
        ), details

    # --- 5. Train vs test distribution stability (KS) ---------------------
    # Skip if either side is constant.
    ks_stat = 0.0
    if np.var(train_filled) > 0 and np.var(test_filled) > 0:
        try:
            ks_stat = float(ks_2samp(train_filled, test_filled).statistic)
        except Exception:  # noqa: BLE001
            ks_stat = 0.0
    details["ks_train_test"] = ks_stat
    if ks_stat > thresholds.max_ks_stat:
        return False, (
            f"distribution_drift: ks(train,test)={ks_stat:.3f} > {thresholds.max_ks_stat}"
        ), details

    return True, "kill_tests_passed", details


# ---------------------------------------------------------------------------
# Phase 3 — addition loop
# ---------------------------------------------------------------------------


@dataclass
class ProposalResult:
    name: str
    accepted: bool
    reason: str
    metrics: Optional[Dict[str, float]] = None
    code_preview: str = ""
    kill_test_details: Optional[Dict[str, float]] = None


def _evaluate_proposal(
    proposal: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    active_features: List[str],
    baseline_metrics: Dict[str, float],
    fill_cfg: FillConfig,
    thresholds: KillThresholds,
) -> ProposalResult:
    """Validate, sentinel-gate, retrain, and promote-gate one proposal."""
    name = proposal.get("name", "")
    code = proposal.get("python_code", "")
    code_preview = (code or "")[:300]

    if not name or " " in name or not name.replace("_", "").isalnum():
        return ProposalResult(name, False, "invalid_name", code_preview=code_preview)
    if name in active_features:
        return ProposalResult(name, False, "name_collision_with_existing", code_preview=code_preview)
    bad = _scan_code(code)
    if bad:
        return ProposalResult(name, False, f"unsafe_code:{bad}", code_preview=code_preview)

    # Materialize the new column on all three splits.
    try:
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df[name] = _exec_compute(code, train_df).to_numpy()
        val_df[name] = _exec_compute(code, val_df).to_numpy()
        test_df[name] = _exec_compute(code, test_df).to_numpy()
    except ValueError as e:
        return ProposalResult(name, False, f"exec_failed:{e}", code_preview=code_preview)

    # Kill tests — cheap statistical gates BEFORE any XGB fit.
    passed, kill_reason, kill_details = _kill_tests(
        name, train_df, val_df, test_df, active_features, thresholds,
    )
    if not passed:
        return ProposalResult(
            name, False, f"kill_test:{kill_reason}",
            code_preview=code_preview, kill_test_details=kill_details,
        )

    # Sentinel gate: compare permutation-importance on val to a random sentinel
    # injected only for this check. We stay in-process (no subprocess to
    # feature_probe.py) to avoid serializing huge dataframes.
    rng = np.random.default_rng(0)
    sentinel_col = "__sentinel_random__"
    train_df[sentinel_col] = rng.standard_normal(len(train_df))
    val_df[sentinel_col] = rng.standard_normal(len(val_df))
    feats = active_features + [name, sentinel_col]
    try:
        importances = _quick_perm_importance(train_df, val_df, feats)
    except Exception as e:  # noqa: BLE001
        return ProposalResult(
            name, False, f"sentinel_failed:{type(e).__name__}",
            code_preview=code_preview, kill_test_details=kill_details,
        )
    if importances[name] <= importances[sentinel_col]:
        return ProposalResult(
            name, False,
            f"below_sentinel: imp={importances[name]:.5f} <= sentinel={importances[sentinel_col]:.5f}",
            code_preview=code_preview, kill_test_details=kill_details,
        )

    # Retrain + evaluate on test split.
    try:
        new_metrics = _fit_and_eval(
            train_df, val_df, test_df, active_features + [name], fill_cfg,
            label=f"add_{name}",
        )
    except Exception as e:  # noqa: BLE001
        return ProposalResult(
            name, False, f"retrain_failed:{type(e).__name__}",
            code_preview=code_preview, kill_test_details=kill_details,
        )
    new_dict = asdict(new_metrics)

    accepted, reason = promote_gate(new_dict, baseline_metrics)
    if not accepted:
        return ProposalResult(
            name, False, f"promote_gate:{reason}",
            new_dict, code_preview, kill_test_details=kill_details,
        )
    if new_dict["pnl"] < baseline_metrics["pnl"]:
        return ProposalResult(
            name, False,
            f"pnl_not_improved: {new_dict['pnl']:.2f} < {baseline_metrics['pnl']:.2f}",
            new_dict, code_preview, kill_test_details=kill_details,
        )
    return ProposalResult(
        name, True, reason, new_dict, code_preview, kill_test_details=kill_details,
    )


def _quick_perm_importance(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str]
) -> Dict[str, float]:
    """Tiny, fast XGB + permutation importance for sentinel gating."""
    from sklearn.inspection import permutation_importance
    from xgboost import XGBClassifier

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=float)
    X_val = val_df[features].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=float)
    model = XGBClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.05,
        eval_metric="logloss", early_stopping_rounds=20,
        tree_method="hist", verbosity=0, n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    result = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=0, n_jobs=-1,
        scoring="neg_brier_score",
    )
    return dict(zip(features, [float(v) for v in result.importances_mean]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-driven feature engineering loop")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run Phase 1 + 2, print Claude's JSON, exit. No retrain, no file writes.")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--synthetic-half-spread", type=float, default=0.01)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--history",
        default="data/feature_engineering_history.jsonl",
        help="Append history records to this JSONL.",
    )
    # Kill-test thresholds (Phase 2.5 — cheap gates before sentinel/retrain)
    parser.add_argument("--kill-max-nan-frac", type=float, default=0.05,
                        help="Kill if NaN fraction > X on any split. Default 0.05.")
    parser.add_argument("--kill-max-corr-existing", type=float, default=0.98,
                        help="Kill if abs corr with any existing feature > X.")
    parser.add_argument("--kill-max-corr-label", type=float, default=0.95,
                        help="Kill if abs corr with label > X (leakage flag).")
    parser.add_argument("--kill-max-ks-stat", type=float, default=0.20,
                        help="Kill if KS-statistic between train and test > X.")
    parser.add_argument("--kill-max-const-frac", type=float, default=0.99,
                        help="Kill if a single value covers > X fraction of train rows.")
    add_period_arguments(parser)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    args = parser.parse_args()

    thresholds = KillThresholds(
        max_nan_frac=args.kill_max_nan_frac,
        max_corr_existing=args.kill_max_corr_existing,
        max_corr_label=args.kill_max_corr_label,
        max_ks_stat=args.kill_max_ks_stat,
        max_const_frac=args.kill_max_const_frac,
    )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not getattr(args, "test_period", None):
        raise SystemExit("--test-period is REQUIRED.")

    # Fail fast on missing API key — before spending 5 minutes on Phase 1.
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Either:\n"
            "  • add ANTHROPIC_API_KEY=... to your .env file (auto-loaded), or\n"
            "  • export ANTHROPIC_API_KEY=... in your shell before running, or\n"
            "  • re-run with --dry-run to skip the LLM call entirely."
        )

    sel = load_selection(args.selection)
    df = pd.read_parquet(sel.probe_dataset)
    if df.empty:
        raise SystemExit("Probe dataset empty.")
    split = resolve_split_from_args(args, df, val_ratio=args.val_ratio)
    train_df = split.frames["training"]
    val_df = split.frames.get("validation")
    test_df = split.frames.get("test")
    if val_df is None or val_df.empty or test_df is None or test_df.empty:
        raise SystemExit("Need non-empty train/val/test splits.")

    fill_cfg = FillConfig(
        fee_bps=args.fee_bps,
        synthetic_half_spread=args.synthetic_half_spread,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history)
    ts_iso = datetime.now(timezone.utc).isoformat()

    # ---------- Phase 1: ablation ----------
    logging.info("=== PHASE 1: ABLATION (12 groups) ===")
    t0 = time.time()
    baseline, ablation_rows = run_ablation(
        train_df, val_df, test_df, sel.active_features, fill_cfg,
    )
    phase1_seconds = time.time() - t0
    logging.info("Phase 1 done in %.1fs", phase1_seconds)

    baseline_dict = asdict(baseline)
    if not args.dry_run:
        (out_dir / "baseline.json").write_text(json.dumps(baseline_dict, indent=2))
        with (out_dir / "ablation_results.csv").open("w", newline="") as f:
            fields = list(ablation_rows[0].keys()) if ablation_rows else []
            if fields:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in ablation_rows:
                    w.writerow(r)

    # ---------- Phase 2: LLM ----------
    logging.info("=== PHASE 2: LLM PROPOSAL (model=%s) ===", args.model)
    user_msg = _build_user_message(ablation_rows, baseline_dict, sel.active_features)
    if not args.dry_run:
        (out_dir / "phase2_user_prompt.txt").write_text(user_msg, encoding="utf-8")
    llm_out = _call_claude(_SYSTEM_PROMPT, user_msg, args.model, args.max_proposals)
    if not args.dry_run:
        (out_dir / "phase2_llm_response.json").write_text(
            json.dumps(llm_out, indent=2), encoding="utf-8"
        )
    logging.info(
        "LLM proposed %d feature(s); recommended dropping groups: %s",
        len(llm_out.get("proposals", [])), llm_out.get("drop_groups", []),
    )
    logging.info("LLM hypothesis: %s", llm_out.get("hypothesis", "(none)"))

    base_record = {
        "ts": ts_iso,
        "phase1_seconds": phase1_seconds,
        "selection_in": args.selection,
        "test_period": args.test_period,
        "baseline": baseline_dict,
        "drop_groups": llm_out.get("drop_groups", []),
        "n_proposals": len(llm_out.get("proposals", [])),
        "hypothesis": llm_out.get("hypothesis", ""),
    }

    if args.dry_run:
        logging.info("DRY RUN: skipping Phase 3 and all file writes (no history append).")
        print("\n=== DRY RUN: PHASE 1+2 RESULTS ===")
        print(f"Baseline: {json.dumps(baseline_dict, indent=2)}")
        print(f"\nLLM proposals ({len(llm_out.get('proposals', []))}):")
        print(json.dumps(llm_out, indent=2))
        return

    # ---------- Phase 3: addition loop ----------
    logging.info("=== PHASE 3: ADDITION LOOP ===")
    proposals = (llm_out.get("proposals") or [])[: args.max_proposals]
    results: List[Dict[str, Any]] = []
    accepted_proposals: List[Dict[str, Any]] = []
    cumulative_features = list(sel.active_features)

    for prop in proposals:
        logging.info("Evaluating proposal: %s", prop.get("name", "(unnamed)"))
        res = _evaluate_proposal(
            proposal=prop,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            active_features=cumulative_features,
            baseline_metrics=baseline_dict,
            fill_cfg=fill_cfg,
            thresholds=thresholds,
        )
        logging.info(
            "  → %s: %s",
            "ACCEPTED" if res.accepted else "REJECTED",
            res.reason,
        )
        results.append(asdict(res))
        if res.accepted:
            accepted_proposals.append(prop)
            # NOTE: we do NOT chain — each proposal is evaluated against the
            # original baseline, not the running-best. Chaining would require
            # the LLM to re-propose given the new model state. Combined eval
            # below catches the case where two correlated additions both pass
            # alone but regress together.

    accepted_features = [p["name"] for p in accepted_proposals]

    # Combined re-evaluation. Each proposal was scored against the ORIGINAL
    # baseline in isolation; correlated features can pass alone yet regress
    # together. Refit on the combined feature set and gate before publishing.
    combined_metrics: Optional[Dict[str, float]] = None
    combined_passed = True
    combined_reason = "single_proposal_no_combined_check_needed"
    if len(accepted_proposals) > 1:
        try:
            train_c, val_c, test_c = train_df.copy(), val_df.copy(), test_df.copy()
            for prop in accepted_proposals:
                code = prop.get("python_code", "")
                train_c[prop["name"]] = _exec_compute(code, train_c).to_numpy()
                val_c[prop["name"]] = _exec_compute(code, val_c).to_numpy()
                test_c[prop["name"]] = _exec_compute(code, test_c).to_numpy()
            combined_metrics_obj = _fit_and_eval(
                train_c, val_c, test_c,
                cumulative_features + accepted_features, fill_cfg,
                label="combined_accepted",
            )
            combined_metrics = asdict(combined_metrics_obj)
            combined_passed, combined_reason = promote_gate(combined_metrics, baseline_dict)
            if combined_passed and combined_metrics["pnl"] < baseline_dict["pnl"]:
                combined_passed = False
                combined_reason = (
                    f"combined_pnl_not_improved: {combined_metrics['pnl']:.2f} "
                    f"< baseline {baseline_dict['pnl']:.2f}"
                )
        except Exception as e:  # noqa: BLE001
            combined_passed = False
            combined_reason = f"combined_eval_failed:{type(e).__name__}:{e}"
        logging.info(
            "COMBINED EVAL (%d features): %s — %s",
            len(accepted_features),
            "PASSED" if combined_passed else "REJECTED",
            combined_reason,
        )

    # Materialize accepted columns into a new parquet so future training runs
    # are reproducible without re-executing the LLM proposals. Also persist
    # the python_code so a human can port them to the live feature builder.
    write_selection = bool(accepted_proposals) and combined_passed
    new_parquet_path: Optional[Path] = None
    formulas_path: Optional[Path] = None
    if write_selection:
        df_full = pd.read_parquet(sel.probe_dataset)
        for prop in accepted_proposals:
            df_full[prop["name"]] = _exec_compute(prop.get("python_code", ""), df_full).to_numpy()
        new_parquet_path = out_dir / "dataset_with_fe.parquet"
        df_full.to_parquet(new_parquet_path)

        formulas_path = out_dir / "feature_definitions.json"
        formulas = {p["name"]: p.get("python_code", "") for p in accepted_proposals}
        formulas_path.write_text(json.dumps(formulas, indent=2), encoding="utf-8")

        import yaml
        with open(args.selection, "r", encoding="utf-8") as f:
            raw_sel = yaml.safe_load(f)
        raw_sel["active_features"] = list(raw_sel["active_features"]) + accepted_features
        raw_sel["probe_dataset"] = str(new_parquet_path)
        raw_sel["run_id"] = f"{raw_sel.get('run_id', 'unnamed')}_fe_{ts_iso[:10]}"
        raw_sel.setdefault("rationale", "")
        raw_sel["rationale"] += (
            f"\n[fe_agent {ts_iso}] added {accepted_features}; "
            f"materialized columns into {new_parquet_path.name}; "
            f"formulas in {formulas_path.name} (port to live feature builder before deploying)"
        )
        out_sel = out_dir / "selection.yaml"
        with out_sel.open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw_sel, f, sort_keys=False)
        logging.info(
            "Wrote selection.yaml + %s + %s",
            new_parquet_path.name, formulas_path.name,
        )
    elif accepted_proposals and not combined_passed:
        logging.warning(
            "Combined set regressed — selection.yaml NOT written. "
            "Individually-accepted: %s. See history for combined metrics.",
            accepted_features,
        )

    _append_history(history_path, {
        **base_record,
        "phase_reached": 3,
        "accepted_features_individual": accepted_features,
        "combined_passed": combined_passed,
        "combined_reason": combined_reason,
        "combined_metrics": combined_metrics,
        "selection_written": write_selection,
        "results": results,
    })

    print("\n=== SUMMARY ===")
    print(f"Baseline PnL: {baseline_dict['pnl']:+.2f} | Brier: {baseline_dict['brier']:.4f}")
    print(f"Proposals tested: {len(proposals)} | Accepted individually: {len(accepted_features)}")
    for r in results:
        tag = "✓" if r["accepted"] else "✗"
        print(f"  {tag} {r['name']}: {r['reason']}")
    if len(accepted_proposals) > 1:
        verdict = "PASSED" if combined_passed else "REJECTED"
        print(f"Combined gate: {verdict} — {combined_reason}")
    if write_selection:
        print(f"Selection written: {out_dir / 'selection.yaml'}")
    elif accepted_proposals:
        print("Selection NOT written (combined gate failed). See history.")


if __name__ == "__main__":
    main()
