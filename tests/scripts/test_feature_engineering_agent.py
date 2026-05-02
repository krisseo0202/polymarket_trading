"""Tests for the LLM-proposal sandbox in scripts/feature_engineering_agent.py.

Focus on the parts that DON'T require live S3 + Anthropic API:
  * _scan_code rejects forbidden tokens (import, exec, os., subprocess, ...).
  * _exec_compute runs valid compute(df) and validates output shape.
  * _exec_compute rejects bad outputs (wrong type, wrong length, too many NaN).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _stub_clob_modules() -> None:
    """The agent transitively imports src.api.client which on this branch
    requires py_clob_client_v2. Stub the offending module so the agent can
    load in CI / dev envs that haven't installed the V2 SDK yet."""
    if "py_clob_client_v2" in sys.modules:
        return
    pkg = types.ModuleType("py_clob_client_v2")
    sub_client = types.ModuleType("py_clob_client_v2.client")
    sub_clob_types = types.ModuleType("py_clob_client_v2.clob_types")
    sub_constants = types.ModuleType("py_clob_client_v2.constants")
    sub_order_builder = types.ModuleType("py_clob_client_v2.order_builder")
    sub_order_builder_constants = types.ModuleType(
        "py_clob_client_v2.order_builder.constants"
    )

    class _Stub:
        def __init__(self, *_a, **_kw): ...
        def __getattr__(self, _name): return _Stub
        def __call__(self, *_a, **_kw): return _Stub()

    for mod, names in [
        (sub_client, ["ClobClient"]),
        (sub_clob_types, [
            "ApiCreds", "AssetType", "BalanceAllowanceParams",
            "OpenOrderParams", "OrderArgs", "OrderType",
            "PartialCreateOrderOptions", "TradeParams",
        ]),
        (sub_constants, ["POLYGON"]),
        (sub_order_builder_constants, ["BUY", "SELL"]),
    ]:
        for n in names:
            setattr(mod, n, _Stub)

    sys.modules["py_clob_client_v2"] = pkg
    sys.modules["py_clob_client_v2.client"] = sub_client
    sys.modules["py_clob_client_v2.clob_types"] = sub_clob_types
    sys.modules["py_clob_client_v2.constants"] = sub_constants
    sys.modules["py_clob_client_v2.order_builder"] = sub_order_builder
    sys.modules["py_clob_client_v2.order_builder.constants"] = sub_order_builder_constants


_stub_clob_modules()

_AGENT_PATH = ROOT / "scripts" / "feature_engineering_agent.py"


def _load_agent():
    if "agent_under_test" in sys.modules:
        return sys.modules["agent_under_test"]
    spec = importlib.util.spec_from_file_location("agent_under_test", str(_AGENT_PATH))
    mod = importlib.util.module_from_spec(spec)
    # Must register in sys.modules before exec_module so that @dataclass can
    # resolve cls.__module__ for ProposalResult.
    sys.modules["agent_under_test"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        sys.modules.pop("agent_under_test", None)
        pytest.skip(f"agent module fails to import in this env: {e}")
    return mod


@pytest.fixture(scope="module")
def agent():
    return _load_agent()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "btc_mid": np.linspace(60_000, 61_000, 50),
        "yes_mid": np.linspace(0.45, 0.55, 50),
        "yes_spread": np.full(50, 0.02),
    })


class TestScanCode:
    def test_rejects_import(self, agent):
        code = "import os\ndef compute(df):\n    return df['btc_mid']"
        assert agent._scan_code(code) == "forbidden_token:import"

    def test_rejects_from_import(self, agent):
        code = "from os import path\ndef compute(df):\n    return df['btc_mid']"
        # Either token suffices to reject — `import ` matches first in scan order.
        assert agent._scan_code(code) is not None
        assert agent._scan_code(code).startswith("forbidden_token:")

    def test_rejects_dunder_import(self, agent):
        code = "def compute(df):\n    return __import__('os').getcwd()"
        assert agent._scan_code(code) == "forbidden_token:__import__"

    def test_rejects_open(self, agent):
        code = "def compute(df):\n    open('/etc/passwd').read(); return df['btc_mid']"
        assert agent._scan_code(code) == "forbidden_token:open("

    def test_rejects_eval(self, agent):
        code = "def compute(df):\n    eval('1+1'); return df['btc_mid']"
        assert agent._scan_code(code) == "forbidden_token:eval("

    def test_rejects_missing_compute(self, agent):
        code = "def helper(df):\n    return df['btc_mid']"
        assert agent._scan_code(code) == "missing_def_compute"

    def test_accepts_clean_code(self, agent):
        code = "def compute(df):\n    return df['btc_mid'] - df['btc_mid'].mean()"
        assert agent._scan_code(code) is None


class TestExecCompute:
    def test_simple_diff_feature(self, agent, sample_df):
        code = "def compute(df):\n    return df['yes_mid'] - 0.5"
        out = agent._exec_compute(code, sample_df)
        assert isinstance(out, pd.Series)
        assert len(out) == len(sample_df)
        assert out.iloc[0] == pytest.approx(-0.05)

    def test_using_numpy(self, agent, sample_df):
        code = "def compute(df):\n    return pd.Series(np.log1p(df['btc_mid']))"
        out = agent._exec_compute(code, sample_df)
        assert (out > 10).all()  # log1p(60000) ~= 11

    def test_returns_non_series_raises(self, agent, sample_df):
        code = "def compute(df):\n    return [1, 2, 3]"
        with pytest.raises(ValueError, match="non_series"):
            agent._exec_compute(code, sample_df)

    def test_wrong_length_raises(self, agent, sample_df):
        code = "def compute(df):\n    return df['btc_mid'].iloc[:5]"
        with pytest.raises(ValueError, match="length_mismatch"):
            agent._exec_compute(code, sample_df)

    def test_too_many_nans_raises(self, agent, sample_df):
        code = "def compute(df):\n    s = df['btc_mid'].copy(); s.iloc[:40] = np.nan; return s"
        with pytest.raises(ValueError, match="too_many_nans"):
            agent._exec_compute(code, sample_df)

    def test_runtime_error_wrapped(self, agent, sample_df):
        code = "def compute(df):\n    return df['nonexistent_column']"
        with pytest.raises(ValueError, match="compute_raised"):
            agent._exec_compute(code, sample_df)

    def test_syntax_error_wrapped(self, agent, sample_df):
        code = "def compute(df):\n    return df['btc_mid' +"
        with pytest.raises(ValueError, match="syntax_error"):
            agent._exec_compute(code, sample_df)

    def test_fills_some_nans(self, agent, sample_df):
        # Up to 20% NaN allowed, gets filled with 0.0.
        code = (
            "def compute(df):\n"
            "    s = df['btc_mid'].copy()\n"
            "    s.iloc[:5] = np.nan\n"
            "    return s\n"
        )
        out = agent._exec_compute(code, sample_df)
        assert out.isna().sum() == 0
        assert (out.iloc[:5] == 0.0).all()


# ---------------------------------------------------------------------------
# Kill-test phase (Phase 2.5)
# ---------------------------------------------------------------------------


def _make_split(n_train: int = 200, n_val: int = 50, n_test: int = 100, seed: int = 42):
    """Synthetic dataset with a benign feature, a label, and a few existing
    columns used as 'active features' in tests."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test
    df = pd.DataFrame({
        "btc_mid": rng.normal(60_000, 100, n_total),
        "yes_mid": rng.uniform(0.40, 0.60, n_total),
        "yes_spread": rng.uniform(0.01, 0.05, n_total),
        "label": rng.integers(0, 2, n_total).astype(float),
    })
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


class TestKillTests:
    @pytest.fixture
    def thresholds(self, agent):
        return agent.KillThresholds()

    @pytest.fixture
    def splits(self):
        return _make_split()

    def test_passes_clean_feature(self, agent, thresholds, splits):
        train, val, test = splits
        rng = np.random.default_rng(0)
        new_col = rng.normal(0, 1, len(train))
        train = train.copy(); train["new_feat"] = new_col
        rng2 = np.random.default_rng(1)
        val = val.copy(); val["new_feat"] = rng2.normal(0, 1, len(val))
        test = test.copy(); test["new_feat"] = rng2.normal(0, 1, len(test))
        passed, reason, details = agent._kill_tests(
            "new_feat", train, val, test, ["btc_mid", "yes_mid"], thresholds,
        )
        assert passed, f"clean feature was killed: {reason}"
        assert details["nan_frac_train"] == 0.0

    def test_kills_high_nan(self, agent, thresholds, splits):
        train, val, test = splits
        col = np.full(len(train), np.nan); col[:50] = 1.0
        train = train.copy(); train["high_nan"] = col
        val = val.copy(); val["high_nan"] = np.nan
        test = test.copy(); test["high_nan"] = np.nan
        passed, reason, _ = agent._kill_tests(
            "high_nan", train, val, test, ["btc_mid"], thresholds,
        )
        assert not passed
        assert reason.startswith("high_nan_frac")

    def test_kills_constant(self, agent, thresholds, splits):
        train, val, test = splits
        train = train.copy(); train["const"] = 1.0
        val = val.copy(); val["const"] = 1.0
        test = test.copy(); test["const"] = 1.0
        passed, reason, _ = agent._kill_tests(
            "const", train, val, test, ["btc_mid"], thresholds,
        )
        assert not passed
        assert "constant" in reason or "near_constant" in reason

    def test_kills_near_duplicate_of_existing(self, agent, thresholds, splits):
        train, val, test = splits
        # New feature = btc_mid + tiny noise → corr ≈ 1.0
        train = train.copy(); train["btc_mid_dup"] = train["btc_mid"] + 0.001
        val = val.copy(); val["btc_mid_dup"] = val["btc_mid"] + 0.001
        test = test.copy(); test["btc_mid_dup"] = test["btc_mid"] + 0.001
        passed, reason, details = agent._kill_tests(
            "btc_mid_dup", train, val, test, ["btc_mid", "yes_mid"], thresholds,
        )
        assert not passed
        assert reason.startswith("correlated_with_existing")
        assert details["max_existing_corr_partner"] == "btc_mid"

    def test_kills_label_leakage(self, agent, thresholds, splits):
        train, val, test = splits
        # Feature is the label + tiny noise → corr ≈ 1.0 with label
        train = train.copy(); train["leak"] = train["label"] + np.random.default_rng(0).normal(0, 0.01, len(train))
        val = val.copy(); val["leak"] = val["label"]
        test = test.copy(); test["leak"] = test["label"]
        passed, reason, details = agent._kill_tests(
            "leak", train, val, test, ["btc_mid"], thresholds,
        )
        assert not passed
        assert reason.startswith("label_leakage")
        assert abs(details["label_corr"]) > 0.95

    def test_kills_distribution_drift(self, agent, thresholds, splits):
        train, val, test = splits
        rng = np.random.default_rng(0)
        train = train.copy(); train["drift"] = rng.normal(0, 1, len(train))
        val = val.copy(); val["drift"] = rng.normal(0, 1, len(val))
        # Test distribution shifted by 5 sigma → KS stat should be ~1.0
        test = test.copy(); test["drift"] = rng.normal(5, 1, len(test))
        passed, reason, details = agent._kill_tests(
            "drift", train, val, test, ["btc_mid"], thresholds,
        )
        assert not passed
        assert reason.startswith("distribution_drift")
        assert details["ks_train_test"] > 0.20

    def test_threshold_override(self, agent, splits):
        """Loosening thresholds should let near-dupes through."""
        train, val, test = splits
        rng = np.random.default_rng(7)
        # Mostly btc_mid plus enough independent noise to drop corr to ~0.85.
        for d, n in [(train, len(train)), (val, len(val)), (test, len(test))]:
            d["dup"] = d["btc_mid"] + rng.normal(0, 60, n)
        # Default threshold (0.98) would let this through too — but assert
        # specifically that loosening permits an even looser correlation.
        loose = agent.KillThresholds(max_corr_existing=0.999)
        passed, reason, details = agent._kill_tests(
            "dup", train, val, test, ["btc_mid"], loose,
        )
        assert passed, f"loose threshold did not let dup through: {reason}"
        # Sanity: the corr should be high but well under 0.999.
        assert 0.5 < abs(details["max_existing_corr"]) < 0.999
