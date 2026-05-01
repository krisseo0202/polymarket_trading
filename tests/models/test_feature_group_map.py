"""Tests for src.models.feature_group_map.

Validates that the group map covers FEATURE_COLUMNS exactly (no missing, no
extra, no duplicates) and that the public helpers behave.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Load schema.py without going through src.models.__init__ which transitively
# imports the (potentially-missing) py_clob_client_v2 package on this branch.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

_FGM_SPEC = importlib.util.spec_from_file_location(
    "fgm_under_test", str(ROOT / "src" / "models" / "feature_group_map.py")
)
fgm = importlib.util.module_from_spec(_FGM_SPEC)
_FGM_SPEC.loader.exec_module(fgm)


def _load_schema_columns():
    """Stub multi_tf_features to avoid importing the broken api/client chain."""
    src_pkg = types.ModuleType("src")
    models_pkg = types.ModuleType("src.models")
    multi_tf_stub = types.ModuleType("src.models.multi_tf_features")
    multi_tf_stub.multi_tf_feature_names = fgm._multi_tf_names
    sys.modules["src"] = src_pkg
    sys.modules["src.models"] = models_pkg
    sys.modules["src.models.multi_tf_features"] = multi_tf_stub
    spec = importlib.util.spec_from_file_location(
        "src.models.schema", str(ROOT / "src" / "models" / "schema.py")
    )
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    return list(schema.FEATURE_COLUMNS)


class TestGroupMapStructure:
    def test_twelve_groups(self):
        assert set(fgm.FEATURE_GROUPS.keys()) == {
            "btc_momentum", "btc_structure", "btc_microstructure",
            "ob_basic", "ob_depth", "ob_coherence", "slot_path", "derived",
            "indicators", "calendar", "recent_outcomes", "multi_tf",
        }

    def test_no_duplicates_within_groups(self):
        for g, members in fgm.FEATURE_GROUPS.items():
            assert len(members) == len(set(members)), f"duplicates in {g}"

    def test_no_overlap_across_groups(self):
        seen = {}
        for g, members in fgm.FEATURE_GROUPS.items():
            for m in members:
                assert m not in seen, (
                    f"feature {m!r} in both {seen[m]!r} and {g!r}"
                )
                seen[m] = g


class TestSchemaCoverage:
    def test_validate_against_real_schema(self):
        cols = _load_schema_columns()
        # Should not raise.
        fgm.validate_against(cols)

    def test_validate_against_extra_feature_raises(self):
        cols = _load_schema_columns() + ["bogus_extra_feature"]
        with pytest.raises(ValueError, match="missing from groups"):
            fgm.validate_against(cols)

    def test_validate_against_dropped_feature_raises(self):
        cols = _load_schema_columns()[:-1]  # drop the last one
        with pytest.raises(ValueError, match="in groups but not in schema"):
            fgm.validate_against(cols)


class TestPublicHelpers:
    def test_group_of_known_feature(self):
        assert fgm.group_of("rsi_14") == "indicators"
        assert fgm.group_of("hour_sin") == "calendar"
        assert fgm.group_of("ut_trend_disagreement") == "multi_tf"

    def test_group_of_unknown_raises(self):
        with pytest.raises(KeyError):
            fgm.group_of("does_not_exist")

    def test_features_in_group(self):
        members = fgm.features_in_group("calendar")
        assert set(members) == {"hour_sin", "hour_cos", "is_weekend"}

    def test_features_in_unknown_group_raises(self):
        with pytest.raises(KeyError):
            fgm.features_in_group("not_a_group")
