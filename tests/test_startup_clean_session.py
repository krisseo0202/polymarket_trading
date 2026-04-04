"""Tests for the clean-session prompt and file-clearing logic in startup.py."""

import os
import json
import tempfile
from io import StringIO
from unittest.mock import patch

import pytest

from src.utils.startup import _clear_session_files, _prompt_new_session
from src.engine.state_store import BotState, StateStore


# ---------------------------------------------------------------------------
# _clear_session_files
# ---------------------------------------------------------------------------

class TestClearSessionFiles:

    def test_removes_existing_targets(self, tmp_path):
        state = tmp_path / "bot_state.json"
        snap = tmp_path / "bot_state_snapshot.json"
        trades = tmp_path / "trades.jsonl"
        ticks = tmp_path / "ticks.jsonl"

        for f in (state, snap, trades, ticks):
            f.write_text("data")

        _clear_session_files(str(state), str(trades), str(ticks))

        assert not state.exists()
        assert not snap.exists()
        assert not trades.exists()
        assert not ticks.exists()

    def test_handles_missing_files_gracefully(self, tmp_path):
        state = str(tmp_path / "nonexistent_state.json")
        trades = str(tmp_path / "nonexistent_trades.jsonl")
        # Should not raise
        _clear_session_files(state, trades, None)

    def test_skips_none_paths(self, tmp_path):
        state = tmp_path / "bot_state.json"
        state.write_text("{}")
        # None paths are silently skipped
        _clear_session_files(str(state), None, None)
        assert not state.exists()

    def test_leaves_perf_db_untouched(self, tmp_path):
        state = tmp_path / "bot_state.json"
        perf = tmp_path / "perf.db"
        state.write_text("{}")
        perf.write_text("sqlite")

        _clear_session_files(str(state), None, None)

        assert perf.exists()  # never deleted

    def test_clears_snapshot_alongside_state(self, tmp_path):
        state = tmp_path / "bot_state.json"
        snap = tmp_path / "bot_state_snapshot.json"
        state.write_text("{}")
        snap.write_text("{}")

        _clear_session_files(str(state), None, None)

        assert not state.exists()
        assert not snap.exists()

    def test_partial_failure_does_not_raise(self, tmp_path):
        """If one file can't be deleted, the rest still get cleared."""
        state = tmp_path / "bot_state.json"
        trades = tmp_path / "trades.jsonl"
        state.write_text("{}")
        trades.write_text("trades")

        # Remove state first so it's "already gone", trades should still be cleared
        state.unlink()
        _clear_session_files(str(state), str(trades), None)

        assert not trades.exists()


# ---------------------------------------------------------------------------
# _prompt_new_session
# ---------------------------------------------------------------------------

class TestPromptNewSession:

    def test_returns_false_on_eof(self):
        with patch("builtins.input", side_effect=EOFError):
            assert _prompt_new_session() is False

    def test_returns_false_on_empty_enter(self):
        with patch("builtins.input", return_value=""):
            assert _prompt_new_session() is False

    def test_returns_false_on_resume(self):
        with patch("builtins.input", return_value="resume"):
            assert _prompt_new_session() is False

    def test_returns_false_on_r(self):
        with patch("builtins.input", return_value="r"):
            assert _prompt_new_session() is False

    def test_returns_true_on_clean(self):
        with patch("builtins.input", return_value="clean"):
            assert _prompt_new_session() is True

    def test_returns_true_on_c(self):
        with patch("builtins.input", return_value="c"):
            assert _prompt_new_session() is True

    def test_loops_on_invalid_then_accepts_clean(self):
        inputs = iter(["invalid", "what", "clean"])
        with patch("builtins.input", side_effect=inputs):
            assert _prompt_new_session() is True


# ---------------------------------------------------------------------------
# StateStore returns fresh BotState after clean delete
# ---------------------------------------------------------------------------

class TestStateStoreAfterClean:

    def test_fresh_state_when_file_deleted(self, tmp_path):
        state_file = str(tmp_path / "bot_state.json")
        store = StateStore(path=state_file)

        # Write some state
        state = BotState()
        state.cycle_count = 42
        state.daily_realized_pnl = 3.14
        store.save(state)

        # Simulate clean session
        _clear_session_files(state_file, None, None)

        # Load should give a fresh default state
        loaded = store.load()
        assert loaded.cycle_count == 0
        assert loaded.daily_realized_pnl == 0.0
        assert loaded.inventories == {}

    def test_state_preserved_when_not_cleaned(self, tmp_path):
        state_file = str(tmp_path / "bot_state.json")
        store = StateStore(path=state_file)

        state = BotState()
        state.cycle_count = 99
        store.save(state)

        loaded = store.load()
        assert loaded.cycle_count == 99
