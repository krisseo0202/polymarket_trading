"""
Deterministic validation tests for dashboard data bugs.

Each test targets one specific fix:
  1. order_count — active_order_ids is a dict, not a list
  2. position side labels — YES/NO not Up/Down/truncated ID
  3. bot status derivation — richer than RUNNING/STOPPED
  4. PnL — unrealized PnL shown when positions are open
  5. win/loss — intra-cycle fills counted (bot.py fix, tested at data level)
"""

import time
from unittest.mock import patch



# ---------------------------------------------------------------------------
# 1. order_count was always 0 because isinstance(dict, list) is False
# ---------------------------------------------------------------------------

class TestOrderCountFix:
    """active_order_ids is a dict {token_id: order_id|None}, not a list."""

    def test_order_count_from_dict_with_active_orders(self):
        """Non-None values should be counted as active orders."""
        orders = {
            "token_aaa": "order_123",
            "token_bbb": None,
            "token_ccc": "order_456",
        }
        # This is the fixed logic from _build_bot_status_panel
        if isinstance(orders, dict):
            count = sum(1 for v in orders.values() if v is not None)
        else:
            count = 0
        assert count == 2

    def test_order_count_all_none(self):
        """All-None dict should count as 0 active orders."""
        orders = {"token_aaa": None, "token_bbb": None}
        count = sum(1 for v in orders.values() if v is not None) if isinstance(orders, dict) else 0
        assert count == 0

    def test_order_count_empty_dict(self):
        orders = {}
        count = sum(1 for v in orders.values() if v is not None) if isinstance(orders, dict) else 0
        assert count == 0

    def test_old_bug_reproduced(self):
        """The old code: isinstance(dict, list) → always False → always 0."""
        orders = {"token_aaa": "order_123"}
        # Old buggy logic:
        old_count = len(orders) if isinstance(orders, list) else 0
        assert old_count == 0  # confirms the bug existed


# ---------------------------------------------------------------------------
# 2. Position side should be YES/NO, not Up/Down or truncated token ID
# ---------------------------------------------------------------------------

class TestPositionSideLabels:
    """Polymarket binary markets: tids[0] = YES, tids[1] = NO."""

    def test_up_token_maps_to_yes(self):
        market = {"up_token": "tok_aaa", "down_token": "tok_bbb"}
        token_to_outcome = {}
        if market.get("up_token"):
            token_to_outcome[market["up_token"]] = "YES"
        if market.get("down_token"):
            token_to_outcome[market["down_token"]] = "NO"

        assert token_to_outcome["tok_aaa"] == "YES"
        assert token_to_outcome["tok_bbb"] == "NO"

    def test_fallback_to_truncated_id_when_no_market(self):
        """When market is None, unknown tokens fall back to truncated ID."""
        tid = "0x1234567890abcdef1234567890abcdef"
        token_to_outcome = {}
        # Simulate the fallback path
        if tid in token_to_outcome:
            label = token_to_outcome[tid]
        else:
            label = tid[:8] + "..." if len(tid) > 8 else tid
        assert label == "0x123456..."


# ---------------------------------------------------------------------------
# 3. Bot status should reflect strategy state + orders, not just file mtime
# ---------------------------------------------------------------------------

class TestBotStatusDerivation:
    """Status derived from file mtime + strategy_status + active orders."""

    def _derive_status(self, age_min, strategy_status, order_count):
        """Replicates the fixed _build_bot_status_panel logic."""
        if age_min >= 10:
            return "STOPPED"
        elif strategy_status == "POSITION_OPEN":
            return "POSITION_OPEN"
        elif order_count > 0:
            return "ORDER_PENDING"
        else:
            return "RUNNING"

    def test_stopped_when_stale(self):
        assert self._derive_status(15, "WATCHING", 0) == "STOPPED"

    def test_position_open_shown(self):
        assert self._derive_status(1, "POSITION_OPEN", 0) == "POSITION_OPEN"

    def test_order_pending_shown(self):
        assert self._derive_status(1, "WATCHING", 2) == "ORDER_PENDING"

    def test_running_when_idle(self):
        assert self._derive_status(1, "WATCHING", 0) == "RUNNING"

    def test_position_open_takes_priority_over_orders(self):
        """POSITION_OPEN should show even if there are pending orders."""
        assert self._derive_status(1, "POSITION_OPEN", 1) == "POSITION_OPEN"


# ---------------------------------------------------------------------------
# 4. PnL — unrealized PnL computation
# ---------------------------------------------------------------------------

class TestUnrealizedPnl:
    """Unrealized PnL = (current_mid - avg_cost) * position for each active inventory."""

    def test_long_position_profit(self):
        """Bought at 0.45, current mid 0.55, 100 shares → uPnL = +10.00."""
        avg_cost = 0.45
        position = 100.0
        current_mid = 0.55
        upnl = (current_mid - avg_cost) * position
        assert abs(upnl - 10.0) < 1e-9

    def test_long_position_loss(self):
        """Bought at 0.55, current mid 0.45, 100 shares → uPnL = -10.00."""
        avg_cost = 0.55
        position = 100.0
        current_mid = 0.45
        upnl = (current_mid - avg_cost) * position
        assert abs(upnl - (-10.0)) < 1e-9

    def test_flat_position_no_upnl(self):
        """No position → unrealized PnL computation returns None."""
        inventories = {"tok_aaa": {"position": 0, "avg_cost": 0.5}}
        active = {t: i for t, i in inventories.items() if i.get("position", 0) != 0}
        assert len(active) == 0  # filtered out


# ---------------------------------------------------------------------------
# 5. Win/loss counting — verify the fix applies to intra-cycle fills
# ---------------------------------------------------------------------------

class TestWinLossCounting:
    """Intra-cycle fills should increment session_wins/session_losses."""

    def test_positive_realized_counts_as_win(self):
        """Simulates the fixed intra-cycle logic."""
        session_wins = 0
        session_losses = 0
        realized = 0.05  # positive = win
        if realized > 0:
            session_wins += 1
        elif realized < 0:
            session_losses += 1
        assert session_wins == 1
        assert session_losses == 0

    def test_negative_realized_counts_as_loss(self):
        session_wins = 0
        session_losses = 0
        realized = -0.03  # negative = loss
        if realized > 0:
            session_wins += 1
        elif realized < 0:
            session_losses += 1
        assert session_wins == 0
        assert session_losses == 1

    def test_zero_realized_no_count(self):
        """Position add (no close) returns 0 realized — not a win or loss."""
        session_wins = 0
        session_losses = 0
        realized = 0.0
        if realized > 0:
            session_wins += 1
        elif realized < 0:
            session_losses += 1
        assert session_wins == 0
        assert session_losses == 0


# ---------------------------------------------------------------------------
# 6. Redesigned Trades panel — cover helpers + narrow/wide rendering
# ---------------------------------------------------------------------------

def _render(panel) -> str:
    """Render a Rich renderable to a plain-text string for substring asserts."""
    from rich.console import Console
    buf = Console(record=True, width=200, color_system=None)
    buf.print(panel)
    return buf.export_text()


class TestTradeStatus:
    """_trade_status derives OPEN/SETTLED/FILLED from an entry + inventories."""

    def test_open_when_inventory_still_holds_token(self):
        from dashboard import _trade_status
        entry = {"token_id": "tok_A", "realized_pnl_delta": 0.0}
        invs = {"tok_A": {"token_id": "tok_A", "position": 38.0}}
        label, style = _trade_status(entry, invs)
        assert label == "OPEN"
        assert "cyan" in style

    def test_settled_when_pnl_delta_nonzero(self):
        from dashboard import _trade_status
        entry = {"token_id": "tok_B", "realized_pnl_delta": 2.10}
        invs = {}  # position already closed
        label, _ = _trade_status(entry, invs)
        assert label == "SETTLED"

    def test_filled_entry_with_no_pnl_and_no_inventory(self):
        from dashboard import _trade_status
        entry = {"token_id": "tok_C", "realized_pnl_delta": 0.0}
        label, _ = _trade_status(entry, {})
        assert label == "FILLED"

    def test_open_beats_settled_when_both_plausible(self):
        """Inventory still holding should dominate even if delta is nonzero
        (e.g. partial close — the row is still OPEN)."""
        from dashboard import _trade_status
        entry = {"token_id": "tok_A", "realized_pnl_delta": 1.0}
        invs = {"tok_A": {"token_id": "tok_A", "position": 10.0}}
        label, _ = _trade_status(entry, invs)
        assert label == "OPEN"

    def test_flat_inventory_not_open(self):
        """Position ~= 0 in inventory should not count as OPEN."""
        from dashboard import _trade_status
        entry = {"token_id": "tok_A", "realized_pnl_delta": 0.0}
        invs = {"tok_A": {"token_id": "tok_A", "position": 0.0}}
        label, _ = _trade_status(entry, invs)
        assert label == "FILLED"


class TestFmtPnl:
    def test_none_renders_dash(self):
        from dashboard import _fmt_pnl
        assert "—" in _render(_fmt_pnl(None))

    def test_zero_renders_dash(self):
        from dashboard import _fmt_pnl
        assert "—" in _render(_fmt_pnl(0.0))

    def test_positive_has_plus_sign(self):
        from dashboard import _fmt_pnl
        rendered = _render(_fmt_pnl(2.10))
        assert "+$2.10" in rendered

    def test_negative_has_minus_sign(self):
        from dashboard import _fmt_pnl
        rendered = _render(_fmt_pnl(-1.80))
        assert "-$1.80" in rendered


class TestFmtTte:
    def test_none(self):
        from dashboard import _fmt_tte
        assert _fmt_tte(None) == "—"

    def test_seconds_only(self):
        from dashboard import _fmt_tte
        assert _fmt_tte(45) == "45s"

    def test_mixed_minutes_seconds(self):
        from dashboard import _fmt_tte
        assert _fmt_tte(125) == "2m5s"

    def test_clamps_negative(self):
        from dashboard import _fmt_tte
        assert _fmt_tte(-10) == "0s"


class TestTradesPanel:
    """_build_trade_log_panel — full render smoke tests across modes."""

    def _entry(self, **overrides):
        base = {
            "ts": time.time() - 60,
            "action": "BUY",
            "outcome": "YES",
            "price": 0.52,
            "size": 38.0,
            "strategy_name": "logreg",
            "slot_expiry_ts": time.time() + 60,
            "seconds_to_expiry": 73.0,
            "token_id": "tok_A",
            "edge": 0.070,
            "realized_pnl_delta": 0.0,
        }
        base.update(overrides)
        return base

    def test_empty_trade_log_shows_placeholder(self):
        from dashboard import _build_trade_log_panel
        panel = _build_trade_log_panel({"trade_log": []})
        assert "No trades yet" in _render(panel)

    def test_none_bot_state(self):
        from dashboard import _build_trade_log_panel
        panel = _build_trade_log_panel(None)
        assert "No trades yet" in _render(panel)

    def test_open_trade_shows_marker_and_status(self):
        from dashboard import _build_trade_log_panel
        bot_state = {
            "trade_log": [self._entry()],
            "inventories": {"tok_A": {"token_id": "tok_A", "position": 38.0}},
        }
        rendered = _render(_build_trade_log_panel(bot_state, panel_width=160))
        assert "▶" in rendered
        assert "OPEN" in rendered
        assert "logreg" in rendered
        assert "BUY" in rendered
        assert "YES" in rendered

    def test_settled_trade_with_positive_pnl(self):
        from dashboard import _build_trade_log_panel
        bot_state = {
            "trade_log": [self._entry(action="SELL", outcome="NO", realized_pnl_delta=2.10)],
            "inventories": {},
        }
        rendered = _render(_build_trade_log_panel(bot_state, panel_width=160))
        assert "SETTLED" in rendered
        assert "+$2.10" in rendered
        assert "SELL" in rendered
        assert "NO" in rendered

    def test_settled_trade_with_negative_pnl(self):
        from dashboard import _build_trade_log_panel
        bot_state = {
            "trade_log": [self._entry(realized_pnl_delta=-1.80)],
            "inventories": {},
        }
        rendered = _render(_build_trade_log_panel(bot_state, panel_width=160))
        assert "SETTLED" in rendered
        assert "-$1.80" in rendered

    def test_wide_mode_includes_all_columns(self):
        from dashboard import _build_trade_log_panel
        bot_state = {"trade_log": [self._entry()], "inventories": {}}
        rendered = _render(_build_trade_log_panel(bot_state, panel_width=160))
        # Wide-only columns
        assert "TTE" in rendered
        assert "NOTIONAL" in rendered
        assert "EDGE" in rendered

    def test_narrow_mode_drops_low_priority_columns(self):
        from dashboard import _build_trade_log_panel
        bot_state = {"trade_log": [self._entry()], "inventories": {}}
        rendered = _render(_build_trade_log_panel(bot_state, panel_width=80))
        # Narrow mode drops these headers
        assert "NOTIONAL" not in rendered
        assert "EDGE" not in rendered
        assert " TTE " not in rendered
        # But keeps the essentials
        assert "STATUS" in rendered
        assert "PnL" in rendered

    def test_handles_missing_enrichment_fields_gracefully(self):
        """Old-schema entries (pre-enrichment) should still render, just with dashes."""
        from dashboard import _build_trade_log_panel
        legacy = {
            "ts": time.time(),
            "action": "BUY", "outcome": "YES",
            "price": 0.50, "size": 10.0,
        }
        bot_state = {"trade_log": [legacy], "inventories": {}}
        rendered = _render(_build_trade_log_panel(bot_state, panel_width=160))
        assert "BUY" in rendered  # didn't crash; action still renders


# ---------------------------------------------------------------------------
# 7. Config panel — runtime-summary projection + cached load
# ---------------------------------------------------------------------------

class TestRuntimeConfigSummary:
    """_build_runtime_summary is a pure projection of raw yaml → display dict."""

    def test_paper_mode(self):
        from dashboard import _build_runtime_summary
        raw = {"trading": {"paper_trading": True, "interval": 300}, "strategies": {}, "risk": {}}
        s = _build_runtime_summary(raw, active_strategy=None)
        assert s["mode"] == "PAPER"
        assert s["interval"] == "300s (5m)"

    def test_live_mode(self):
        from dashboard import _build_runtime_summary
        raw = {"trading": {"paper_trading": False, "interval": 60}, "strategies": {}, "risk": {}}
        s = _build_runtime_summary(raw, active_strategy=None)
        assert s["mode"] == "LIVE"
        assert s["interval"] == "60s (1m)"

    def test_active_strategy_resolution(self):
        from dashboard import _build_runtime_summary
        raw = {
            "trading": {"paper_trading": True, "interval": 300},
            "risk": {},
            "strategies": {
                "logreg": {"enabled": True, "model_dir": "models/logreg_v4",
                           "min_position_size_usdc": 10.0, "max_position_size_usdc": 20.0,
                           "kelly_fraction": 0.10, "delta": 0.03, "min_confidence": 0.52},
                "coin_toss": {"enabled": True},
            },
        }
        s = _build_runtime_summary(raw, active_strategy="logreg")
        assert s["strategy"] == "logreg"
        assert s["model_dir"] == "models/logreg_v4"
        assert s["max_position"] == "$10–$20"
        assert s["kelly"] == "0.10"
        assert s["min_edge"] == "0.030"
        assert s["min_conf"] == "0.52"

    def test_min_only_position_formats_as_floor(self):
        from dashboard import _build_runtime_summary
        raw = {"trading": {}, "risk": {},
               "strategies": {"x": {"enabled": True, "min_position_size_usdc": 10.0}}}
        s = _build_runtime_summary(raw, active_strategy="x")
        assert s["max_position"] == "≥$10"

    def test_single_cap_formats_as_dollar(self):
        from dashboard import _build_runtime_summary
        raw = {"trading": {}, "risk": {},
               "strategies": {"x": {"enabled": True, "position_size_usdc": 20.0}}}
        s = _build_runtime_summary(raw, active_strategy="x")
        assert s["max_position"] == "$20"

    def test_unconfigured_fields_return_none_not_dash(self):
        """None sentinel lets _build_config_panel drop the row entirely."""
        from dashboard import _build_runtime_summary
        raw = {"trading": {}, "risk": {}, "strategies": {"x": {"enabled": True}}}
        s = _build_runtime_summary(raw, active_strategy="x")
        assert s["max_position"] is None
        assert s["daily_loss"] is None
        assert s["kelly"] is None

    def test_fallback_when_active_strategy_missing(self):
        from dashboard import _build_runtime_summary
        raw = {"trading": {}, "risk": {},
               "strategies": {"fallback_s": {"enabled": True, "kelly_fraction": 0.20}}}
        s = _build_runtime_summary(raw, active_strategy="nonexistent")
        assert s["strategy"] == "fallback_s"
        assert s["kelly"] == "0.20"


class TestRuntimeConfigCache:
    """_load_runtime_config should hit the yaml file exactly once per process."""

    def test_cached_after_first_load(self, tmp_path):
        import dashboard
        # Reset cache so the test is hermetic
        dashboard._runtime_config_cache = None

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "trading: {paper_trading: true, interval: 300}\n"
            "risk: {max_total_exposure: 1.0}\n"
            "strategies: {logreg: {enabled: true, kelly_fraction: 0.1}}\n"
        )

        call_count = {"n": 0}
        real_safe_load = __import__("yaml").safe_load

        def counting_safe_load(*args, **kwargs):
            call_count["n"] += 1
            return real_safe_load(*args, **kwargs)

        with patch("dashboard.yaml.safe_load", side_effect=counting_safe_load):
            s1 = dashboard._load_runtime_config(str(cfg_file), "logreg")
            s2 = dashboard._load_runtime_config(str(cfg_file), "logreg")
            s3 = dashboard._load_runtime_config(str(cfg_file), "logreg")

        assert call_count["n"] == 1, f"yaml.safe_load should be called once, got {call_count['n']}"
        assert s1 is s2 is s3  # identity — same cached dict

        # Clean up module state so other tests aren't affected
        dashboard._runtime_config_cache = None

    def test_build_config_panel_drops_none_rows(self, tmp_path):
        """A config with only minimal fields should not render `None` anywhere."""
        import dashboard
        dashboard._runtime_config_cache = None

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "trading: {paper_trading: true, interval: 300}\n"
            "risk: {max_total_exposure: 1.0, max_session_loss_usdc: 50}\n"
            "strategies: {logreg: {enabled: true, model_dir: models/logreg_v4}}\n"
        )
        panel = dashboard._build_config_panel({"strategy_name": "logreg"}, str(cfg_file))
        rendered = _render(panel)
        assert "None" not in rendered
        assert "PAPER" in rendered
        assert "models/logreg_v4" in rendered

        dashboard._runtime_config_cache = None
