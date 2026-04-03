from rich.console import Console

import dashboard


def _render_panel(panel) -> str:
    console = Console(record=True, width=120)
    console.print(panel)
    return console.export_text()


def test_strategy_panel_renders_probability_context():
    bot_state = {
        "strategy_name": "prob_edge",
        "strategy_status": "WATCHING",
        "strategy_prob_yes": 0.612,
        "strategy_prob_no": 0.388,
        "strategy_edge_yes": 0.041,
        "strategy_edge_no": -0.022,
        "strategy_net_edge_yes": 0.031,
        "strategy_net_edge_no": -0.033,
        "strategy_expected_fill_yes": 0.581,
        "strategy_expected_fill_no": 0.420,
        "strategy_required_edge": 0.030,
        "strategy_tte_seconds": 118.0,
        "strategy_distance_to_break_pct": 0.002,
        "strategy_distance_to_strike_bps": 20.0,
        "strategy_model_version": "btc_sigmoid_v1",
        "strategy_feature_status": "ready",
        "strategy_score_breakdown": {
            "dist_contrib": 0.22,
            "mom1_contrib": 0.10,
            "mom3_contrib": 0.07,
            "mom5_contrib": 0.03,
            "td_contrib": 0.0,
            "time_weight": 1.75,
            "score": 0.42,
        },
    }

    rendered = _render_panel(dashboard._build_strategy_panel(bot_state))

    assert "Prob YES" in rendered
    assert "Prob NO" in rendered
    assert "Px vs K" in rendered
    assert "TTE" in rendered
    assert "Net YES" in rendered
    assert "Net NO" in rendered
    assert "Req Edge" in rendered
    assert "Distance" in rendered
    assert "Momentum 1m" in rendered
    assert "Time Weight" in rendered
    assert "TD" not in rendered


def test_strategy_panel_falls_back_to_snapshot_fields():
    snapshot = {
        "strategy_name": "prob_edge",
        "strategy_status": "WATCHING",
        "strategy_prob_yes": 0.55,
        "strategy_net_edge_yes": 0.02,
        "strategy_required_edge": 0.03,
        "strategy_tte_seconds": 90.0,
        "strategy_distance_to_break_pct": -0.001,
        "strategy_distance_to_strike_bps": -10.0,
    }

    rendered = _render_panel(dashboard._build_strategy_panel(None, snapshot=snapshot))

    assert "prob_edge" in rendered
    assert "Prob YES" in rendered
    assert "Net YES" in rendered
    assert "Px vs K" in rendered
