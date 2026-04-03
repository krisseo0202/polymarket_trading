from src.engine.cycle_snapshot import CycleSnapshot


def test_cycle_snapshot_strategy_fields_round_trip():
    snap = CycleSnapshot(
        market_id="mkt_1",
        strategy_name="prob_edge",
        strategy_status="WATCHING",
        strategy_prob_yes=0.61,
        strategy_prob_no=0.39,
        strategy_edge_yes=0.04,
        strategy_edge_no=-0.03,
        strategy_net_edge_yes=0.031,
        strategy_net_edge_no=-0.041,
        strategy_expected_fill_yes=0.579,
        strategy_expected_fill_no=0.421,
        strategy_required_edge=0.03,
        strategy_tte_seconds=118.0,
        strategy_distance_to_break_pct=0.002,
        strategy_distance_to_strike_bps=20.0,
        strategy_model_version="btc_sigmoid_v1",
        strategy_feature_status="ready",
        strategy_score_breakdown={"score": 0.44},
    )

    loaded = CycleSnapshot.from_dict(snap.to_dict())

    assert loaded.strategy_name == "prob_edge"
    assert loaded.strategy_prob_yes == 0.61
    assert loaded.strategy_prob_no == 0.39
    assert loaded.strategy_net_edge_yes == 0.031
    assert loaded.strategy_expected_fill_yes == 0.579
    assert loaded.strategy_distance_to_strike_bps == 20.0
    assert loaded.strategy_score_breakdown == {"score": 0.44}
