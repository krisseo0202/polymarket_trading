import json

from src.engine.state_store import BotState, StateStore


def test_save_load_preserves_chainlink_ref_slot_ts(tmp_path):
    path = tmp_path / "bot_state.json"
    store = StateStore(str(path))

    state = BotState(
        chainlink_ref_price=84_500.25,
        chainlink_ref_slot_ts=1_700_000_100,
        chainlink_healthy=True,
    )
    store.save(state)

    loaded = store.load()
    assert loaded.chainlink_ref_price == 84_500.25
    assert loaded.chainlink_ref_slot_ts == 1_700_000_100
    assert loaded.chainlink_healthy is True


def test_save_load_preserves_session_counters(tmp_path):
    """session_wins / session_losses must survive restart — otherwise the
    mid-slot counter fix is defeated the first time the bot is killed."""
    path = tmp_path / "bot_state.json"
    store = StateStore(str(path))

    state = BotState(session_wins=3, session_losses=7)
    store.save(state)

    loaded = store.load()
    assert loaded.session_wins == 3
    assert loaded.session_losses == 7


def test_load_older_state_without_session_counters(tmp_path):
    """Legacy state files don't have the counter keys — defaults apply."""
    path = tmp_path / "bot_state.json"
    path.write_text(json.dumps({"daily_realized_pnl": 1.23}), encoding="utf-8")
    loaded = StateStore(str(path)).load()
    assert loaded.session_wins == 0
    assert loaded.session_losses == 0


def test_load_older_state_without_chainlink_ref_slot_ts(tmp_path):
    path = tmp_path / "bot_state.json"
    path.write_text(
        json.dumps(
            {
                "chainlink_ref_price": 84_500.25,
                "chainlink_healthy": True,
            }
        ),
        encoding="utf-8",
    )

    loaded = StateStore(str(path)).load()
    assert loaded.chainlink_ref_price == 84_500.25
    assert loaded.chainlink_ref_slot_ts is None
    assert loaded.chainlink_healthy is True
