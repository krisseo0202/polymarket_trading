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
