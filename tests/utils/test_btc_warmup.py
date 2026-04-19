"""Tests for the BTC history warmup helper."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.btc_warmup import btc_dataframe_to_tuples, warmup_btc_history


def _frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _synth(start_ts: int, n: int, step: int = 1):
    rows = [(start_ts + i * step, 100.0, 101.0, 99.0, 100.5, 1.0) for i in range(n)]
    return _frame(rows)


def test_returns_existing_when_already_covers_window():
    base = _synth(1000, n=100)
    with patch("src.utils.btc_warmup.fetch_btc_klines") as fake:
        out = warmup_btc_history(base, need_start_ts=1010, need_end_ts=1080)
        fake.assert_not_called()
    # Nothing extra fetched → same frame back.
    assert out["timestamp"].iloc[0] == 1000
    assert len(out) == 100


def test_fetches_head_gap_when_base_starts_too_late():
    base = _synth(1050, n=50)
    head = _synth(1000, n=50)
    with patch("src.utils.btc_warmup.fetch_btc_klines", return_value=head) as fake:
        out = warmup_btc_history(base, need_start_ts=1000, need_end_ts=1099)
        assert fake.call_count == 1
    assert out["timestamp"].iloc[0] == 1000
    assert out["timestamp"].iloc[-1] == 1099
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].is_unique


def test_fetches_tail_gap_when_base_ends_too_early():
    base = _synth(1000, n=50)
    tail = _synth(1050, n=50)
    with patch("src.utils.btc_warmup.fetch_btc_klines", return_value=tail) as fake:
        out = warmup_btc_history(base, need_start_ts=1000, need_end_ts=1099)
        assert fake.call_count == 1
    assert out["timestamp"].iloc[-1] == 1099


def test_fetches_both_gaps_when_base_is_in_middle():
    base = _synth(1030, n=40)  # covers [1030, 1069]
    fetch_calls = []

    def fake_fetch(start_ts, end_ts, **kwargs):
        fetch_calls.append((start_ts, end_ts))
        # Return a synthetic chunk covering the requested span.
        n = int(end_ts - start_ts) + 1
        return _synth(int(start_ts), n=n)

    with patch("src.utils.btc_warmup.fetch_btc_klines", side_effect=fake_fetch):
        out = warmup_btc_history(base, need_start_ts=1000, need_end_ts=1099)

    # Both head and tail were fetched.
    assert len(fetch_calls) == 2
    assert out["timestamp"].iloc[0] <= 1000
    assert out["timestamp"].iloc[-1] >= 1099


def test_empty_existing_triggers_full_fetch():
    with patch("src.utils.btc_warmup.fetch_btc_klines", return_value=_synth(1000, n=100)) as fake:
        out = warmup_btc_history(None, need_start_ts=1000, need_end_ts=1099)
        assert fake.call_count == 1
    assert len(out) == 100


def test_dedups_on_merge():
    base = _synth(1000, n=50)  # [1000, 1049]
    tail = _synth(1040, n=50)  # [1040, 1089] — overlap [1040..1049]
    with patch("src.utils.btc_warmup.fetch_btc_klines", return_value=tail):
        out = warmup_btc_history(base, need_start_ts=1000, need_end_ts=1089)
    assert out["timestamp"].is_unique
    assert len(out) == 90  # 50 + 50 − 10 overlap


def test_rejects_bad_window():
    with pytest.raises(ValueError):
        warmup_btc_history(None, need_start_ts=100, need_end_ts=50)


def test_dataframe_to_tuples_emits_ts_close_volume():
    df = _synth(1000, n=3)
    # Mutate close to a distinctive value so we can verify mapping.
    df["close"] = [10.0, 11.0, 12.0]
    df["volume"] = [1.0, 2.0, 3.0]
    tuples = btc_dataframe_to_tuples(df)
    assert tuples == [(1000.0, 10.0, 1.0), (1001.0, 11.0, 2.0), (1002.0, 12.0, 3.0)]
