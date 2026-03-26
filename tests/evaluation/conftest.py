"""Fixtures for evaluation agent tests."""

import random
from datetime import datetime, timezone

import pandas as pd
import pytest


def make_sample_df(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """Generate n rows of synthetic resolved BTC Up/Down market slots."""
    rng = random.Random(seed)
    rows = []
    base_ts = 1_700_000_000
    for i in range(n):
        mid = 0.50 + rng.uniform(-0.15, 0.15)
        mid = max(0.05, min(0.95, mid))
        outcome = "Up" if i % 2 == 0 else "Down"
        rows.append({
            "slot_ts": base_ts + i * 300,
            "slot_utc": datetime.fromtimestamp(base_ts + i * 300, tz=timezone.utc).isoformat(),
            "question": f"BTC >/< $50000 at slot {i}",
            "up_token": f"up_token_{i}",
            "down_token": f"down_token_{i}",
            "outcome": outcome,
            "volume": 5000.0,
            "up_price_start": mid,
            "up_price_end": 1.0 if outcome == "Up" else 0.0,
            "down_price_start": 1.0 - mid,
            "down_price_end": 0.0 if outcome == "Up" else 1.0,
            "strike_price": 50000.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return make_sample_df(n=40, seed=42)
