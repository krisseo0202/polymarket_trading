"""Realized volatility estimation utilities."""

import math
from typing import List


def estimate_realized_vol(
    prices: List[float],
    window_sec: int,
    method: str = "std",
    sample_interval_sec: float = 1.0,
) -> float:
    """Estimate realized volatility from a price series.

    Args:
        prices: List of price observations (chronological order).
        window_sec: Lookback window in seconds.
        method: "std" for sample standard deviation, "ema" for EMA-weighted std dev.
        sample_interval_sec: Seconds between consecutive price observations.

    Returns:
        Realized volatility of simple returns over the window. Returns 0.0 when
        there is insufficient data.

    Raises:
        ValueError: If method is not "std" or "ema".
    """
    if method not in ("std", "ema"):
        raise ValueError(f"method must be 'std' or 'ema', got '{method}'")

    window_count = int(window_sec / sample_interval_sec)
    window_count = min(window_count, len(prices))

    tail = prices[-window_count:] if window_count > 0 else []

    if len(tail) < 2:
        return 0.0

    # Simple returns, skipping zero denominators
    returns: List[float] = []
    for i in range(1, len(tail)):
        if tail[i - 1] != 0:
            returns.append((tail[i] - tail[i - 1]) / tail[i - 1])

    if len(returns) < 1:
        return 0.0

    if method == "std":
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var)

    # EMA method
    n = len(returns)
    alpha = 2.0 / (n + 1)
    ema_mean = returns[0]
    ema_sq = returns[0] ** 2
    for r in returns[1:]:
        ema_mean = alpha * r + (1 - alpha) * ema_mean
        ema_sq = alpha * r ** 2 + (1 - alpha) * ema_sq
    variance = ema_sq - ema_mean ** 2
    return math.sqrt(max(variance, 0.0))
