"""Kelly criterion bet sizing."""


def kelly_fraction(
    p: float,
    q: float,
    odds: float,
    max_fraction: float = 0.05,
) -> float:
    """Compute the Kelly bet fraction, clamped to [0, max_fraction].

    The Kelly criterion gives the optimal fraction of bankroll to wager:

        f* = (p * odds - q) / odds
           = p - q / odds

    where:
        p = probability of winning
        q = probability of losing (should equal 1 - p for a binary bet)
        odds = net payout per dollar wagered (e.g. 2.0 means you win $2 on a $1 bet)

    When the edge is negative (expected loss), returns 0.0.
    When the edge implies a fraction above max_fraction, returns max_fraction.

    Args:
        p: Probability of winning (0 to 1).
        q: Probability of losing (0 to 1).
        odds: Net payout per unit wagered. Must be positive.
        max_fraction: Upper clamp on the returned fraction. Default 0.05 (5%).

    Returns:
        Fraction of bankroll to wager, in [0, max_fraction].

    Raises:
        ValueError: If odds <= 0.
    """
    if odds <= 0:
        raise ValueError(f"odds must be positive, got {odds}")

    f = (p * odds - q) / odds
    return max(0.0, min(f, max_fraction))
