"""
edge_detector.py — Edge detection between bot probability and Polymarket odds.

Compares the bot's estimated UP probability against market-implied odds to
identify when a tradeable edge exists. Pure function — no I/O, no side effects.

Usage::

    from src.utils.edge_detector import detect_edge

    decision = detect_edge(
        bot_up_prob=0.65,
        market_up_odds=0.55,
        market_down_odds=0.45,
    )
    print(decision.side)          # "BUY_UP"
    print(decision.edge)          # ~0.10
    print(decision.kelly_fraction) # Kelly-sized position fraction
"""

from dataclasses import dataclass
from typing import Optional

from .kelly import kelly_fraction as _kelly_fraction


# ── Constants ─────────────────────────────────────────────────────────────────

_LIQUIDITY_THRESHOLD = 0.85  # skip trade if up_odds + down_odds < this


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class EdgeDecision:
    """
    Result of one edge-detection evaluation.

    Fields
    ------
    side : str
        Canonical trade side: ``"BUY_UP"``, ``"BUY_DOWN"``, or ``"NO_TRADE"``.
    bot_up_prob : float
        Bot-estimated probability BTC finishes >= cycle-start price.
    market_up_odds : float
        Polymarket-implied probability for the UP outcome (mid-price).
    market_down_odds : float
        Polymarket-implied probability for the DOWN outcome (mid-price).
    up_edge : float
        ``bot_up_prob - market_up_odds`` (raw, may be negative).
    down_edge : float
        ``(1 - bot_up_prob) - market_down_odds`` (raw, may be negative).
    edge : float
        Edge on the selected side (0.0 if ``NO_TRADE``).
    kelly_fraction : float
        Kelly-sized position fraction in [0, max_kelly] (0.0 if ``NO_TRADE``).
    min_edge_threshold : float
        The threshold used to make the decision.
    skip_reason : str | None
        Human-readable reason when side is ``NO_TRADE`` due to a guard condition
        (e.g. thin liquidity). None when NO_TRADE is simply below-threshold.
    """
    side: str
    bot_up_prob: float
    market_up_odds: float
    market_down_odds: float
    up_edge: float
    down_edge: float
    edge: float
    kelly_fraction: float
    min_edge_threshold: float
    skip_reason: Optional[str] = None


# ── Public API ────────────────────────────────────────────────────────────────

def detect_edge(
    bot_up_prob: float,
    market_up_odds: float,
    market_down_odds: float,
    min_edge: float = 0.03,
    max_kelly: float = 0.05,
) -> EdgeDecision:
    """
    Compare bot probability against Polymarket odds and decide whether to trade.

    Returns ``NO_TRADE`` in three scenarios:
      1. Neither side has sufficient edge (< ``min_edge``).
      2. The order book is too thin: ``market_up_odds + market_down_odds < 0.85``.

    When both sides have positive edge, the side with the larger edge is chosen.

    Args:
        bot_up_prob:      Bot-estimated probability BTC finishes UP (0–1).
        market_up_odds:   Polymarket mid-price for the UP outcome (0–1).
        market_down_odds: Polymarket mid-price for the DOWN outcome (0–1).
        min_edge:         Minimum edge to produce a trade signal (default 0.03).
        max_kelly:        Maximum Kelly fraction cap (default 0.05 = 5%).

    Returns:
        :class:`EdgeDecision` with all computed fields.
    """
    up_edge   = bot_up_prob - market_up_odds
    down_edge = (1.0 - bot_up_prob) - market_down_odds

    # ── Liquidity guard ───────────────────────────────────────────────────────
    if market_up_odds + market_down_odds < _LIQUIDITY_THRESHOLD:
        return EdgeDecision(
            side="NO_TRADE",
            bot_up_prob=bot_up_prob,
            market_up_odds=market_up_odds,
            market_down_odds=market_down_odds,
            up_edge=up_edge,
            down_edge=down_edge,
            edge=0.0,
            kelly_fraction=0.0,
            min_edge_threshold=min_edge,
            skip_reason=(
                f"thin order book: up_odds({market_up_odds:.3f}) + "
                f"down_odds({market_down_odds:.3f}) = "
                f"{market_up_odds + market_down_odds:.3f} < {_LIQUIDITY_THRESHOLD}"
            ),
        )

    # ── Edge selection ────────────────────────────────────────────────────────
    up_qualifies   = up_edge   >= min_edge
    down_qualifies = down_edge >= min_edge

    if not up_qualifies and not down_qualifies:
        return EdgeDecision(
            side="NO_TRADE",
            bot_up_prob=bot_up_prob,
            market_up_odds=market_up_odds,
            market_down_odds=market_down_odds,
            up_edge=up_edge,
            down_edge=down_edge,
            edge=0.0,
            kelly_fraction=0.0,
            min_edge_threshold=min_edge,
        )

    # Pick the side with the larger edge when both qualify
    if up_qualifies and down_qualifies:
        use_up = up_edge >= down_edge
    else:
        use_up = up_qualifies

    if use_up:
        side  = "BUY_UP"
        edge  = up_edge
        kf    = _compute_kelly(bot_up_prob, market_up_odds, max_kelly)
    else:
        side  = "BUY_DOWN"
        edge  = down_edge
        kf    = _compute_kelly(1.0 - bot_up_prob, market_down_odds, max_kelly)

    return EdgeDecision(
        side=side,
        bot_up_prob=bot_up_prob,
        market_up_odds=market_up_odds,
        market_down_odds=market_down_odds,
        up_edge=up_edge,
        down_edge=down_edge,
        edge=edge,
        kelly_fraction=kf,
        min_edge_threshold=min_edge,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _compute_kelly(p: float, market_prob: float, max_kelly: float) -> float:
    """
    Compute Kelly fraction for a binary bet.

    Args:
        p:           Probability of winning (bot estimate).
        market_prob: Market-implied probability (used to derive net odds).
        max_kelly:   Cap on returned fraction.

    Returns:
        Kelly fraction in [0.0, max_kelly].
    """
    if market_prob <= 0.0 or market_prob >= 1.0:
        return 0.0
    q    = 1.0 - p
    odds = (1.0 / market_prob) - 1.0  # net payout per unit wagered
    if odds <= 0.0:
        return 0.0
    return _kelly_fraction(p=p, q=q, odds=odds, max_fraction=max_kelly)
