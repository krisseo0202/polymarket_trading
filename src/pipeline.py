"""
pipeline.py — End-to-end BTC probability engine + cycle-aligned execution.

Wires together:
  BtcPriceFeed → volatility estimator → GBM Monte Carlo → edge detector
  → cycle scheduler → (optional) order placement

Usage::

    from src.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(market_id="12345")
    run_pipeline(config)               # blocks until stop_event set or KeyboardInterrupt
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from quant_desk import simulate_up_prob
from src.utils import (
    BtcPriceFeed,
    estimate_realized_vol,
    detect_edge,
    EdgeDecision,
)
from src.utils.cycle_scheduler import (
    aligned_cycle_anchor,
    cycle_index,
    run_last_second_strategy,
)
from src.utils.market_utils import fetch_market_odds


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    All tunable parameters for one pipeline run.

    Fields
    ------
    market_id         : Polymarket Gamma market ID for the BTC Up/Down market.
    cycle_len         : Cycle duration in seconds (default 300 = 5 min).
    trigger_window_s  : Fire callback when this many seconds remain in cycle.
    vol_window_s      : Lookback window for realised-vol estimation (seconds).
    vol_method        : ``"ema"`` (default) or ``"std"``.
    n_paths           : Monte Carlo path count.
    min_edge          : Minimum edge threshold for a trade signal.
    max_kelly         : Kelly fraction cap.
    dry_run           : If True, log decisions but do NOT place orders.
    btc_symbol        : Binance bookTicker symbol (lowercase).
    """
    market_id:        str
    cycle_len:        int   = 300
    trigger_window_s: float = 30.0
    vol_window_s:     float = 300.0
    vol_method:       str   = "ema"
    n_paths:          int   = 1000
    min_edge:         float = 0.03
    max_kelly:        float = 0.05
    dry_run:          bool  = True
    btc_symbol:       str   = "btcusdt"


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    config: PipelineConfig,
    client=None,
    stop_event: Optional[threading.Event] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Start the BTC price feed, align to the 5-minute wall-clock grid, and fire
    a probability + edge evaluation callback once per cycle in the last
    ``trigger_window_s`` seconds.

    Blocks until ``stop_event`` is set (or KeyboardInterrupt).

    Args:
        config:      Pipeline configuration.
        client:      Optional ``PolymarketClient`` for live order placement.
                     Ignored when ``config.dry_run`` is True.
        stop_event:  ``threading.Event`` — set it to stop the loop cleanly.
        logger:      Logger instance; defaults to the module logger.
    """
    log = logger or logging.getLogger(__name__)

    if stop_event is None:
        stop_event = threading.Event()

    # ── Start price feed ──────────────────────────────────────────────────────
    feed = BtcPriceFeed(symbol=config.btc_symbol)
    feed.start()
    log.info(f"BTC feed started (symbol={config.btc_symbol})")

    # ── Align anchor to wall-clock grid ──────────────────────────────────────
    anchor = aligned_cycle_anchor(config.cycle_len)
    log.info(f"Cycle anchor={anchor:.3f}  cycle_len={config.cycle_len}s")

    # Mutable state shared between the outer scope and the callback closure.
    state = {
        "current_cycle_idx": -1,
        "btc_start_price":   None,
    }

    # ── Per-cycle callback ────────────────────────────────────────────────────
    def _callback() -> None:
        # 1. Detect cycle rollover → capture new start price
        idx = cycle_index(anchor, cycle_len=config.cycle_len)
        if idx != state["current_cycle_idx"]:
            state["current_cycle_idx"] = idx
            mid = feed.get_latest_mid()
            state["btc_start_price"] = mid
            log.debug(f"Cycle {idx} started — start_price={mid}")

        # 2. Feed health check
        if not feed.is_healthy():
            log.warning("BTC feed unhealthy — skipping this cycle")
            return

        # 3. Gather recent prices and estimate volatility
        recent = feed.get_recent_prices(config.vol_window_s)
        prices = [p for _, p in recent]
        vol = estimate_realized_vol(prices, window_sec=config.vol_window_s)

        # 4. Estimate up-probability
        start_price = state["btc_start_price"] or feed.get_latest_mid()
        current_price = feed.get_latest_mid()
        if start_price is None or current_price is None:
            log.warning("No BTC price available — skipping this cycle")
            return

        from src.utils.cycle_scheduler import detect_current_cycle
        time_left_sec = detect_current_cycle(anchor, cycle_len=config.cycle_len)

        up_prob = simulate_up_prob(
            start_price=start_price,
            current_price=current_price,
            time_left_sec=max(time_left_sec, 0.001),
            vol=max(vol, 0.001),
            n_paths=config.n_paths,
        )

        # 5. Fetch market odds (skip cycle on API failure)
        try:
            market_up_odds, market_down_odds = fetch_market_odds(config.market_id)
        except (RuntimeError, ValueError) as exc:
            log.warning(f"Market odds fetch failed — skipping cycle: {exc}")
            return

        # 6. Detect edge
        decision: EdgeDecision = detect_edge(
            bot_up_prob=up_prob,
            market_up_odds=market_up_odds,
            market_down_odds=market_down_odds,
            min_edge=config.min_edge,
            max_kelly=config.max_kelly,
        )

        # 7. Log CycleTick summary
        log.info(
            f"[Cycle {idx}] "
            f"btc={current_price:.2f}  vol={vol:.4f}  up_prob={up_prob:.4f}  "
            f"mkt_up={market_up_odds:.4f}  mkt_dn={market_down_odds:.4f}  "
            f"side={decision.side}  edge={decision.edge:.4f}  "
            f"kelly={decision.kelly_fraction:.4f}"
            + (f"  skip={decision.skip_reason}" if decision.skip_reason else "")
        )

        # 8. Place order (live mode only)
        if not config.dry_run and client is not None and decision.side != "NO_TRADE":
            try:
                client.place_order(decision)
            except Exception as exc:
                log.error(f"Order placement failed: {exc}")

    # ── Run scheduler loop (blocks) ───────────────────────────────────────────
    try:
        run_last_second_strategy(
            anchor,
            _callback,
            cycle_len=config.cycle_len,
            trigger_window_s=config.trigger_window_s,
            stop_event=stop_event,
        )
    finally:
        feed.stop()
        log.info("Pipeline stopped")
