"""
BTC Up/Down Bot — wall-clock-aligned 5-minute trading loop.

Usage:
    python bot.py

Configuration: config/config.yaml
    Set paper_trading: false and provide PRIVATE_KEY / PROXY_FUNDER env vars
    for live trading.
"""

import argparse
import json
import math
import os
import requests
import signal
import sys
import threading
import time
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.api.client import PolymarketClient
from src.api.types import OrderBook, OrderBookEntry
from src.engine.cycle_runner import CycleResult, CycleRunner
from src.engine.slot_state import SlotContext
from src.models.feature_builder import _realized_vol, build_live_features
from src.models.slot_path_state import SlotPathState
from src.utils.btc_feed import BtcPriceFeed
from src.utils.market_utils import get_server_time
from src.utils.startup import STRATEGIES, init_services

# ---------------------------------------------------------------------------
# Globals (written by signal handlers, read by main loop / ticker)
# ---------------------------------------------------------------------------
_shutdown_requested = False

# Structured price-tick log path — set once at startup from config
_price_tick_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers (used by ticker_until_next_cycle)
# ---------------------------------------------------------------------------

def _log_price_tick(record: dict) -> None:
    """Append one JSON price-tick record to the probability tick log."""
    if _price_tick_path is None:
        return
    line = json.dumps({k: v for k, v in record.items() if v is not None}, default=str)
    with open(_price_tick_path, "a") as f:
        f.write(line + "\n")
        f.flush()


def _l1_book(bid: Optional[float], ask: Optional[float]) -> Optional[OrderBook]:
    """Build a 1-level OrderBook from ticker top-of-book. Returns None if missing."""
    if bid is None and ask is None:
        return None
    bids = [OrderBookEntry(price=float(bid), size=100.0)] if bid is not None else []
    asks = [OrderBookEntry(price=float(ask), size=100.0)] if ask is not None else []
    return OrderBook(market_id="", token_id="", bids=bids, asks=asks, tick_size=0.001)


def _log_intra_cycle_features(
    logger,
    model,
    slot_state: SlotPathState,
    btc_feed,
    slot_mgr,
    yes_bid: Optional[float],
    yes_ask: Optional[float],
    no_bid: Optional[float],
    no_ask: Optional[float],
) -> None:
    """Recompute features from the ticker's in-memory data and log them.

    Does NOT call model.predict — this is read-only display. Family A depth
    features will sit at 0.0 because the ticker only has L1 (top-of-book)
    prices; full depth only refreshes on cycle/intra-cycle fetches. BTC
    momentum, vol, moneyness, and Family C path features update at ticker
    cadence (~5s).
    """
    if logger is None or model is None or btc_feed is None or slot_mgr is None:
        return

    ctx = slot_mgr.get()
    if ctx is None or ctx.strike_price is None or ctx.slot_end_ts is None:
        return

    yes_book = _l1_book(yes_bid, yes_ask)
    no_book = _l1_book(no_bid, no_ask)
    if yes_book is None or no_book is None:
        return

    btc_prices = btc_feed.get_recent_prices(300)
    if len(btc_prices) < 2:
        return

    now_ts = time.time()
    strike = float(ctx.strike_price)
    btc_now = float(btc_prices[-1][1])
    slot_ts = int(float(ctx.slot_end_ts)) - 300

    # Reset on slot rollover; fold each new BTC tick since the last update.
    if slot_state.slot_ts != slot_ts:
        slot_state.reset(slot_ts)
    last_ts = slot_state.last_ts or float(slot_ts)
    for ts, price, *_ in btc_prices:
        ts_f = float(ts)
        if ts_f <= last_ts or ts_f < slot_ts:
            continue
        slot_state.update(ts_f, float(price), strike)

    slot_path_features = slot_state.to_features(now_ts, btc_now, strike)

    snapshot = {
        "btc_prices": btc_prices,
        "yes_book": yes_book,
        "no_book": no_book,
        "yes_history": [],
        "no_history": [],
        "question": "",
        "strike_price": strike,
        "slot_expiry_ts": ctx.slot_end_ts,
        "now_ts": now_ts,
        "slot_path_features": slot_path_features,
    }

    built = build_live_features(snapshot)
    feats = built.features
    names = getattr(model, "feature_names", None) or []
    if not feats or not names:
        return

    version = getattr(model, "model_version", "model")
    logger.info(
        "=== INTRA-CYCLE FEATURES [%s] — %d cols (status=%s) ===",
        version, len(names), built.status,
    )
    for name in names:
        val = feats.get(name, 0.0)
        try:
            fval = float(val)
        except (TypeError, ValueError):
            fval = 0.0
        logger.info("  %-32s = %+.6f", name, fval)
    logger.info("=== END INTRA-CYCLE FEATURES ===")


def _log_feature_snapshot(svc) -> None:
    """Log the feature values the model used in its last prediction.

    Only walks the model's own schema so the output is exactly what drove
    the decision — not the 150+ columns build_live_features emits.
    """
    model = getattr(svc.strategy, "model_service", None) if svc.strategy else None
    if model is None:
        return
    feats = getattr(model, "last_features", None) or {}
    names = getattr(model, "feature_names", None) or []
    if not feats or not names:
        svc.logger.info("FEATURES: model has no last_features yet (warmup)")
        return

    version = getattr(model, "model_version", "model")
    svc.logger.info("=== FEATURES [%s] — %d cols ===", version, len(names))
    for name in names:
        val = feats.get(name, 0.0)
        try:
            fval = float(val)
        except (TypeError, ValueError):
            fval = 0.0
        svc.logger.info("  %-32s = %+.6f", name, fval)
    svc.logger.info("=== END FEATURES ===")


def _stime() -> float:
    """Return local time corrected by the last-observed Polymarket server offset."""
    return get_server_time()


def sleep_until_next_cycle(interval_s: int) -> None:
    """Sleep until the next wall-clock-aligned cycle boundary (no drift)."""
    now = _stime()
    next_run = math.ceil(now / interval_s) * interval_s
    delay = max(0.0, next_run - now)
    if delay > 0:
        time.sleep(delay)


def _fetch_book_top(yes_token_id: str, no_token_id: str, logger=None) -> tuple:
    """Fetch best bid/ask for YES and NO tokens via /book."""
    from concurrent.futures import ThreadPoolExecutor

    def _top(token_id: str):
        bid, ask = None, None
        try:
            r = requests.get(
                "https://clob.polymarket.com/book",
                params={"token_id": token_id},
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            bid = float(bids[0].get("price") or bids[0].get("p")) if bids else None
            ask = float(asks[0].get("price") or asks[0].get("p")) if asks else None
        except Exception as e:
            if logger:
                logger.debug(f"Book fetch failed for {token_id[:12]}...: {e}")

        spread = (ask - bid) if (bid is not None and ask is not None) else float("inf")
        if bid is None or ask is None or spread > 0.50:
            mid = None
            try:
                r = requests.get(
                    "https://clob.polymarket.com/midpoint",
                    params={"token_id": token_id},
                    timeout=5,
                )
                r.raise_for_status()
                mid = float(r.json().get("mid", 0)) or None
            except Exception:
                pass
            if mid is not None:
                if bid is None or spread > 0.50:
                    bid = mid
                if ask is None or spread > 0.50:
                    ask = mid
        return bid, ask

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_yes = pool.submit(_top, yes_token_id)
        f_no = pool.submit(_top, no_token_id)
        yes_bid, yes_ask = f_yes.result()
        no_bid, no_ask = f_no.result()
    return yes_bid, yes_ask, no_bid, no_ask


def ticker_until_next_cycle(
    client: PolymarketClient,
    yes_token_id: str,
    no_token_id: str,
    interval_s: int,
    strategy=None,
    intra_cycle_analyze_fn=None,
    logger=None,
    btc_feed: Optional[BtcPriceFeed] = None,
    slot_mgr=None,
    display_features: bool = False,
) -> None:
    """
    Record prices in the background while waiting for the next cycle.

    Background thread records prices and runs intra-cycle analysis every 30s.
    """
    yes_bid: List[Optional[float]] = [None]
    yes_ask: List[Optional[float]] = [None]
    no_bid:  List[Optional[float]] = [None]
    no_ask:  List[Optional[float]] = [None]
    stop_evt = threading.Event()
    first_fetch_evt = threading.Event()

    # Display-only slot-path state. Kept separate from the model's own
    # SlotPathState so reading features for display never mutates what the
    # trading model sees on cycle boundaries.
    display_slot_state = SlotPathState() if display_features else None
    display_model = (
        getattr(strategy, "model_service", None) if display_features and strategy else None
    )

    def _fetch_loop():
        last_analyzed = time.monotonic()
        while not stop_evt.is_set():
            y_bid, y_ask, n_bid, n_ask = _fetch_book_top(yes_token_id, no_token_id, logger=logger)
            if y_bid is not None: yes_bid[0] = y_bid
            if y_ask is not None: yes_ask[0] = y_ask
            if n_bid is not None: no_bid[0]  = n_bid
            if n_ask is not None: no_ask[0]  = n_ask
            if y_bid is not None or n_bid is not None:
                first_fetch_evt.set()

            if y_bid is not None and y_ask is not None and strategy:
                strategy.record_price(yes_token_id, (y_bid + y_ask) / 2)
            if n_bid is not None and n_ask is not None and strategy:
                strategy.record_price(no_token_id, (n_bid + n_ask) / 2)

            yb, ya = yes_bid[0], yes_ask[0]
            nb, na = no_bid[0], no_ask[0]
            _now_ts = time.time()
            _btc_now = btc_feed.get_latest_mid() if btc_feed else None
            _btc_prices = btc_feed.get_recent_prices(300) if btc_feed else []
            _vol_30s = _realized_vol(_btc_prices, _now_ts, 30) if len(_btc_prices) >= 3 else None
            _vol_60s = _realized_vol(_btc_prices, _now_ts, 60) if len(_btc_prices) >= 3 else None
            _strike, _strike_src = None, None
            if slot_mgr:
                _ctx = slot_mgr.get()
                if _ctx:
                    _strike, _strike_src = _ctx.strike_price, _ctx.strike_source
            _log_price_tick({
                "ts": _now_ts,
                "slot_ts": SlotContext.slot_for(_stime()),
                "yes_bid": yb, "yes_ask": ya,
                "no_bid": nb, "no_ask": na,
                "yes_mid": (yb + ya) / 2 if yb and ya else None,
                "no_mid": (nb + na) / 2 if nb and na else None,
                "yes_spread": ya - yb if yb and ya else None,
                "no_spread": na - nb if nb and na else None,
                "btc_now": _btc_now,
                "strike": _strike,
                "strike_source": _strike_src,
                "realized_vol_30s": _vol_30s,
                "realized_vol_60s": _vol_60s,
            })

            now_m = time.monotonic()
            if intra_cycle_analyze_fn and (now_m - last_analyzed) >= 30:
                intra_cycle_analyze_fn()
                last_analyzed = now_m

            if display_features and display_slot_state is not None:
                try:
                    _log_intra_cycle_features(
                        logger=logger,
                        model=display_model,
                        slot_state=display_slot_state,
                        btc_feed=btc_feed,
                        slot_mgr=slot_mgr,
                        yes_bid=yes_bid[0],
                        yes_ask=yes_ask[0],
                        no_bid=no_bid[0],
                        no_ask=no_ask[0],
                    )
                except Exception as exc:
                    # Promoted from debug to warning: silent failures here
                    # were hiding real bugs (e.g. slot_mgr returning None on
                    # Chainlink reconnect, or a missing attribute on the
                    # model). We want to know if display stops working.
                    if logger:
                        logger.warning("intra-cycle feature logging failed: %s", exc)

            for _ in range(5):
                if stop_evt.wait(1.0):
                    return

    fetcher = threading.Thread(target=_fetch_loop, daemon=True)
    fetcher.start()

    def _fmt(v): return f"{v:.4f}" if v is not None else " --- "

    def _pos_str() -> str:
        pos = strategy.get_position_info() if strategy and hasattr(strategy, "get_position_info") else None
        if pos is None:
            return "FLAT"
        outcome     = pos["outcome"]
        entry_price = pos["entry_price"]
        entry_size  = pos["entry_size"]
        if outcome == "YES":
            cb, ca = yes_bid[0], yes_ask[0]
        else:
            cb, ca = no_bid[0], no_ask[0]
        if cb is not None and ca is not None:
            cur_mid = (cb + ca) / 2
        elif cb is not None:
            cur_mid = cb
        elif ca is not None:
            cur_mid = ca
        else:
            cur_mid = entry_price
        unrealized = (cur_mid - entry_price) * entry_size
        pnl_pct = (cur_mid - entry_price) / entry_price * 100 if entry_price else 0.0
        return (
            f"POS: {outcome} {entry_size:.1f}sh @{entry_price:.4f} "
            f"uPnL={unrealized:+.2f} ({pnl_pct:+.1f}%)"
        )

    first_fetch_evt.wait(timeout=5)
    last_logged = 0.0

    def _log_status():
        now = _stime()
        next_run = math.ceil(now / interval_s) * interval_s
        remaining = max(0, next_run - now)
        mins, secs = divmod(int(remaining), 60)
        if logger:
            logger.info(
                f"  YES b={_fmt(yes_bid[0])} a={_fmt(yes_ask[0])}"
                f" | NO b={_fmt(no_bid[0])} a={_fmt(no_ask[0])}"
                f" | {_pos_str()}"
                f" | Next cycle in {mins:02d}:{secs:02d}"
            )

    _now = _stime()
    target_boundary = math.ceil(_now / interval_s) * interval_s

    try:
        while not _shutdown_requested:
            now = _stime()
            if now >= target_boundary:
                break
            if now - last_logged >= 30.0:
                _log_status()
                last_logged = now
            time.sleep(1.0)
    finally:
        stop_evt.set()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _price_tick_path

    parser = argparse.ArgumentParser(description="Polymarket trading bot")
    parser.add_argument("--asset", default=None,
                        help="Asset to trade: BTC, ETH, SOL, DOGE, XRP (default: from config)")
    # Keep pre-existing legacy names + merge anything startup.STRATEGIES now advertises.
    _legacy_strategies = [
        "btc_updown", "btc_updown_xgb", "btc_vol_reversion", "coin_toss",
        "logreg_edge", "logreg", "prob_edge", "td_rsi",
    ]
    _strategy_choices = sorted(set(_legacy_strategies) | set(STRATEGIES))
    parser.add_argument("--strategy", choices=_strategy_choices, default=None)
    parser.add_argument("--paper", action="store_true", default=False)
    parser.add_argument("--live", action="store_true", default=False)
    parser.add_argument("--no-confirm", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Start a clean session: clear trade logs and reset bot state")
    parser.add_argument("--delta", type=float, default=None,
                        help="Minimum edge threshold for entry (default: 0.025)")
    parser.add_argument("--balance", type=float, default=None,
                        help="Paper trading balance in USD (default: 10000)")
    parser.add_argument("--position-size", type=float, default=None,
                        help="Max position size per trade in USD (default: 30)")
    parser.add_argument("--kelly", type=float, default=None,
                        help="Kelly fraction for position sizing (default: 0.15)")
    parser.add_argument("--exit-rule", choices=["default", "hold_to_expiry"],
                        default=None, help="Exit strategy: default (strategy-defined) or hold_to_expiry")
    parser.add_argument("--display-features", action="store_true", default=False,
                        help="After every cycle, log the live feature values the model used to predict. "
                             "Only the features in the loaded model's schema are shown.")
    parser.add_argument("--interval", type=int, default=None,
                        help="Cycle period in seconds. Overrides trading.interval in config.yaml. "
                             "Default 300 (5-min slot boundary). Lower values cause more trade decisions "
                             "per slot; useful for testing.")
    args = parser.parse_args()

    svc = init_services(args)
    # CLI --interval overrides whatever init_services read from config.yaml.
    if args.interval is not None:
        if args.interval <= 0:
            raise SystemExit(f"--interval must be positive, got {args.interval}")
        svc.interval = int(args.interval)
    _price_tick_path = svc.price_tick_path

    runner = CycleRunner(svc)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _handle_signal)

    while True:
        if _shutdown_requested:
            runner.shutdown()

        svc.logger.info(f"=== Main loop iteration (cycle {svc.state.cycle_count + 1}) ===")

        try:
            result = runner.run_cycle()
        except KeyboardInterrupt:
            runner.shutdown()
        except Exception as e:
            svc.logger.error(f"Cycle error: {e}", exc_info=True)
            try:
                from src.engine.state_store import snapshot_chainlink_state
                snapshot_chainlink_state(svc.state, svc.chainlink_feed, slot_mgr=svc.slot_mgr)
                svc.state_store.save(svc.state)
            except Exception:
                pass
            result = CycleResult.NO_MARKET

        if args.display_features:
            _log_feature_snapshot(svc)

        if result == CycleResult.CIRCUIT_BREAKER:
            # Tick prices but suppress analysis until circuit resets
            if runner.has_market():
                ticker_until_next_cycle(
                    svc.client, runner.yes_token_id, runner.no_token_id, svc.interval,
                    strategy=svc.strategy, intra_cycle_analyze_fn=None,
                    logger=svc.logger, btc_feed=svc.btc_feed, slot_mgr=svc.slot_mgr,
                    display_features=args.display_features,
                )
            else:
                sleep_until_next_cycle(svc.interval)
            continue

        if runner.has_market():
            ticker_until_next_cycle(
                svc.client, runner.yes_token_id, runner.no_token_id, svc.interval,
                strategy=svc.strategy,
                intra_cycle_analyze_fn=runner.run_intra_cycle,
                logger=svc.logger, btc_feed=svc.btc_feed, slot_mgr=svc.slot_mgr,
                display_features=args.display_features,
            )
        else:
            sleep_until_next_cycle(svc.interval)


if __name__ == "__main__":
    main()
