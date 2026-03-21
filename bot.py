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
import logging
import math
import os
import requests
import signal
import sys
import threading
import time
import yaml
from datetime import date
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.api.client import PolymarketClient
from src.engine.execution import ExecutionTracker
from src.engine.inventory import InventoryState
from src.engine.risk_manager import RiskLimits, RiskManager
from src.engine.state_store import BotState, StateStore
from src.strategies.base import Signal
from src.strategies.btc_updown import BTCUpDownStrategy
from src.strategies.btc_vol_reversion import BTCVolatilityReversionStrategy
from src.strategies.coin_toss import CoinTossStrategy
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.market_utils import cancel_if_exists, get_mid_price

# ---------------------------------------------------------------------------
# Globals (written by signal handlers, read by main loop)
# ---------------------------------------------------------------------------
_shutdown_requested = False
_client: Optional[PolymarketClient] = None
_state: Optional[BotState] = None
_state_store: Optional[StateStore] = None
_paper_trading: bool = True

# Session PnL counters — reset each run, not persisted
_session_wins: int = 0
_session_losses: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_fill_to_state(
    inv: "InventoryState",
    side: str,
    price: float,
    size: float,
    state: "BotState",
    risk_manager: "RiskManager",
) -> float:
    """Apply a fill to inventory + state PnL. Returns realized PnL."""
    realized = inv.apply_fill(side, price, size)
    state.daily_realized_pnl += realized
    if realized != 0.0:
        risk_manager.record_trade(realized)
    return realized


def _sync_inventories_to_state(
    state: "BotState", inventories: Dict[str, "InventoryState"]
) -> None:
    """Serialize all inventory states into the bot state dict."""
    for token_id, inv in inventories.items():
        state.inventories[token_id] = {
            "position": inv.position,
            "avg_cost": inv.avg_cost,
        }


def _sync_strategy_from_inventories(
    strategy, inventories: Dict[str, "InventoryState"], token_ids: tuple
) -> None:
    """Sync strategy position state from authoritative inventory."""
    for tid in token_ids:
        inv = inventories.get(tid)
        if inv and inv.position > 0:
            strategy.sync_position_from_inventory(tid, inv.position, inv.avg_cost)
            return
    strategy.sync_position_from_inventory(None, 0.0, 0.0)


def sleep_until_next_cycle(interval_s: int) -> None:
    """Sleep until the next wall-clock-aligned cycle boundary (no drift)."""
    now = time.time()
    next_run = math.ceil(now / interval_s) * interval_s
    delay = max(0.0, next_run - now)
    if delay > 0:
        time.sleep(delay)


def _fetch_book_top(yes_token_id: str, no_token_id: str, logger=None) -> tuple:
    """Fetch best bid/ask for YES and NO tokens via /book.

    Returns (yes_bid, yes_ask, no_bid, no_ask).
    Also fetches CLOB midpoint as a fallback when the book spread is wide.
    """
    def _top(token_id: str):
        bid, ask, mid = None, None, None
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

        # Fetch CLOB midpoint — more reliable than (bid+ask)/2 for wide spreads
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

        # Use midpoint as bid/ask when book is too wide or empty
        if mid is not None:
            if bid is None or (ask is not None and (ask - bid) > 0.50):
                bid = mid
            if ask is None or (bid is not None and (ask - bid) > 0.50):
                ask = mid

        return bid, ask

    yes_bid, yes_ask = _top(yes_token_id)
    no_bid,  no_ask  = _top(no_token_id)
    return yes_bid, yes_ask, no_bid, no_ask


def ticker_until_next_cycle(
    client: PolymarketClient,
    yes_token_id: str,
    no_token_id: str,
    interval_s: int,
    strategy=None,
    intra_cycle_analyze_fn=None,
    logger=None,
) -> None:
    """
    Record prices in the background while waiting for the next cycle.

    Prints a single static status line after the first price fetch,
    then sleeps quietly until the next wall-clock boundary.
    Background thread continues recording prices and running
    intra-cycle analysis every 30 s.
    """
    yes_bid: List[Optional[float]] = [None]
    yes_ask: List[Optional[float]] = [None]
    no_bid:  List[Optional[float]] = [None]
    no_ask:  List[Optional[float]] = [None]
    stop_evt = threading.Event()

    def _fetch_loop():
        last_analyzed = time.monotonic()
        while not stop_evt.is_set():
            y_bid, y_ask, n_bid, n_ask = _fetch_book_top(
                yes_token_id, no_token_id, logger=logger,
            )

            if y_bid is not None: yes_bid[0] = y_bid
            if y_ask is not None: yes_ask[0] = y_ask
            if n_bid is not None: no_bid[0]  = n_bid
            if n_ask is not None: no_ask[0]  = n_ask

            if y_bid is not None and y_ask is not None and strategy:
                strategy.record_price(yes_token_id, (y_bid + y_ask) / 2)
            if n_bid is not None and n_ask is not None and strategy:
                strategy.record_price(no_token_id, (n_bid + n_ask) / 2)

            now_m = time.monotonic()
            if intra_cycle_analyze_fn and (now_m - last_analyzed) >= 30:
                intra_cycle_analyze_fn()
                last_analyzed = now_m

            stop_evt.wait(1.0)

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

        # Current mid for the held side
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
        pnl_pct    = (cur_mid - entry_price) / entry_price * 100 if entry_price else 0.0
        return (
            f"POS: {outcome} {entry_size:.1f}sh @{entry_price:.4f} "
            f"uPnL={unrealized:+.2f} ({pnl_pct:+.1f}%)"
        )

    # Wait for first price fetch, then log status every 30 seconds until next cycle
    time.sleep(2)
    last_logged = 0.0

    def _log_status():
        now = time.time()
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

    # Compute the target boundary ONCE — break when we reach it.
    _now = time.time()
    target_boundary = math.ceil(_now / interval_s) * interval_s

    # Sleep in 1-second increments, refreshing status every 30 s
    try:
        while not _shutdown_requested:
            now = time.time()
            if now >= target_boundary:
                break
            if now - last_logged >= 30.0:
                _log_status()
                last_logged = now
            time.sleep(1.0)
    finally:
        stop_evt.set()


def _execute_signals(
    signals: list,
    client: PolymarketClient,
    strategy,
    risk_manager: RiskManager,
    state: BotState,
    current_market_id: str,
    yes_token_id: str,
    no_token_id: str,
    balance: float,
    positions: list,
    paper_trading: bool,
    logger,
    inventories: Dict[str, "InventoryState"] = None,
) -> None:
    """Risk-gate BUY signals and place orders for all passing signals."""
    for sig in signals:
        if sig.action == "BUY":
            if not strategy.should_enter(sig):
                logger.debug(f"BUY rejected by strategy: {sig.reason}")
                continue
            ok, reason = risk_manager.validate_signal(sig, balance, positions, [])
            if not ok:
                logger.info(f"BUY rejected by risk manager: {reason}")
                continue
        # SELL signals always proceed

        token_id = yes_token_id if sig.outcome == "YES" else no_token_id

        existing_order_id = state.active_order_ids.get(token_id)
        if existing_order_id:
            logger.info(f"Cancelling stale order {existing_order_id}")
            cancel_if_exists(client, existing_order_id, dry_run=paper_trading)
            state.active_order_ids[token_id] = None

        try:
            order = client.place_order(
                market_id=current_market_id,
                token_id=token_id,
                outcome=sig.outcome,
                side=sig.action,
                price=sig.price,
                size=sig.size,
            )
            state.active_order_ids[token_id] = order.order_id
            state.strategy_last_signal = (
                f"{sig.action} {sig.outcome} @ {sig.price:.4f} | {sig.reason[:60]}"
            )
            state.strategy_last_signal_ts = time.time()
            if sig.action == "BUY":
                state.strategy_status = "POSITION_OPEN"
            elif sig.action == "SELL":
                state.strategy_status = "EXITED"

            # Paper trading: apply fill directly to inventory (skip reconciliation dance)
            if paper_trading and inventories is not None:
                inv = inventories.get(token_id)
                if inv is None:
                    inv = InventoryState(token_id=token_id)
                    inventories[token_id] = inv
                _apply_fill_to_state(inv, sig.action, sig.price, sig.size, state, risk_manager)
                _sync_inventories_to_state(state, inventories)
            if sig.action == "BUY":
                cost = sig.price * sig.size
                logger.info(
                    f"BUY  {sig.outcome:3s}  {sig.size:.2f}sh @ {sig.price:.4f}"
                    f"  (cost ${cost:.2f})  [{sig.reason}]"
                )
            else:
                entry_price = getattr(strategy, "_pending_exit_entry_price", None)
                if entry_price and entry_price > 0 and sig.size:
                    realized = (sig.price - entry_price) * sig.size
                    pct      = (sig.price - entry_price) / entry_price * 100
                    logger.info(
                        f"SELL {sig.outcome:3s}  {sig.size:.2f}sh @ {sig.price:.4f}"
                        f"  PnL: {realized:+.2f} ({pct:+.1f}%)  [{sig.reason}]"
                    )
                else:
                    logger.info(
                        f"SELL {sig.outcome:3s}  {sig.size:.2f}sh @ {sig.price:.4f}"
                        f"  [{sig.reason}]"
                    )
        except Exception as e:
            logger.error(f"Order placement failed: {e}")


def _snapshot_strategy_state(strategy, state: "BotState") -> None:
    """Write strategy internal state into BotState fields for dashboard visibility."""
    state.strategy_name = getattr(strategy, "name", "")
    active_tid = getattr(strategy, "active_token_id", None)
    state.strategy_status = "POSITION_OPEN" if active_tid else "WATCHING"

    # Bias (btc_updown)
    bias_obj = getattr(strategy, "current_bias", None)
    state.strategy_bias = bias_obj.value if bias_obj is not None else ""

    # Momentum % (btc_updown)
    yes_tid = getattr(strategy, "_yes_token_id", None)
    lookback_fn = getattr(strategy, "_lookback_mid", None)
    price_history = getattr(strategy, "_price_history", {})
    if yes_tid and lookback_fn and yes_tid in price_history:
        now_mono = time.monotonic()
        window = getattr(strategy, "confirmation_window_seconds", 60)
        hist = price_history.get(yes_tid, [])
        mid_now = hist[-1][1] if hist else None
        mid_prev = lookback_fn(yes_tid, now_mono, window)
        if mid_now and mid_prev and mid_prev > 0:
            state.strategy_momentum_pct = (mid_now - mid_prev) / mid_prev
        else:
            state.strategy_momentum_pct = None
    else:
        state.strategy_momentum_pct = None

    # Z-score (btc_vol_reversion)
    compute_z = getattr(strategy, "_compute_zscore", None)
    if compute_z and yes_tid:
        try:
            state.strategy_zscore = compute_z(yes_tid, time.monotonic())
        except Exception:
            state.strategy_zscore = None
    else:
        state.strategy_zscore = None


def _make_intra_cycle_fn(
    client: PolymarketClient,
    strategy,
    risk_manager: RiskManager,
    state: BotState,
    state_store: StateStore,
    market_id: str,
    yes_tid: str,
    no_tid: str,
    paper_trading: bool,
    logger,
    execution_tracker=None,
    inventories=None,
):
    """Return a zero-arg callable for intra-cycle analyze+execute passes."""
    captured_slot = int(math.floor(time.time() / 300) * 300)
    slot_expiry_ts = captured_slot + 300

    def _fn():
        try:
            now_wall = time.time()
            if int(math.floor(now_wall / 300) * 300) != captured_slot:
                logger.debug("Intra-cycle skipped: market slot rolled over")
                return

            time_remaining = slot_expiry_ts - now_wall
            min_entry_window = getattr(strategy, 'max_hold_seconds', 240)

            # Reconcile orders + positions → update inventories
            if execution_tracker and inventories is not None:
                closed_orders = execution_tracker.reconcile(
                    client, [yes_tid, no_tid], inventories
                )
                for order in closed_orders:
                    tid = order.token_id
                    if state.active_order_ids.get(tid) == order.order_id:
                        state.active_order_ids[tid] = None
                for fill in execution_tracker.inferred_fills:
                    inv = inventories.get(fill.token_id)
                    if inv:
                        _apply_fill_to_state(inv, fill.side, fill.price, fill.size, state, risk_manager)
                _sync_inventories_to_state(state, inventories)
                _sync_strategy_from_inventories(strategy, inventories, (yes_tid, no_tid))

            yes_book  = client.get_order_book(yes_tid)
            no_book   = client.get_order_book(no_tid)
            positions = client.get_positions()
            balance   = client.get_balance()
            market_data = {
                "markets": [],
                "order_books": {yes_tid: yes_book, no_tid: no_book},
                "positions": positions,
                "balance": balance,
                "price_history": {},
            }
            signals = strategy.analyze(market_data)

            if time_remaining < min_entry_window:
                buy_count = sum(1 for s in signals if s.action == "BUY")
                signals = [s for s in signals if s.action == "SELL"]
                if buy_count:
                    logger.info(
                        f"Intra-cycle: BUY suppressed ({time_remaining:.0f}s < "
                        f"{min_entry_window}s window) — exits only"
                    )
                    strategy._reset_position_state()

            if signals:
                logger.info(f"Intra-cycle: {len(signals)} signal(s)")
                _execute_signals(
                    signals, client, strategy, risk_manager, state,
                    market_id, yes_tid, no_tid, balance, positions,
                    paper_trading, logger, inventories=inventories,
                )
                _snapshot_strategy_state(strategy, state)
                state_store.save(state)
        except Exception as e:
            logger.error(f"Intra-cycle analyze error: {e}", exc_info=True)
    return _fn


def _parse_event_market(event: dict) -> Optional[Dict]:
    """Extract market_id / yes_token_id / no_token_id from a gamma-api event dict."""
    for m in event.get("markets", []):
        tids = json.loads(m.get("clobTokenIds", "[]"))
        yes_id = tids[0] if len(tids) > 0 else m.get("yes_token_id")
        no_id  = tids[1] if len(tids) > 1 else m.get("no_token_id")
        if yes_id and no_id:
            return {
                "market_id": m.get("id") or m.get("market_id", ""),
                "yes_token_id": yes_id,
                "no_token_id": no_id,
            }
    return None


def find_btc_updown_market(
    keywords: List[str],
    min_volume: int,
    logger,
) -> Optional[Dict]:
    """
    Discover the current BTC Up/Down 5-minute market.

    Discovery order:
      1. Slug-based lookup: gamma-api /events?slug=btc-updown-5m-{slot}
         The slot timestamp = floor(now / 300) * 300 — changes every 5 min.
      2. Keyword search across gamma-api events (min_volume filter).
      3. Keyword search across gamma-api /markets endpoint.

    Returns dict {market_id, yes_token_id, no_token_id} or None.
    """
    lower_kws = [k.lower() for k in keywords]

    # --- 1. Slug-based lookup (fastest, most precise) ---
    slot = int(math.floor(time.time() / 300) * 300)
    slug = f"btc-updown-5m-{slot}"
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        events = data if isinstance(data, list) else [data]
        for event in events:
            result = _parse_event_market(event)
            if result:
                logger.info(
                    f"Found market via slug '{slug}': "
                    f"yes={result['yes_token_id'][:12]}..."
                )
                return result
    except Exception as e:
        logger.warning(f"Slug lookup failed ({slug}): {e}")

    # --- 2. Keyword search across events (min_volume=1M) ---
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"active": "true", "closed": "false", "limit": 100, "offset": 0},
            timeout=10,
        )
        resp.raise_for_status()
        for event in resp.json():
            title = event.get("title", "").lower()
            total_vol = sum(
                float(m.get("volume", 0) or 0) for m in event.get("markets", [])
            )
            if total_vol < min_volume:
                continue
            if all(kw in title for kw in lower_kws):
                result = _parse_event_market(event)
                if result:
                    logger.info(
                        f"Found market via keyword search '{event['title']}': "
                        f"yes={result['yes_token_id'][:12]}..."
                    )
                    return result
    except Exception as e:
        logger.warning(f"Keyword/events search failed: {e}")

    # --- 3. gamma-api /markets endpoint ---
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 100, "offset": 0},
            timeout=10,
        )
        resp.raise_for_status()
        for m in resp.json():
            question = m.get("question", "").lower()
            if all(kw in question for kw in lower_kws):
                tids = json.loads(m.get("clobTokenIds", "[]"))
                yes_id = tids[0] if len(tids) > 0 else None
                no_id  = tids[1] if len(tids) > 1 else None
                if yes_id and no_id:
                    logger.info(
                        f"Found market via gamma-api/markets: '{m['question']}'"
                    )
                    return {
                        "market_id": m.get("id", ""),
                        "yes_token_id": yes_id,
                        "no_token_id": no_id,
                    }
    except Exception as e:
        logger.warning(f"gamma-api/markets search failed: {e}")

    logger.warning("BTC Up/Down market not found this cycle")
    return None


def check_daily_reset(state: BotState, risk_manager: RiskManager, logger) -> None:
    """Reset daily PnL tracking when the calendar date rolls over."""
    today = str(date.today())
    if state.daily_reset_date != today:
        logger.info(
            f"New day {today} — resetting daily PnL "
            f"(previous: {state.daily_realized_pnl:.2f})"
        )
        state.daily_realized_pnl = 0.0
        state.daily_reset_date = today
        risk_manager.reset_daily()


def graceful_shutdown(
    state: BotState,
    state_store: StateStore,
    client: PolymarketClient,
    paper_trading: bool,
    logger,
) -> None:
    """Cancel all resting orders, save state, log summary, exit."""
    logger.info("Graceful shutdown initiated")
    for token_id, order_id in list(state.active_order_ids.items()):
        if order_id:
            logger.info(f"Cancelling resting order {order_id} for token {token_id[:12]}...")
            cancel_if_exists(client, order_id, dry_run=paper_trading)
    state_store.save(state)
    trades = _session_wins + _session_losses
    win_rate = _session_wins / trades if trades > 0 else 0.0
    logger.info(
        f"Shutdown complete | cycles={state.cycle_count} "
        f"| daily_pnl={state.daily_realized_pnl:+.4f} "
        f"| session trades={trades} wins={_session_wins} ({win_rate:.0%} win rate)"
    )
    sys.exit(0)


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _shutdown_requested, _client, _state, _state_store, _paper_trading
    global _session_wins, _session_losses

    parser = argparse.ArgumentParser(description="Polymarket trading bot")
    parser.add_argument(
        "--strategy",
        choices=["btc_updown", "btc_vol_reversion", "coin_toss"],
        default="btc_updown",
        help="Strategy to run (default: btc_updown)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=None,
        help="Force paper trading mode",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=None,
        help="Force live trading mode",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------------
    cfg = load_config("config/config.yaml")

    raw_yaml: dict = {}
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r") as f:
            raw_yaml = yaml.safe_load(f) or {}

    bot_cfg = raw_yaml.get("btc_updown_bot", {})
    market_keywords: List[str] = bot_cfg.get("market_keywords", ["Bitcoin", "Up or Down"])
    state_file: str = bot_cfg.get("state_file", "bot_state.json")
    min_volume: int = int(bot_cfg.get("min_volume", 1_000_000))
    interval: int = int(raw_yaml.get("trading", {}).get("interval", 300))
    paper_trading: bool = cfg.paper_trading
    if args.live:
        paper_trading = False
    elif args.paper:
        paper_trading = True
    _paper_trading = paper_trading

    strategy_cfg: dict = cfg.strategies.get(args.strategy, {})

    # -----------------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------------
    log_file = raw_yaml.get("logging", {}).get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger(level=cfg.log_level, log_file=log_file)
    logger.info(
        f"Starting bot | strategy={args.strategy} | paper_trading={paper_trading} | interval={interval}s"
    )

    # -----------------------------------------------------------------------
    # Core components
    # -----------------------------------------------------------------------
    client = PolymarketClient(
        private_key=cfg.private_key,
        funder_address=cfg.funder_address,
        host=cfg.api_host,
        chain_id=cfg.chain_id,
        paper_trading=paper_trading,
    )
    _client = client

    risk_limits = RiskLimits(
        max_position_size=cfg.risk_limits.get("max_position_size", 200.0),
        max_position_pct=cfg.risk_limits.get("max_position_pct", 0.05),
        max_total_exposure=cfg.risk_limits.get("max_total_exposure", 0.15),
        max_daily_loss=cfg.risk_limits.get("max_daily_loss", 0.05),
        max_exposure_per_market=cfg.risk_limits.get("max_exposure_per_market", 0.10),
        stop_loss_pct=cfg.risk_limits.get("stop_loss_pct", 0.12),
        circuit_breaker_enabled=cfg.risk_limits.get("circuit_breaker_enabled", True),
        circuit_breaker_threshold=cfg.risk_limits.get("circuit_breaker_threshold", 0.20),
    )
    risk_manager = RiskManager(limits=risk_limits)

    execution_tracker = ExecutionTracker(
        orders_sync_interval_s=5.0,       # reconcile orders every 5s (intra-cycle needs fast detection)
        positions_sync_interval_s=10.0,   # reconcile positions every 10s
    )

    if args.strategy == "coin_toss":
        strategy = CoinTossStrategy(config=strategy_cfg)
    elif args.strategy == "btc_vol_reversion":
        strategy = BTCVolatilityReversionStrategy(config=strategy_cfg)
    else:
        strategy = BTCUpDownStrategy(config=strategy_cfg)

    # -----------------------------------------------------------------------
    # State + inventories
    # -----------------------------------------------------------------------
    state_store = StateStore(path=state_file)
    _state_store = state_store
    state = state_store.load()
    _state = state

    inventories: Dict[str, InventoryState] = {}
    for token_id, inv_data in state.inventories.items():
        inv = InventoryState(token_id=token_id)
        inv.position = float(inv_data.get("position", 0.0))
        inv.avg_cost = float(inv_data.get("avg_cost", 0.0))
        inventories[token_id] = inv

    logger.info(
        f"State loaded | cycle_count={state.cycle_count} "
        f"| daily_pnl={state.daily_realized_pnl:.4f}"
    )

    # Track current market so we can detect rollover
    current_market_id: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    intra_cycle_analyze_fn = None

    # -----------------------------------------------------------------------
    # Signal handlers
    # -----------------------------------------------------------------------
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGTSTP, _handle_signal)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while True:
        if _shutdown_requested:
            graceful_shutdown(state, state_store, client, paper_trading, logger)

        cycle_start = time.monotonic()
        logger.info(f"=== Main loop iteration (cycle {state.cycle_count + 1}) ===")

        try:
            # ------------------------------------------------------------------
            # Discover market every cycle — ID changes every 5 minutes
            # ------------------------------------------------------------------
            market_info = find_btc_updown_market(market_keywords, min_volume, logger)
            if not market_info:
                logger.warning("Skipping cycle: market not found")
                sleep_until_next_cycle(interval)
                continue

            new_market_id = market_info["market_id"]

            # Detect rollover: clear stale order IDs for old tokens
            if new_market_id != current_market_id and current_market_id:
                logger.info(
                    f"Market rolled over: {current_market_id} → {new_market_id}"
                )
                # Old token order IDs are now stale — clear them
                for old_tid in (yes_token_id, no_token_id):
                    state.active_order_ids.pop(old_tid, None)
                # Clear execution tracker state for old tokens
                for old_tid in (yes_token_id, no_token_id):
                    execution_tracker.active_orders = {
                        oid: o for oid, o in execution_tracker.active_orders.items()
                        if o.token_id != old_tid
                    }
                    execution_tracker._last_positions_by_token.pop(old_tid, None)

            current_market_id = new_market_id
            yes_token_id = market_info["yes_token_id"]
            no_token_id  = market_info["no_token_id"]
            logger.debug(
                f"Cycle {state.cycle_count}: market={current_market_id[:8]} "
                f"yes={yes_token_id[:8]}"
            )

            # ------------------------------------------------------------------
            balance = client.get_balance()
            check_daily_reset(state, risk_manager, logger)

            if risk_manager.check_circuit_breaker(balance):
                logger.warning(
                    f"Circuit breaker active (balance={balance:.2f}) — skipping cycle"
                )
                # intra_cycle_analyze_fn is not yet built for this cycle; pass None
                # so the ticker only records prices but fires no signals.
                ticker_until_next_cycle(
                    client, yes_token_id, no_token_id, interval,
                    strategy=strategy, intra_cycle_analyze_fn=None,
                    logger=logger,
                )
                continue

            # ------------------------------------------------------------------
            # Reconcile open orders + positions
            # ------------------------------------------------------------------
            closed_orders = execution_tracker.reconcile(
                client, [yes_token_id, no_token_id], inventories
            )

            for order in closed_orders:
                tid = order.token_id
                if state.active_order_ids.get(tid) == order.order_id:
                    state.active_order_ids[tid] = None

            for fill in execution_tracker.inferred_fills:
                inv = inventories.get(fill.token_id)
                if inv:
                    realized = _apply_fill_to_state(inv, fill.side, fill.price, fill.size, state, risk_manager)
                    if realized > 0:
                        _session_wins += 1
                    elif realized < 0:
                        _session_losses += 1

            _sync_inventories_to_state(state, inventories)

            # ------------------------------------------------------------------
            # Sync strategy position state with inventory truth
            # ------------------------------------------------------------------
            _sync_strategy_from_inventories(strategy, inventories, (yes_token_id, no_token_id))

            # ------------------------------------------------------------------
            # Fetch order books and positions
            # ------------------------------------------------------------------
            yes_book  = client.get_order_book(yes_token_id)
            no_book   = client.get_order_book(no_token_id)
            positions = client.get_positions()

            # ------------------------------------------------------------------
            # Orphaned position cleanup: both YES and NO open simultaneously
            # (can happen after a crash/restart with a prior manual trade).
            # Emit SELL for both sides before handing off to the strategy.
            # ------------------------------------------------------------------
            by_token_live = {p.token_id: p for p in positions}
            yes_live = by_token_live.get(yes_token_id)
            no_live  = by_token_live.get(no_token_id)
            if yes_live and yes_live.size > 0 and no_live and no_live.size > 0:
                logger.warning(
                    f"Orphaned positions detected — YES: {yes_live.size:.2f}, "
                    f"NO: {no_live.size:.2f}. Emitting cleanup SELLs."
                )
                orphan_sigs = []
                for tok_id, outcome, live_pos, book in [
                    (yes_token_id, "YES", yes_live, yes_book),
                    (no_token_id,  "NO",  no_live,  no_book),
                ]:
                    mid = get_mid_price(book) if book else None
                    best_bid = book.bids[0].price if book and book.bids else mid
                    if best_bid:
                        orphan_sigs.append(Signal(
                            market_id=current_market_id,
                            outcome=outcome,
                            action="SELL",
                            confidence=1.0,
                            price=best_bid,
                            size=float(live_pos.size),
                            reason="orphan_cleanup",
                        ))
                if orphan_sigs:
                    _execute_signals(
                        orphan_sigs, client, strategy, risk_manager, state,
                        current_market_id, yes_token_id, no_token_id,
                        balance, positions, paper_trading, logger,
                        inventories=inventories,
                    )
                    state_store.save(state)
                    positions = client.get_positions()  # refresh after cleanup

            market_data = {
                "markets": [],
                "order_books": {
                    yes_token_id: yes_book,
                    no_token_id:  no_book,
                },
                "positions": positions,
                "balance": balance,
                "price_history": {},
            }

            # ------------------------------------------------------------------
            # Strategy → signals
            # ------------------------------------------------------------------
            strategy.set_tokens(current_market_id, yes_token_id, no_token_id)
            intra_cycle_analyze_fn = _make_intra_cycle_fn(
                client, strategy, risk_manager, state, state_store,
                current_market_id, yes_token_id, no_token_id, paper_trading, logger,
                execution_tracker=execution_tracker, inventories=inventories,
            )
            signals = strategy.analyze(market_data)

            # Guard: suppress BUY entries if insufficient time remains in market window
            _slot_expiry = (int(math.floor(time.time() / 300) * 300) + 300)
            _time_remaining = _slot_expiry - time.time()
            _min_entry = getattr(strategy, 'max_hold_seconds', 240)
            if _time_remaining < _min_entry:
                _original = len(signals)
                signals = [s for s in signals if s.action == "SELL"]
                if len(signals) < _original:
                    logger.info(
                        f"Entry suppressed at cycle start: {_time_remaining:.0f}s < "
                        f"{_min_entry}s required window"
                    )
                    # Undo optimistic position state set by the strategy
                    strategy._reset_position_state()

            _execute_signals(
                signals, client, strategy, risk_manager, state,
                current_market_id, yes_token_id, no_token_id,
                balance, positions, paper_trading, logger,
                inventories=inventories,
            )

            # ------------------------------------------------------------------
            state.cycle_count += 1
            _snapshot_strategy_state(strategy, state)
            state_store.save(state)

            elapsed = time.monotonic() - cycle_start
            logger.info(
                f"Cycle {state.cycle_count} done | market={current_market_id} "
                f"| balance={balance:.2f} | daily_pnl={state.daily_realized_pnl:.4f} "
                f"| signals={len(signals)} | elapsed={elapsed:.1f}s"
            )

        except KeyboardInterrupt:
            graceful_shutdown(state, state_store, client, paper_trading, logger)

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            try:
                state_store.save(state)
            except Exception:
                pass

        # Show live prices every second while waiting for next cycle.
        # Falls back to plain sleep if we don't have token IDs yet.
        if yes_token_id and no_token_id:
            ticker_until_next_cycle(
                client, yes_token_id, no_token_id, interval,
                strategy=strategy,
                intra_cycle_analyze_fn=intra_cycle_analyze_fn,
                logger=logger,
            )
        else:
            sleep_until_next_cycle(interval)


if __name__ == "__main__":
    main()
