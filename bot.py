"""
BTC Up/Down Bot — wall-clock-aligned 5-minute trading loop.

Usage:
    python bot.py

Configuration: config/config.yaml
    Set paper_trading: false and provide PRIVATE_KEY / PROXY_FUNDER env vars
    for live trading.
"""

import json
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
from src.strategies.btc_updown import BTCUpDownStrategy
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sleep_until_next_cycle(interval_s: int) -> None:
    """Sleep until the next wall-clock-aligned cycle boundary (no drift)."""
    now = time.time()
    next_run = math.ceil(now / interval_s) * interval_s
    delay = max(0.0, next_run - now)
    if delay > 0:
        time.sleep(delay)


def _fetch_book_top(yes_token_id: str, no_token_id: str) -> tuple:
    """Fetch best bid/ask for YES and NO tokens via /book. Returns (yes_bid, yes_ask, no_bid, no_ask)."""
    def _top(token_id: str):
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
            bid = float(bids[0]["p"]) if bids else None
            ask = float(asks[0]["p"]) if asks else None
            return bid, ask
        except Exception:
            return None, None

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
) -> None:
    """
    Print a live one-line price ticker every second while waiting for the
    next cycle boundary.

    Price fetches run in a background thread every 5 s so network latency
    never affects the 1-second countdown cadence.  If `strategy` is provided,
    each fetch also calls strategy.record_price() to populate sub-cycle history.

    If `intra_cycle_analyze_fn` is provided, it is called every 30 s inside
    the fetch thread so signals can fire mid-cycle once price history exists.

    Example output (overwrites same line):
      YES b=0.5100 a=0.5200 | NO b=0.4790 a=0.4880 | Next cycle in 04:27
    """
    yes_bid: List[Optional[float]] = [None]
    yes_ask: List[Optional[float]] = [None]
    no_bid:  List[Optional[float]] = [None]
    no_ask:  List[Optional[float]] = [None]
    stop_evt = threading.Event()

    def _fetch_loop():
        last_analyzed = time.monotonic()
        while not stop_evt.is_set():
            y_bid, y_ask, n_bid, n_ask = _fetch_book_top(yes_token_id, no_token_id)

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

            stop_evt.wait(1.0)  # 1-second cadence (was 5.0)

    fetcher = threading.Thread(target=_fetch_loop, daemon=True)
    fetcher.start()

    def _fmt(v): return f"{v:.4f}" if v is not None else " --- "

    try:
        while True:
            now = time.time()
            next_run = math.ceil(now / interval_s) * interval_s
            remaining = next_run - now
            if remaining <= 0:
                break

            mins, secs = divmod(int(remaining), 60)

            print(
                f"\r  YES b={_fmt(yes_bid[0])} a={_fmt(yes_ask[0])}"
                f" | NO b={_fmt(no_bid[0])} a={_fmt(no_ask[0])}"
                f" | Next cycle in {mins:02d}:{secs:02d}  ",
                end="",
                flush=True,
            )
            time.sleep(1.0)
    finally:
        stop_evt.set()
        print()  # newline so the next log line isn't clobbered


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
            logger.info(
                f"Order placed | {sig.action} {sig.outcome} "
                f"size={sig.size:.2f} price={sig.price:.4f} "
                f"order_id={order.order_id} | {sig.reason}"
            )
        except Exception as e:
            logger.error(f"Order placement failed: {e}")


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
):
    """Return a zero-arg callable for intra-cycle analyze+execute passes."""
    def _fn():
        try:
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
            if signals:
                logger.info(f"Intra-cycle: {len(signals)} signal(s)")
                _execute_signals(
                    signals, client, strategy, risk_manager, state,
                    market_id, yes_tid, no_tid, balance, positions,
                    paper_trading, logger,
                )
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
    logger.info(
        f"Shutdown complete | cycles={state.cycle_count} "
        f"| daily_pnl={state.daily_realized_pnl:.4f}"
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
    _paper_trading = paper_trading

    strategy_cfg: dict = cfg.strategies.get("btc_updown", {})

    # -----------------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------------
    log_file = raw_yaml.get("logging", {}).get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger(level=cfg.log_level, log_file=log_file)
    logger.info(
        f"Starting BTC Up/Down bot | paper_trading={paper_trading} | interval={interval}s"
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
        orders_sync_interval_s=interval * 0.5,
        positions_sync_interval_s=interval,
    )

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

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while True:
        if _shutdown_requested:
            graceful_shutdown(state, state_store, client, paper_trading, logger)

        cycle_start = time.monotonic()

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
                    realized = inv.apply_fill(fill.side, fill.price, fill.size)
                    state.daily_realized_pnl += realized
                    if realized != 0.0:
                        risk_manager.record_trade(realized)

            for token_id, inv in inventories.items():
                state.inventories[token_id] = {
                    "position": inv.position,
                    "avg_cost": inv.avg_cost,
                }

            # ------------------------------------------------------------------
            # Fetch order books and positions
            # ------------------------------------------------------------------
            yes_book  = client.get_order_book(yes_token_id)
            no_book   = client.get_order_book(no_token_id)
            positions = client.get_positions()

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
            )
            signals = strategy.analyze(market_data)

            _execute_signals(
                signals, client, strategy, risk_manager, state,
                current_market_id, yes_token_id, no_token_id,
                balance, positions, paper_trading, logger,
            )

            # ------------------------------------------------------------------
            state.cycle_count += 1
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
            )
        else:
            sleep_until_next_cycle(interval)


if __name__ == "__main__":
    main()
