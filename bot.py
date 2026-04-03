"""
BTC Up/Down Bot — wall-clock-aligned 5-minute trading loop.

Usage:
    python bot.py

Configuration: config/config.yaml
    Set paper_trading: false and provide PRIVATE_KEY / PROXY_FUNDER env vars
    for live trading.
"""

import argparse
import email.utils
import json
import logging
import math
import os
import requests
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import yaml
from dotenv import load_dotenv
load_dotenv()
from datetime import date
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.api.client import PolymarketClient
from src.engine.cycle_snapshot import BotStatus, CycleSnapshot, SnapshotStore
from src.engine.execution import ExecutionTracker
from src.engine.inventory import InventoryState
from src.engine.risk_manager import RiskLimits, RiskManager
from src.engine.slot_state import SLOT_INTERVAL_S, SlotContext, SlotStateManager
from src.engine.performance_store import PerformanceStore
from src.engine.state_store import BotState, StateStore
from src.models import BTCSigmoidModel, parse_strike_price
from src.models.feature_builder import _realized_vol
from src.strategies.base import Signal
from src.strategies.btc_updown import BTCUpDownStrategy
from src.strategies.btc_updown_xgb import BTCUpDownXGBStrategy
from src.strategies.btc_vol_reversion import BTCVolatilityReversionStrategy
from src.strategies.coin_toss import CoinTossStrategy
from src.strategies.prob_edge import ProbEdgeStrategy
from src.utils.btc_feed import BtcPriceFeed
from src.utils.chainlink_feed import ChainlinkFeed
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
_server_offset: float = 0.0  # server_time - local_time, updated each market discovery

# Session PnL counters — reset each run, not persisted
_session_wins: int = 0
_session_losses: int = 0
_bot_start_ts: float = 0.0

# Structured trade log (JSONL) — set once at startup from config
_trade_log_path: Optional[str] = None
_price_tick_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_price_tick(record: dict) -> None:
    """Append one JSON price-tick record to the probability tick log."""
    if _price_tick_path is None:
        return
    line = json.dumps({k: v for k, v in record.items() if v is not None}, default=str)
    with open(_price_tick_path, "a") as f:
        f.write(line + "\n")
        f.flush()


def _log_trade_jsonl(record: dict) -> None:
    """Append one JSON trade record to the structured trade log."""
    if _trade_log_path is None:
        return
    line = json.dumps({k: v for k, v in record.items() if v is not None}, default=str)
    with open(_trade_log_path, "a") as f:
        f.write(line + "\n")
        f.flush()


def _book_summary(yes_book, no_book) -> dict:
    """Extract top-of-book bid/ask from OrderBook objects."""
    return {
        "yes_bid": yes_book.bids[0].price if yes_book and yes_book.bids else None,
        "yes_ask": yes_book.asks[0].price if yes_book and yes_book.asks else None,
        "no_bid": no_book.bids[0].price if no_book and no_book.bids else None,
        "no_ask": no_book.asks[0].price if no_book and no_book.asks else None,
    }

def _compute_unrealized_pnl(
    inventories: Dict[str, "InventoryState"],
    yes_token_id: str,
    no_token_id: str,
    yes_book,
    no_book,
) -> Optional[float]:
    """Compute total unrealized PnL from live book mids. Returns None if any mid is missing."""
    book_map = {yes_token_id: yes_book, no_token_id: no_book}
    total = 0.0
    has_position = False
    for tid, inv in inventories.items():
        if inv.position == 0:
            continue
        has_position = True
        book = book_map.get(tid)
        mid = get_mid_price(book) if book else None
        if mid is None:
            return None
        total += (mid - inv.avg_cost) * inv.position
    return total if has_position else None


def _build_and_save_snapshot(
    snapshot_store: "SnapshotStore",
    state: "BotState",
    market_id: str,
    question: str,
    yes_token_id: str,
    no_token_id: str,
    strike: Optional[float],
    btc_now: Optional[float],
    yes_book,
    no_book,
    inventories: Dict[str, "InventoryState"],
    execution_tracker: "ExecutionTracker",
    risk_manager: "RiskManager",
    paper_trading: bool,
) -> None:
    """Assemble CycleSnapshot from current cycle state and write atomically."""
    now = time.time()
    start_ts = int(math.floor(now / 300) * 300)
    end_ts = start_ts + 300

    yes_bid = yes_book.bids[0].price if yes_book and yes_book.bids else None
    yes_ask = yes_book.asks[0].price if yes_book and yes_book.asks else None
    no_bid  = no_book.bids[0].price  if no_book  and no_book.bids  else None
    no_ask  = no_book.asks[0].price  if no_book  and no_book.asks  else None
    yes_mid = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else (yes_bid or yes_ask)
    no_mid  = (no_bid  + no_ask)  / 2 if no_bid  and no_ask  else (no_bid  or no_ask)

    pos_dict = {
        tid: {"size": inv.position, "avg_cost": inv.avg_cost}
        for tid, inv in inventories.items()
        if inv.position != 0
    }

    orders = [
        {
            "order_id": o.order_id,
            "token_id": o.token_id,
            "outcome": o.outcome,
            "side": o.side,
            "price": o.price,
            "size": o.size,
            "status": o.status,
            "filled_qty": o.filled_qty,
        }
        for o in execution_tracker.active_orders.values()
    ]

    has_position = any(inv.position != 0 for inv in inventories.values())
    cb_active = getattr(risk_manager, "circuit_breaker_active", False)
    if cb_active:
        status = BotStatus.COOLDOWN.value
    elif has_position:
        status = BotStatus.IN_POSITION.value
    else:
        status = BotStatus.EVALUATING.value

    unrealized = _compute_unrealized_pnl(
        inventories, yes_token_id, no_token_id, yes_book, no_book
    )

    snap = CycleSnapshot(
        market_id=market_id,
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        start_ts=start_ts,
        end_ts=end_ts,
        tte_seconds=max(0.0, end_ts - now),
        strike=strike,
        btc_now=btc_now,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mid=yes_mid,
        no_bid=no_bid,
        no_ask=no_ask,
        no_mid=no_mid,
        positions=pos_dict,
        active_orders=orders,
        bot_status=status,
        daily_realized_pnl=state.daily_realized_pnl,
        slot_realized_pnl=state.slot_realized_pnl,
        unrealized_pnl=unrealized,
        strategy_name=state.strategy_name,
        strategy_status=state.strategy_status,
        strategy_prob_yes=state.strategy_prob_yes,
        strategy_prob_no=state.strategy_prob_no,
        strategy_edge_yes=state.strategy_edge_yes,
        strategy_edge_no=state.strategy_edge_no,
        strategy_net_edge_yes=state.strategy_net_edge_yes,
        strategy_net_edge_no=state.strategy_net_edge_no,
        strategy_expected_fill_yes=state.strategy_expected_fill_yes,
        strategy_expected_fill_no=state.strategy_expected_fill_no,
        strategy_required_edge=state.strategy_required_edge,
        strategy_tte_seconds=state.strategy_tte_seconds,
        strategy_distance_to_break_pct=state.strategy_distance_to_break_pct,
        strategy_distance_to_strike_bps=state.strategy_distance_to_strike_bps,
        strategy_model_version=state.strategy_model_version,
        strategy_feature_status=state.strategy_feature_status,
        strategy_score_breakdown=state.strategy_score_breakdown,
        cycle_count=state.cycle_count,
        paper_trading=paper_trading,
    )
    try:
        snapshot_store.save(snap)
    except Exception:
        pass  # never let snapshot write crash the main loop


_GAMMA_API = "https://gamma-api.polymarket.com"


def _fetch_slot_outcome(slot_ts: int, logger) -> Optional[str]:
    """
    Query Gamma API for the resolution outcome of a specific 5-min slot.
    Returns "Up", "Down", or None (unresolved / API error).
    """
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        resp = requests.get(
            f"{_GAMMA_API}/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
        if not events:
            return None
        market = (events[0].get("markets") or [{}])[0]
        closed = market.get("closed", False)
        raw = market.get("outcomePrices", "")
        outcome_prices: list = json.loads(raw) if isinstance(raw, str) else (raw or [])
        if not closed or len(outcome_prices) < 2:
            return None
        if float(outcome_prices[0]) > 0.9:
            return "Up"
        if float(outcome_prices[1]) > 0.9:
            return "Down"
        return None
    except Exception as exc:
        logger.warning(f"Gamma API error fetching outcome for slot {slot_ts}: {exc}")
        return None


def _settle_expiring_positions(
    yes_token_id: str,
    no_token_id: str,
    slot_ts: int,
    inventories: Dict[str, "InventoryState"],
    state: "BotState",
    risk_manager: "RiskManager",
    paper_trading: bool,
    logger,
) -> None:
    """
    Apply synthetic settlement SELLs for open positions in an expired slot.

    Called at market rollover BEFORE resetting slot_realized_pnl.
    Queries Gamma API for the outcome, then realizes PnL at settlement price
    (0.99 for the winning side, 0.01 for the losing side).
    """
    positions_to_settle = [
        (yes_token_id, "YES"),
        (no_token_id,  "NO"),
    ]
    has_open = any(
        inventories.get(tid, None) is not None and inventories[tid].position > 0
        for tid, _ in positions_to_settle
    )
    if not has_open:
        return

    outcome = _fetch_slot_outcome(slot_ts, logger)
    if outcome is None:
        logger.warning(
            f"Cannot settle slot {slot_ts}: outcome not yet available "
            "(market may still be resolving — positions left open)"
        )
        return

    # YES wins if outcome == "Up"; NO wins if outcome == "Down"
    settlement = {
        "YES": 0.99 if outcome == "Up"   else 0.01,
        "NO":  0.99 if outcome == "Down" else 0.01,
    }

    for token_id, side in positions_to_settle:
        inv = inventories.get(token_id)
        if inv is None or inv.position <= 0:
            continue
        price  = settlement[side]
        size   = inv.position
        realized = _apply_fill_to_state(inv, "SELL", price, size, state, risk_manager)
        if realized > 0:
            state.session_wins += 1
        elif realized < 0:
            state.session_losses += 1
        label = "paper" if paper_trading else "live"
        logger.info(
            f"[{label}] Settlement SELL: {side} {token_id[:8]} "
            f"{size:.2f}sh @ {price:.2f} → realized {realized:+.4f} "
            f"(slot outcome: {outcome})"
        )


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
    state.slot_realized_pnl += realized
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


def _snapshot_chainlink_state(
    state: "BotState",
    chainlink_feed: Optional["ChainlinkFeed"],
    slot_mgr: Optional["SlotStateManager"] = None,
) -> None:
    """Persist the latest Chainlink slot-open snapshot for dashboard recovery."""
    if slot_mgr is not None:
        slot_mgr.sync_to_bot_state(state)
    elif chainlink_feed is not None:
        slot_open = chainlink_feed.get_slot_open_price()
        if slot_open is not None:
            state.chainlink_ref_price = slot_open.price
            state.chainlink_ref_slot_ts = slot_open.slot_ts
        else:
            state.chainlink_ref_price = None
            state.chainlink_ref_slot_ts = None
    else:
        state.chainlink_ref_price = None
        state.chainlink_ref_slot_ts = None
    state.chainlink_healthy = chainlink_feed.is_healthy() if chainlink_feed else False


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


def _fetch_market_data_parallel(client, yes_tid, no_tid):
    """Fetch order books, positions, and balance in parallel."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_yes = pool.submit(client.get_order_book, yes_tid)
        f_no = pool.submit(client.get_order_book, no_tid)
        f_pos = pool.submit(client.get_positions)
        f_bal = pool.submit(client.get_balance)
        return f_yes.result(), f_no.result(), f_pos.result(), f_bal.result()


def _stime() -> float:
    """Return local time corrected by the last-observed Polymarket server offset."""
    return time.time() + _server_offset


def sleep_until_next_cycle(interval_s: int) -> None:
    """Sleep until the next wall-clock-aligned cycle boundary (no drift)."""
    now = _stime()
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

        # Only fetch /midpoint when book is empty or spread is wide
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
    btc_feed: Optional["BtcPriceFeed"] = None,
    slot_mgr: Optional["SlotStateManager"] = None,
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
    first_fetch_evt = threading.Event()

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

            # Wait up to 5 s before the next fetch, but check the stop
            # signal every 1 s so shutdown exits promptly.
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

    # Compute the target boundary ONCE — break when we reach it.
    _now = _stime()
    target_boundary = math.ceil(_now / interval_s) * interval_s

    # Sleep in 1-second increments, refreshing status every 30 s
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
    book_summary: Optional[dict] = None,
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
            state.trade_log.append({
                "ts": time.time(),
                "action": sig.action,
                "outcome": sig.outcome,
                "price": sig.price,
                "size": sig.size,
            })
            state.trade_log = state.trade_log[-20:]

            # -- Structured JSONL trade log --
            _pre_inv = inventories.get(token_id) if inventories else None
            _slot_ts = SlotContext.slot_for(time.time()) + SLOT_INTERVAL_S
            _log_trade_jsonl({
                # timing
                "ts": time.time(),
                "cycle": state.cycle_count,
                "slot_expiry_ts": _slot_ts,
                "seconds_to_expiry": _slot_ts - time.time(),
                # trade
                "action": sig.action,
                "outcome": sig.outcome,
                "price": sig.price,
                "size": sig.size,
                "confidence": sig.confidence,
                "reason": sig.reason,
                "market_id": current_market_id,
                "token_id": token_id,
                "order_id": order.order_id,
                # market snapshot
                **(book_summary or {}),
                "balance": balance,
                # pre-trade position
                "pre_position": _pre_inv.position if _pre_inv else 0,
                "pre_avg_cost": _pre_inv.avg_cost if _pre_inv else 0,
                # pnl
                "daily_pnl": state.daily_realized_pnl,
                "slot_pnl": state.slot_realized_pnl,
                # strategy state
                "strategy_name": state.strategy_name,
                "prob_yes": state.strategy_prob_yes,
                "edge_yes": state.strategy_edge_yes,
                "edge_no": state.strategy_edge_no,
                "bias": state.strategy_bias,
                "momentum_pct": state.strategy_momentum_pct,
                "zscore": state.strategy_zscore,
                "model_version": state.strategy_model_version or None,
                "feature_status": state.strategy_feature_status or None,
                # chainlink
                "chainlink_ref_price": state.chainlink_ref_price,
                "chainlink_healthy": state.chainlink_healthy or None,
                # meta
                "paper_trading": paper_trading,
            })

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

    state.strategy_prob_yes = getattr(strategy, "last_prob_yes", None)
    state.strategy_prob_no = getattr(strategy, "last_prob_no", None)
    state.strategy_edge_yes = getattr(strategy, "last_edge_yes", None)
    state.strategy_edge_no = getattr(strategy, "last_edge_no", None)
    state.strategy_net_edge_yes = getattr(strategy, "last_net_edge_yes", None)
    state.strategy_net_edge_no = getattr(strategy, "last_net_edge_no", None)
    state.strategy_expected_fill_yes = getattr(strategy, "last_expected_fill_yes", None)
    state.strategy_expected_fill_no = getattr(strategy, "last_expected_fill_no", None)
    state.strategy_tte_seconds = getattr(strategy, "last_tte_seconds", None)
    state.strategy_distance_to_break_pct = getattr(strategy, "last_distance_to_break_pct", None)
    state.strategy_distance_to_strike_bps = getattr(strategy, "last_distance_to_strike_bps", None)
    state.strategy_model_version = getattr(strategy, "last_model_version", "")
    state.strategy_feature_status = getattr(strategy, "last_feature_status", "")
    state.strategy_score_breakdown = getattr(strategy, "last_score_breakdown", None)
    state.strategy_required_edge = getattr(strategy, "last_required_edge", None)


def _make_intra_cycle_fn(
    client: PolymarketClient,
    strategy,
    risk_manager: RiskManager,
    state: BotState,
    state_store: StateStore,
    market_id: str,
    yes_tid: str,
    no_tid: str,
    question: str,
    strike_price: Optional[float],
    paper_trading: bool,
    logger,
    execution_tracker=None,
    inventories=None,
    chainlink_feed: Optional["ChainlinkFeed"] = None,
    slot_mgr: Optional["SlotStateManager"] = None,
):
    """Return a zero-arg callable for intra-cycle analyze+execute passes."""
    captured_slot = slot_mgr.current_slot_ts() if slot_mgr else SlotContext.slot_for(time.time())
    slot_expiry_ts = captured_slot + SLOT_INTERVAL_S

    def _fn():
        try:
            now_wall = time.time()
            if SlotContext.slot_for(now_wall) != captured_slot:
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
                        realized = _apply_fill_to_state(inv, fill.side, fill.price, fill.size, state, risk_manager)
                        if realized > 0:
                            state.session_wins += 1
                        elif realized < 0:
                            state.session_losses += 1
                _sync_inventories_to_state(state, inventories)
                _sync_strategy_from_inventories(strategy, inventories, (yes_tid, no_tid))

            # Re-read Chainlink reference price (may change on slot rollover)
            if slot_mgr is not None:
                ctx = slot_mgr.update_from_chainlink(chainlink_feed, fallback_question=question)
                effective_strike = ctx.strike_price if ctx.strike_price is not None else strike_price
            else:
                effective_strike = strike_price
                if chainlink_feed is not None:
                    cl_ref = chainlink_feed.get_reference_price()
                    if cl_ref is not None:
                        effective_strike = cl_ref

            yes_book, no_book, positions, balance = _fetch_market_data_parallel(
                client, yes_tid, no_tid,
            )
            market_data = {
                "markets": [],
                "order_books": {yes_tid: yes_book, no_tid: no_book},
                "positions": positions,
                "balance": balance,
                "price_history": {},
                "question": question,
                "strike_price": effective_strike,
                "slot_expiry_ts": slot_expiry_ts,
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
                    book_summary=_book_summary(yes_book, no_book),
                )
                _snapshot_strategy_state(strategy, state)
                _snapshot_chainlink_state(state, chainlink_feed, slot_mgr=slot_mgr)
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
                "question": m.get("question", event.get("title", "")),
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
    slot = int(math.floor(_stime() / 300) * 300)
    slug = f"btc-updown-5m-{slot}"
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        # Update server-clock offset from HTTP Date: header
        global _server_offset
        date_hdr = resp.headers.get("Date")
        if date_hdr:
            try:
                _server_offset = email.utils.parsedate_to_datetime(date_hdr).timestamp() - time.time()
            except Exception:
                pass
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
                        "question": m.get("question", ""),
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
        state.session_wins = 0
        state.session_losses = 0
        risk_manager.reset_daily()


def _close_open_positions(
    client: PolymarketClient,
    state: BotState,
    market_id: str,
    token_outcome_map: dict,
    logger,
) -> None:
    """SELL any open positions at shutdown. Uses a low limit price (0.01) to sweep all bids."""
    if not market_id or not token_outcome_map:
        logger.warning("Shutdown: no market context — cannot close open positions")
        return

    for token_id, inv in state.inventories.items():
        size = inv.get("position", 0) if isinstance(inv, dict) else getattr(inv, "position", 0)
        if size <= 0:
            continue
        outcome = token_outcome_map.get(token_id, "")
        if not outcome:
            continue
        logger.info(f"Shutdown: closing {outcome} position  {size:.4f}sh @ market-sell")
        try:
            client.place_order(
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                side="SELL",
                price=0.01,
                size=round(size, 4),
            )
        except Exception as exc:
            logger.error(f"Shutdown: failed to close {outcome} position: {exc}")


def graceful_shutdown(
    state: BotState,
    state_store: StateStore,
    client: PolymarketClient,
    paper_trading: bool,
    logger,
    btc_feed: Optional[BtcPriceFeed] = None,
    chainlink_feed: Optional["ChainlinkFeed"] = None,
    current_market_id: str = "",
    token_outcome_map: Optional[dict] = None,
    perf_store: Optional[PerformanceStore] = None,
    strategy_name: str = "",
) -> None:
    """Cancel all resting orders, close open positions, save state, log summary, exit."""
    logger.info("Graceful shutdown initiated")
    for token_id, order_id in list(state.active_order_ids.items()):
        if order_id:
            logger.info(f"Cancelling resting order {order_id} for token {token_id[:12]}...")
            cancel_if_exists(client, order_id, dry_run=paper_trading)
    _close_open_positions(client, state, current_market_id, token_outcome_map or {}, logger)
    if btc_feed is not None:
        btc_feed.stop()
    if chainlink_feed is not None:
        chainlink_feed.stop()
    _snapshot_chainlink_state(state, chainlink_feed)
    state_store.save(state)
    trades = _session_wins + _session_losses
    win_rate = _session_wins / trades if trades > 0 else 0.0
    logger.info(
        f"Shutdown complete | cycles={state.cycle_count} "
        f"| daily_pnl={state.daily_realized_pnl:+.4f} "
        f"| session trades={trades} wins={_session_wins} ({win_rate:.0%} win rate)"
    )
    if perf_store is not None:
        try:
            perf_store.record_session(
                state,
                strategy_name=strategy_name or state.strategy_name,
                start_ts=_bot_start_ts,
                end_ts=time.time(),
                paper_trading=paper_trading,
            )
        except Exception as exc:
            logger.warning(f"Failed to record session to perf.db: {exc}")
    sys.exit(0)


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Pre-launch config display
# ---------------------------------------------------------------------------

_STRATEGIES = ["btc_updown", "btc_updown_xgb", "btc_vol_reversion", "coin_toss", "prob_edge"]


def _prompt_strategy_and_mode(default_strategy: str, default_paper: bool):
    """Interactively ask user to pick strategy and trading mode. Returns (strategy, paper_trading)."""
    print("\nAvailable strategies:")
    for i, s in enumerate(_STRATEGIES, 1):
        marker = " (default)" if s == default_strategy else ""
        print(f"  {i}. {s}{marker}")

    while True:
        try:
            raw = input(f"\nSelect strategy [1-{len(_STRATEGIES)}] or name (Enter = {default_strategy}): ").strip()
        except EOFError:
            return default_strategy, default_paper
        if not raw:
            strategy = default_strategy
            break
        if raw.isdigit() and 1 <= int(raw) <= len(_STRATEGIES):
            strategy = _STRATEGIES[int(raw) - 1]
            break
        if raw in _STRATEGIES:
            strategy = raw
            break
        print(f"  Invalid choice. Enter a number 1–{len(_STRATEGIES)} or a strategy name.")

    default_mode = "paper" if default_paper else "live"
    while True:
        try:
            raw = input(f"Trading mode — paper or live? (Enter = {default_mode}): ").strip().lower()
        except EOFError:
            return strategy, default_paper
        if not raw:
            paper_trading = default_paper
            break
        if raw in ("paper", "p"):
            paper_trading = True
            break
        if raw in ("live", "l"):
            paper_trading = False
            break
        print("  Enter 'paper' or 'live'.")

    return strategy, paper_trading


def _display_and_confirm_config(
    strategy_name: str,
    strategy_cfg: dict,
    risk_cfg: dict,
    paper_trading: bool,
    interval: int,
) -> dict:
    """Display active config and allow inline overrides before launch."""
    W = 48  # inner width between box edges
    mode_str = "PAPER TRADING" if paper_trading else "LIVE TRADING"

    # Detect whether stdout can handle box-drawing chars
    try:
        "╔═╗║╠╣╚╝".encode(sys.stdout.encoding or "utf-8")
        TL, H, TR, V, ML, MR, BL, BR = "╔", "═", "╗", "║", "╠", "╣", "╚", "╝"
    except (UnicodeEncodeError, LookupError):
        TL, H, TR, V, ML, MR, BL, BR = "+", "-", "+", "|", "+", "+", "+", "+"

    def _row(label: str, value: str) -> str:
        content = f"  {label:<15}{value}"
        return f"{V}{content:<{W}}{V}"

    def _header(text: str) -> str:
        content = f"  {text}"
        return f"{V}{content:<{W}}{V}"

    border_top = f"{TL}{H * W}{TR}"
    border_mid = f"{ML}{H * W}{MR}"
    border_bot = f"{BL}{H * W}{BR}"

    lines = [border_top]
    lines.append(_header("Polymarket Bot — Pre-launch Config"))
    lines.append(border_mid)
    lines.append(_row("Mode:", mode_str))
    lines.append(_row("Strategy:", strategy_name))
    lines.append(_row("Interval:", f"{interval}s ({interval // 60}-min cycles)"))
    lines.append(border_mid)
    lines.append(_header("Strategy Parameters"))

    for key in sorted(strategy_cfg.keys()):
        val = strategy_cfg[key]
        lines.append(_row(f"  {key}:", str(val)))

    lines.append(border_mid)
    lines.append(_header("Risk Limits"))
    for key in sorted(risk_cfg.keys()):
        val = risk_cfg[key]
        lines.append(_row(f"  {key}:", str(val)))

    lines.append(border_bot)
    print("\n".join(lines))

    # Override loop
    while True:
        try:
            user_input = input("\nOverride? (e.g. 'stop_loss_pct=0.08', Enter to start)\n> ").strip()
        except EOFError:
            break
        if not user_input:
            break
        if "=" not in user_input:
            print("  Invalid format. Use key=value")
            continue
        key, _, raw_value = user_input.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in strategy_cfg:
            print(f"  Unknown key '{key}'. Valid keys: {', '.join(sorted(strategy_cfg.keys()))}")
            continue
        # Type coercion: int → float → str
        old_value = strategy_cfg[key]
        try:
            new_value = int(raw_value)
        except ValueError:
            try:
                new_value = float(raw_value)
            except ValueError:
                new_value = raw_value
        strategy_cfg[key] = new_value
        print(f"  Updated {key}: {old_value} → {new_value}")

    return strategy_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _shutdown_requested, _client, _state, _state_store, _paper_trading
    global _session_wins, _session_losses, _bot_start_ts
    _bot_start_ts = time.time()

    parser = argparse.ArgumentParser(description="Polymarket trading bot")
    parser.add_argument(
        "--strategy",
        choices=_STRATEGIES,
        default=None,
        help="Strategy to run (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=False,
        help="Force paper trading mode",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Force live trading mode",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        default=False,
        help="Skip interactive config confirmation prompt",
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

    # Determine paper/live mode from config, then CLI flags
    paper_trading: bool = cfg.paper_trading
    if args.live:
        paper_trading = False
    elif args.paper:
        paper_trading = True

    # If no --strategy flag and running interactively, prompt the user
    strategy_name: str
    if args.strategy is None and sys.stdin.isatty() and not args.no_confirm:
        strategy_name, paper_trading = _prompt_strategy_and_mode(
            default_strategy="btc_updown_xgb",
            default_paper=paper_trading,
        )
    else:
        strategy_name = args.strategy or "btc_updown_xgb"

    _paper_trading = paper_trading

    strategy_cfg: dict = cfg.strategies.get(strategy_name, {})

    # -----------------------------------------------------------------------
    # Pre-launch config display + optional override
    # -----------------------------------------------------------------------
    if not args.no_confirm and sys.stdin.isatty():
        strategy_cfg = _display_and_confirm_config(
            strategy_name=strategy_name,
            strategy_cfg=strategy_cfg,
            risk_cfg=raw_yaml.get("risk", {}),
            paper_trading=paper_trading,
            interval=interval,
        )

    # -----------------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------------
    log_file = raw_yaml.get("logging", {}).get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger(level=cfg.log_level, log_file=log_file)

    # Structured trade log (JSONL)
    global _trade_log_path, _price_tick_path
    _trade_log_path = raw_yaml.get("logging", {}).get("trade_log_file")
    if _trade_log_path:
        os.makedirs(os.path.dirname(_trade_log_path), exist_ok=True)
    _price_tick_path = raw_yaml.get("logging", {}).get("price_tick_file")
    if _price_tick_path:
        os.makedirs(os.path.dirname(_price_tick_path), exist_ok=True)

    logger.info(
        f"Starting bot | strategy={strategy_name} | paper_trading={paper_trading} | interval={interval}s"
    )

    # VPN pre-flight check
    from src.utils.vpn import check_vpn
    if raw_yaml.get("vpn", {}).get("require_non_us", True):
        check_vpn(abort_if_us=True)

    # Performance store
    perf_db_path: str = bot_cfg.get("perf_db_path", "perf.db")
    perf_store = PerformanceStore(perf_db_path)

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
        circuit_breaker_enabled=cfg.risk_limits.get("circuit_breaker_enabled", True),
        circuit_breaker_threshold=cfg.risk_limits.get("circuit_breaker_threshold", 0.20),
    )
    risk_manager = RiskManager(limits=risk_limits)
    session_loss_cap: float = raw_yaml.get("risk", {}).get("max_session_loss_usdc", float("inf"))

    execution_tracker = ExecutionTracker(
        orders_sync_interval_s=5.0,       # reconcile orders every 5s (intra-cycle needs fast detection)
        positions_sync_interval_s=10.0,   # reconcile positions every 10s
    )

    btc_feed: Optional[BtcPriceFeed] = None
    if strategy_name == "coin_toss":
        strategy = CoinTossStrategy(config=strategy_cfg)
    elif strategy_name == "btc_updown_xgb":
        btc_feed = BtcPriceFeed(
            symbol=str(strategy_cfg.get("btc_symbol", "BTC-USD")),
            exchange=str(strategy_cfg.get("btc_exchange", "coinbase")),
            logger=logger,
        ).start()
        strategy = BTCUpDownXGBStrategy(config=strategy_cfg, btc_feed=btc_feed, logger=logger)
    elif strategy_name == "prob_edge":
        btc_feed = BtcPriceFeed(
            symbol=str(strategy_cfg.get("btc_symbol", "BTC-USD")),
            exchange=str(strategy_cfg.get("btc_exchange", "coinbase")),
            logger=logger,
        ).start()
        strategy = ProbEdgeStrategy(
            config=strategy_cfg,
            btc_feed=btc_feed,
            model_service=BTCSigmoidModel(logger=logger),
            logger=logger,
        )
    elif strategy_name == "btc_vol_reversion":
        strategy = BTCVolatilityReversionStrategy(config=strategy_cfg)
    else:
        strategy = BTCUpDownStrategy(config=strategy_cfg)

    # -----------------------------------------------------------------------
    # Chainlink reference price feed
    # -----------------------------------------------------------------------
    chainlink_cfg = raw_yaml.get("chainlink_feed", {})
    chainlink_feed: Optional[ChainlinkFeed] = None
    if chainlink_cfg.get("enabled", True):
        chainlink_feed = ChainlinkFeed(
            symbol=chainlink_cfg.get("symbol", "btc/usd"),
            slot_interval_s=interval,
            logger=logger,
        ).start()
        logger.info("Chainlink reference price feed started")

    slot_mgr = SlotStateManager(clock_fn=_stime)

    # -----------------------------------------------------------------------
    # State + inventories
    # -----------------------------------------------------------------------
    state_store = StateStore(path=state_file)
    _state_store = state_store
    state = state_store.load()
    _state = state

    snapshot_store = SnapshotStore(
        path=state_file.replace(".json", "_snapshot.json")
    )

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
    # Sync risk manager with persisted daily PnL so validate_signal() uses
    # the real accumulated loss, not the stale 0.0 from __init__.
    risk_manager.daily_pnl = state.daily_realized_pnl

    # Track current market so we can detect rollover
    current_market_id: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    current_question: str = ""
    current_strike_price: Optional[float] = None
    intra_cycle_analyze_fn = None

    # -----------------------------------------------------------------------
    # Signal handlers
    # -----------------------------------------------------------------------
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _handle_signal)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while True:
        if _shutdown_requested:
            graceful_shutdown(
                state, state_store, client, paper_trading, logger,
                btc_feed=btc_feed, chainlink_feed=chainlink_feed,
                current_market_id=current_market_id,
                token_outcome_map={yes_token_id: "YES", no_token_id: "NO"} if yes_token_id else None,
                perf_store=perf_store, strategy_name=strategy_name,
            )

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
                # Settle any open positions in the expiring slot BEFORE resetting PnL
                ended_slot_ts = SlotContext.slot_for(_stime()) - SLOT_INTERVAL_S
                _settle_expiring_positions(
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    slot_ts=ended_slot_ts,
                    inventories=inventories,
                    state=state,
                    risk_manager=risk_manager,
                    paper_trading=paper_trading,
                    logger=logger,
                )
                _sync_inventories_to_state(state, inventories)

                logger.info(
                    f"Market rolled over: {current_market_id} → {new_market_id} | "
                    f"Slot PnL for {current_market_id[:8]}: {state.slot_realized_pnl:+.4f}"
                )
                state.slot_realized_pnl = 0.0
                # Reset per-slot re-entry counters for strategies that support it
                if hasattr(strategy, "reset_slot_state"):
                    strategy.reset_slot_state()
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
            current_question = market_info.get("question", "")

            # Strike price: prefer Chainlink slot-open, fall back to regex
            _slot_ctx = slot_mgr.update_from_chainlink(
                chainlink_feed, fallback_question=current_question
            )
            current_strike_price = _slot_ctx.strike_price
            logger.info(
                f"Strike from {_slot_ctx.strike_source}: "
                + (f"${current_strike_price:,.2f}" if current_strike_price else "None")
            )

            logger.debug(
                f"Cycle {state.cycle_count}: market={current_market_id[:8]} "
                f"yes={yes_token_id[:8]}"
            )

            # ------------------------------------------------------------------
            balance = client.get_balance()
            check_daily_reset(state, risk_manager, logger)

            if state.slot_realized_pnl <= -abs(session_loss_cap):
                logger.warning(
                    f"Slot loss cap reached ({state.slot_realized_pnl:.2f} <= "
                    f"-{session_loss_cap:.2f}) — shutting down"
                )
                graceful_shutdown(
                state, state_store, client, paper_trading, logger,
                btc_feed=btc_feed, chainlink_feed=chainlink_feed,
                current_market_id=current_market_id,
                token_outcome_map={yes_token_id: "YES", no_token_id: "NO"} if yes_token_id else None,
                perf_store=perf_store, strategy_name=strategy_name,
            )

            if risk_manager.check_circuit_breaker(balance):
                logger.warning(
                    f"Circuit breaker active (balance={balance:.2f}) — skipping cycle"
                )
                # intra_cycle_analyze_fn is not yet built for this cycle; pass None
                # so the ticker only records prices but fires no signals.
                ticker_until_next_cycle(
                    client, yes_token_id, no_token_id, interval,
                    strategy=strategy, intra_cycle_analyze_fn=None,
                    logger=logger, btc_feed=btc_feed, slot_mgr=slot_mgr,
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
                        state.session_wins += 1
                    elif realized < 0:
                        state.session_losses += 1

            _sync_inventories_to_state(state, inventories)

            # ------------------------------------------------------------------
            # Sync strategy position state with inventory truth
            # ------------------------------------------------------------------
            _sync_strategy_from_inventories(strategy, inventories, (yes_token_id, no_token_id))

            # ------------------------------------------------------------------
            # Fetch order books and positions (parallel)
            # ------------------------------------------------------------------
            yes_book, no_book, positions, _bal = _fetch_market_data_parallel(
                client, yes_token_id, no_token_id,
            )

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
                        book_summary=_book_summary(yes_book, no_book),
                    )
                    _snapshot_chainlink_state(state, chainlink_feed, slot_mgr=slot_mgr)
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
                "question": current_question,
                "strike_price": current_strike_price,
                "slot_expiry_ts": slot_mgr.get().slot_end_ts if slot_mgr.get() else SlotContext.slot_for(time.time()) + SLOT_INTERVAL_S,
            }

            # ------------------------------------------------------------------
            # Strategy → signals
            # ------------------------------------------------------------------
            strategy.set_tokens(current_market_id, yes_token_id, no_token_id)
            intra_cycle_analyze_fn = _make_intra_cycle_fn(
                client, strategy, risk_manager, state, state_store,
                current_market_id, yes_token_id, no_token_id,
                current_question, current_strike_price,
                paper_trading, logger,
                execution_tracker=execution_tracker, inventories=inventories,
                chainlink_feed=chainlink_feed,
                slot_mgr=slot_mgr,
            )
            signals = strategy.analyze(market_data)

            # Guard: suppress BUY entries if insufficient time remains in market window
            _slot_expiry = slot_mgr.get().slot_end_ts if slot_mgr.get() else SlotContext.slot_for(time.time()) + SLOT_INTERVAL_S
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
                book_summary=_book_summary(yes_book, no_book),
            )

            # ------------------------------------------------------------------
            state.cycle_count += 1
            _snapshot_strategy_state(strategy, state)
            _snapshot_chainlink_state(state, chainlink_feed, slot_mgr=slot_mgr)
            state_store.save(state)

            # Write cycle snapshot (single source of truth for dashboard)
            _btc_now = None
            if btc_feed is not None:
                _bk = btc_feed.get_latest_book()
                if _bk is not None:
                    _btc_now = _bk.mid
            _build_and_save_snapshot(
                snapshot_store, state,
                current_market_id, current_question,
                yes_token_id, no_token_id,
                current_strike_price, _btc_now,
                yes_book, no_book,
                inventories, execution_tracker,
                risk_manager, paper_trading,
            )

            elapsed = time.monotonic() - cycle_start
            logger.info(
                f"Cycle {state.cycle_count} done | market={current_market_id} "
                f"| balance={balance:.2f} | daily_pnl={state.daily_realized_pnl:.4f} "
                f"| signals={len(signals)} | elapsed={elapsed:.1f}s"
            )

        except KeyboardInterrupt:
            graceful_shutdown(
                state, state_store, client, paper_trading, logger,
                btc_feed=btc_feed, chainlink_feed=chainlink_feed,
                current_market_id=current_market_id,
                token_outcome_map={yes_token_id: "YES", no_token_id: "NO"} if yes_token_id else None,
                perf_store=perf_store, strategy_name=strategy_name,
            )

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            try:
                _snapshot_chainlink_state(state, chainlink_feed, slot_mgr=slot_mgr)
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
                logger=logger, btc_feed=btc_feed, slot_mgr=slot_mgr,
            )
        else:
            sleep_until_next_cycle(interval)


if __name__ == "__main__":
    main()
