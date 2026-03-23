#!/usr/bin/env python3
"""
dashboard.py — Live BTC/USDT price feed monitor (debug tool).

Reads from BtcPriceFeed and renders a terminal dashboard using Rich.
No strategy logic — monitoring and debugging only.

Requirements:
    pip install rich websockets

Usage:
    python dashboard.py
    python dashboard.py --symbol ethusdt
    python dashboard.py --window 120 --refresh 2
"""

import argparse
import email.utils
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.utils.btc_feed import BtcPriceFeed
from src.utils.chainlink_feed import ChainlinkFeed


# ── Caches ────────────────────────────────────────────────────────────────────

_pos_cache: Optional[List[dict]] = None
_pos_cache_ts: float = 0.0
_POS_CACHE_TTL = 30.0  # seconds

_market_cache: Optional[Dict[str, Any]] = None
_market_cache_ts: float = 0.0
_market_cache_slot: int = 0
_MARKET_CACHE_TTL = 60.0  # re-discover every 60s

_book_cache: Optional[Dict[str, Any]] = None
_book_cache_ts: float = 0.0
_BOOK_CACHE_TTL = 3.0  # refresh book every 3s
_book_fetch_ok_ts: float = 0.0  # last SUCCESSFUL fetch timestamp

# Server-clock offset: updated from HTTP Date: headers on each market fetch.
# Compensates for WSL clock drift vs. Polymarket server time.
_clock_offset: float = 0.0


def _server_now() -> float:
    """Return local time adjusted by the last-observed Polymarket server clock offset."""
    return time.time() + _clock_offset


def _current_slot_ts(now: Optional[float] = None) -> int:
    """Return the active 5-minute slot timestamp."""
    if now is None:
        now = _server_now()
    return int(math.floor(now / 300) * 300)


# ── Chart defaults ────────────────────────────────────────────────────────────

_CHART_W = 60
_CHART_H = 12


# ── ASCII line chart renderer ─────────────────────────────────────────────────

def _render_chart(
    prices: List[Tuple[float, float]],
    width: int,
    height: int,
    window_s: int,
) -> str:
    """
    Render a plain-text line chart from (ts, mid) price pairs.

    Returns a multi-line string with a labelled y-axis on the left
    and an x-axis time label at the bottom.
    """
    if len(prices) < 2:
        blank = "\n" * (height // 2)
        return blank + "       (waiting for price data…)"

    mids = [p[1] for p in prices]
    mn, mx = min(mids), max(mids)

    # Sample prices evenly to fit chart width.
    # When fewer ticks than columns, left-pad with None so data is flush-right.
    n = len(mids)
    if n >= width:
        step = n / width
        sampled: List[Optional[float]] = [
            mids[min(int(i * step), n - 1)] for i in range(width)
        ]
    else:
        sampled = [None] * (width - n) + mids  # type: ignore[list-item]

    # Build a character grid (rows × columns)
    grid = [[" "] * width for _ in range(height)]

    prev_row: Optional[int] = None
    for col, price in enumerate(sampled):
        if price is None:
            prev_row = None
            continue

        if mx == mn:
            row = height // 2
        else:
            row = height - 1 - int((price - mn) / (mx - mn) * (height - 1))
        row = max(0, min(height - 1, row))

        if prev_row is None:
            # First data point
            grid[row][col] = "─"
        elif prev_row == row:
            # Flat — same row
            grid[row][col] = "─"
        elif prev_row > row:
            # Price rising (row number decreasing)
            grid[prev_row][col] = "╯"
            grid[row][col] = "╭"
            for r in range(row + 1, prev_row):
                grid[r][col] = "│"
        else:
            # Price falling (row number increasing)
            grid[prev_row][col] = "╮"
            grid[row][col] = "╰"
            for r in range(prev_row + 1, row):
                grid[r][col] = "│"

        prev_row = row

    # Compose lines with right-aligned y-axis labels
    Y_W = 10  # label column width
    mid_val = (mn + mx) / 2.0
    lines: List[str] = []

    for r, row_chars in enumerate(grid):
        if r == 0:
            label = f"{mx:>{Y_W},.2f} ┤"
        elif r == height - 1:
            label = f"{mn:>{Y_W},.2f} ┤"
        elif r == height // 2:
            label = f"{mid_val:>{Y_W},.2f} ┤"
        else:
            label = " " * (Y_W + 2)
        lines.append(label + "".join(row_chars))

    # X-axis
    pad = " " * (Y_W + 2)
    lines.append(pad + "└" + "─" * width)
    left_lbl = f"T-{window_s}s"
    right_lbl = "now"
    gap = max(0, width - len(left_lbl) - len(right_lbl))
    lines.append(pad + " " + left_lbl + " " * gap + right_lbl)

    return "\n".join(lines)


# ── Polymarket order book helpers ─────────────────────────────────────────────

def _discover_market() -> Optional[Dict[str, Any]]:
    """Find the current BTC Up/Down 5-min market via gamma API. Cached 60s (or until slot rolls)."""
    global _market_cache, _market_cache_ts, _market_cache_slot, _clock_offset

    now = _server_now()
    current_slot = int(math.floor(now / 300) * 300)
    slot_changed = (current_slot != _market_cache_slot)

    if _market_cache is not None and not slot_changed and (now - _market_cache_ts) < _MARKET_CACHE_TTL:
        return _market_cache

    slot = current_slot
    slug = f"btc-updown-5m-{slot}"
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug},
            timeout=5,
        )
        resp.raise_for_status()

        # Update server-clock offset from HTTP Date: header
        date_hdr = resp.headers.get("Date")
        if date_hdr:
            try:
                server_ts = email.utils.parsedate_to_datetime(date_hdr).timestamp()
                _clock_offset = server_ts - time.time()
            except Exception:
                pass
        events = resp.json()
        for event in (events if isinstance(events, list) else [events]):
            for m in event.get("markets", []):
                tids = json.loads(m.get("clobTokenIds", "[]"))
                outcomes = json.loads(m.get("outcomes", "[]"))
                if len(tids) >= 2:
                    volume = float(m.get("volume", 0) or 0)
                    end_ts = None
                    raw_end = m.get("endDate") or m.get("endDateIso")
                    if raw_end:
                        try:
                            end_ts = datetime.fromisoformat(raw_end.replace("Z", "+00:00")).timestamp()
                        except Exception:
                            pass
                    _market_cache = {
                        "up_token": tids[0],
                        "down_token": tids[1],
                        "outcomes": outcomes,
                        "volume": volume,
                        "title": m.get("question", ""),
                        "end_ts": end_ts,
                    }
                    _market_cache_ts = now
                    _market_cache_slot = current_slot
                    return _market_cache
    except Exception:
        pass
    return _market_cache  # return stale on error


def _fetch_clob_book(token_id: str) -> Optional[Dict[str, Any]]:
    """Fetch order book from CLOB for a token. Cached 3s."""
    global _book_cache, _book_cache_ts, _book_fetch_ok_ts

    now = time.time()
    if (_book_cache is not None
            and (now - _book_cache_ts) < _BOOK_CACHE_TTL
            and _book_cache.get("_token_id") == token_id):
        return _book_cache

    try:
        resp = requests.get(
            "https://clob.polymarket.com/book",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        data["_token_id"] = token_id

        # Compute midpoint from book data; only fall back to /midpoint if both sides empty
        raw_bids = data.get("bids") or []
        raw_asks = data.get("asks") or []
        new_mid = None
        if raw_bids and raw_asks:
            new_mid = (float(raw_bids[0].get("price") or raw_bids[0].get("p"))
                       + float(raw_asks[0].get("price") or raw_asks[0].get("p"))) / 2
        elif raw_bids:
            new_mid = float(raw_bids[0].get("price") or raw_bids[0].get("p"))
        elif raw_asks:
            new_mid = float(raw_asks[0].get("price") or raw_asks[0].get("p"))
        else:
            # Book completely empty — fetch /midpoint as last resort
            try:
                mr = requests.get(
                    "https://clob.polymarket.com/midpoint",
                    params={"token_id": token_id},
                    timeout=3,
                )
                mr.raise_for_status()
                new_mid = float(mr.json().get("mid", 0)) or None
            except Exception:
                new_mid = None

        # Track previous midpoint for direction arrow
        old_mid = _book_cache.get("_midpoint") if _book_cache else None
        data["_prev_midpoint"] = old_mid
        data["_midpoint"] = new_mid

        _book_cache = data
        _book_cache_ts = now
        _book_fetch_ok_ts = now
        return _book_cache
    except Exception:
        return _book_cache  # stale on error


_OB_LEVELS = 5  # number of price levels to show per side


def _build_order_book_panel(market: Optional[Dict[str, Any]] = None) -> Panel:
    """Build a Polymarket-style order book panel (asks on top, bids below)."""
    if market is None:
        market = _discover_market()
    if market is None:
        return Panel(
            Text("No market found", style="dim italic"),
            title="[dim]Order Book[/dim]",
            border_style="dim",
        )

    # Fetch book for the "Up" token (Trade Up view)
    book_data = _fetch_clob_book(market["up_token"])
    if book_data is None:
        return Panel(
            Text("Book unavailable", style="dim italic"),
            title="[dim]Order Book[/dim]",
            border_style="dim",
        )

    raw_bids = book_data.get("bids") or []
    raw_asks = book_data.get("asks") or []
    midpoint = book_data.get("_midpoint")
    volume = market.get("volume", 0)

    # Parse and sort
    bids = sorted(
        [{"price": float(b.get("price") or b.get("p")),
          "size": float(b.get("size") or b.get("s"))}
         for b in raw_bids],
        key=lambda x: x["price"], reverse=True,
    )
    asks = sorted(
        [{"price": float(a.get("price") or a.get("p")),
          "size": float(a.get("size") or a.get("s"))}
         for a in raw_asks],
        key=lambda x: x["price"],
    )

    # Take top N levels
    top_bids = bids[:_OB_LEVELS]
    top_asks = asks[:_OB_LEVELS]

    # Compute cumulative totals
    def _cumulate(levels):
        cumul = []
        running = 0.0
        for lvl in levels:
            running += lvl["price"] * lvl["size"]
            cumul.append({**lvl, "total": running})
        return cumul

    top_asks_cum = _cumulate(top_asks)
    top_bids_cum = _cumulate(top_bids)

    # Max total for depth bar scaling
    all_totals = [l["total"] for l in top_asks_cum + top_bids_cum]
    max_total = max(all_totals) if all_totals else 1.0
    bar_width = 6

    def _depth_bar(total: float, char: str) -> str:
        fill = int((total / max_total) * bar_width) if max_total > 0 else 0
        return char * fill + " " * (bar_width - fill)

    # Build the table
    tbl = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    tbl.add_column("", width=bar_width, no_wrap=True)  # depth bar
    tbl.add_column("PRICE", justify="center", no_wrap=True, min_width=7)
    tbl.add_column("SHARES", justify="right", no_wrap=True, min_width=9)
    tbl.add_column("TOTAL", justify="right", no_wrap=True, min_width=10)

    # Asks — show highest first (reversed), so lowest ask is nearest spread
    for lvl in reversed(top_asks_cum):
        cents = int(round(lvl["price"] * 100))
        tbl.add_row(
            Text(_depth_bar(lvl["total"], "\u2588"), style="red"),
            Text(f"{cents}\u00a2", style="bold red"),
            Text(f"{lvl['size']:,.2f}", style="white"),
            Text(f"${lvl['total']:,.2f}", style="dim"),
        )

    # Spread line
    if top_bids and top_asks:
        best_bid = top_bids[0]["price"]
        best_ask = top_asks[0]["price"]
        spread = best_ask - best_bid
        spread_str = f"{spread * 100:.2f}"
        # Direction arrow comparing to previous fetch
        prev_mid = book_data.get("_prev_midpoint")
        if midpoint and prev_mid:
            if midpoint > prev_mid + 1e-6:
                arrow = "\u2191"  # ↑
            elif midpoint < prev_mid - 1e-6:
                arrow = "\u2193"  # ↓
            else:
                arrow = "="
        else:
            arrow = ""
        last_str = f"{midpoint * 100:.2f}\u00a2{arrow}" if midpoint else "–"
        # Book age: seconds since last successful API fetch
        age = time.time() - _book_fetch_ok_ts if _book_fetch_ok_ts > 0 else 0
        age_str = f"{age:.0f}s ago" if age > 0 else ""
        tbl.add_row(
            Text("", style="dim"),
            Text(f"Last: {last_str}", style="bold yellow"),
            Text(age_str, style="dim italic"),
            Text(f"Spread: {spread_str}\u00a2", style="dim"),
        )

    # Bids — highest first
    for lvl in top_bids_cum:
        cents = int(round(lvl["price"] * 100))
        tbl.add_row(
            Text(_depth_bar(lvl["total"], "\u2588"), style="green"),
            Text(f"{cents}\u00a2", style="bold green"),
            Text(f"{lvl['size']:,.2f}", style="white"),
            Text(f"${lvl['total']:,.2f}", style="dim"),
        )

    vol_str = f"${volume:,.0f} Vol." if volume else ""
    return Panel(
        tbl,
        title=f"[dim]Order Book — Trade Up[/dim]",
        subtitle=f"[dim]{vol_str}[/dim]" if vol_str else None,
        border_style="dim",
    )


# ── CLOB midpoint cache (for unrealized PnL) ─────────────────────────────────

_mid_cache: Dict[str, Tuple[float, float]] = {}  # token_id -> (mid, fetch_ts)
_MID_CACHE_TTL = 5.0


def _cached_midpoint(token_id: str) -> Optional[float]:
    """Fetch CLOB midpoint for a token, cached for 5s."""
    now = time.time()
    cached = _mid_cache.get(token_id)
    if cached is not None and (now - cached[1]) < _MID_CACHE_TTL:
        return cached[0]
    try:
        r = requests.get(
            "https://clob.polymarket.com/midpoint",
            params={"token_id": token_id},
            timeout=3,
        )
        r.raise_for_status()
        mid = float(r.json().get("mid", 0)) or None
        if mid is not None:
            _mid_cache[token_id] = (mid, now)
        return mid
    except Exception:
        return cached[0] if cached else None


def _compute_total_unrealized(bot_state: dict) -> Optional[float]:
    """Compute total unrealized PnL from active inventories + CLOB midpoints."""
    inventories = bot_state.get("inventories", {})
    active = {
        tid: inv for tid, inv in inventories.items()
        if inv.get("position", 0) != 0
    }
    if not active:
        return None

    total = 0.0
    for tid, inv in active.items():
        pos = inv.get("position", 0)
        avg_cost = inv.get("avg_cost", 0)
        mid = _cached_midpoint(tid)
        if mid is None:
            return None  # can't compute reliably — bail
        total += (mid - avg_cost) * pos
    return total


# ── Bot state & positions helpers ─────────────────────────────────────────────

_state_cache: Optional[Tuple[Dict[str, Any], float]] = None
_state_cache_mtime: float = 0.0


def _load_bot_state(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Read and parse bot_state.json. Cached; re-reads only when mtime changes."""
    global _state_cache, _state_cache_mtime
    try:
        mtime = os.path.getmtime(path)
        if _state_cache is not None and mtime == _state_cache_mtime:
            return _state_cache
        with open(path, "r") as f:
            data = json.load(f)
        _state_cache = (data, mtime)
        _state_cache_mtime = mtime
        return data, mtime
    except (OSError, json.JSONDecodeError):
        return None, None


def _fetch_live_positions() -> Optional[List[dict]]:
    """Fetch positions from Polymarket data API. Cached for 30s."""
    global _pos_cache, _pos_cache_ts

    funder = os.environ.get("PROXY_FUNDER")
    if not funder:
        return None

    now = time.time()
    if _pos_cache is not None and (now - _pos_cache_ts) < _POS_CACHE_TTL:
        return _pos_cache

    try:
        resp = requests.get(
            "https://data-api.polymarket.com/positions",
            params={"user": funder, "sizeThreshold": "0"},
            timeout=5,
        )
        resp.raise_for_status()
        _pos_cache = resp.json()
        _pos_cache_ts = now
        return _pos_cache
    except Exception:
        return _pos_cache  # return stale cache on error, or None


def _build_positions_panel(
    bot_state: Optional[dict],
    live_positions: Optional[List[dict]],
    market: Optional[Dict[str, Any]] = None,
) -> Panel:
    """Build the Positions panel from bot state + optional live enrichment."""
    if bot_state is None:
        return Panel(
            Text("No state data", style="dim italic"),
            title="[dim]Positions[/dim]",
            border_style="dim",
        )

    inventories = bot_state.get("inventories", {})
    # Filter to non-zero positions
    active = {
        tid: inv for tid, inv in inventories.items()
        if inv.get("position", 0) != 0
    }

    if not active:
        return Panel(
            Text("FLAT — no open positions", style="dim italic"),
            title="[dim]Positions[/dim]",
            border_style="dim",
        )

    # Build lookup from live positions by asset (token_id)
    live_lookup: Dict[str, dict] = {}
    if live_positions:
        for pos in live_positions:
            asset = pos.get("asset") or pos.get("token_id") or ""
            if asset:
                live_lookup[asset] = pos

    # Build token_id → side label map from current market cache.
    # Polymarket binary markets: tids[0] = YES, tids[1] = NO — always.
    if market is None:
        market = _discover_market()
    token_to_outcome: dict = {}
    if market:
        if market.get("up_token"):
            token_to_outcome[market["up_token"]] = "YES"
        if market.get("down_token"):
            token_to_outcome[market["down_token"]] = "NO"

    tbl = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    tbl.add_column("Side", style="bold", no_wrap=True)
    tbl.add_column("Size", justify="right", no_wrap=True)
    tbl.add_column("AvgCost", justify="right", no_wrap=True)
    tbl.add_column("uPnL", justify="right", no_wrap=True)

    for tid, inv in active.items():
        pos_size = inv.get("position", 0)
        avg_cost = inv.get("avg_cost", 0)

        # Determine label: market outcome > live API outcome > truncated ID
        live = live_lookup.get(tid)
        if tid in token_to_outcome:
            label = token_to_outcome[tid]
        elif live and live.get("outcome"):
            label = live["outcome"]
        else:
            label = tid[:8] + "…" if len(tid) > 8 else tid

        # Unrealized PnL: live API price > CLOB midpoint > fallback "–"
        cur: Optional[float] = None
        if live and "cur_price" in live:
            cur = float(live["cur_price"])
        if cur is None:
            cur = _cached_midpoint(tid)
        if cur is not None and avg_cost > 0:
            upnl = (cur - avg_cost) * pos_size
            upnl_style = "green" if upnl >= 0 else "red"
            upnl_str = f"{upnl:+.4f}"
        else:
            upnl_str = "–"
            upnl_style = "dim"

        size_str = f"{abs(pos_size):.1f}"
        cost_str = f"${avg_cost:.2f}"

        tbl.add_row(label, size_str, cost_str, Text(upnl_str, style=upnl_style))

    return Panel(tbl, title="[dim]Positions[/dim]", border_style="dim")


def _build_pnl_panel(
    bot_state: Optional[dict],
    total_upnl: Optional[float] = None,
) -> Panel:
    """Build the PnL panel from bot state."""
    if bot_state is None:
        return Panel(
            Text("No state data", style="dim italic"),
            title="[dim]PnL[/dim]",
            border_style="dim",
        )

    daily_pnl = bot_state.get("daily_realized_pnl", 0.0)
    reset_date = bot_state.get("daily_reset_date", "–")
    session_wins = bot_state.get("session_wins", 0)
    session_losses = bot_state.get("session_losses", 0)

    rpnl_style = "green" if daily_pnl >= 0 else "red"

    total_trades = session_wins + session_losses
    if total_trades > 0:
        win_rate = session_wins / total_trades
        wr_str = f"{win_rate*100:.0f}%  ({session_wins}W/{session_losses}L)"
        wr_style = "bold green" if win_rate >= 0.5 else "bold yellow"
    else:
        wr_str = "—"
        wr_style = "dim"

    slot_pnl = bot_state.get("slot_realized_pnl", 0.0)
    slot_style = "green" if slot_pnl >= 0 else "red"

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=8)
    tbl.add_column(justify="right")
    tbl.add_row("Slot PnL", Text(f"{slot_pnl:+.4f}", style=f"bold {slot_style}"))
    tbl.add_row("Daily", Text(f"{daily_pnl:+.4f}", style=f"bold {rpnl_style}"))

    if total_upnl is not None:
        upnl_style = "green" if total_upnl >= 0 else "red"
        tbl.add_row("Unreal.", Text(f"{total_upnl:+.4f}", style=f"bold {upnl_style}"))
        combined = daily_pnl + total_upnl
        comb_style = "green" if combined >= 0 else "red"
        tbl.add_row("Total", Text(f"{combined:+.4f}", style=f"bold {comb_style}"))

    tbl.add_row("Trades", Text(str(total_trades), style="white"))
    tbl.add_row("Win Rate", Text(wr_str, style=wr_style))
    tbl.add_row("Since", Text(str(reset_date), style="dim"))

    return Panel(tbl, title="[dim]PnL[/dim]", border_style="dim")


def _build_bot_status_panel(
    bot_state: Optional[dict],
    file_mtime: Optional[float],
) -> Panel:
    """Build the Bot Status panel."""
    if bot_state is None:
        return Panel(
            Text("Bot offline", style="dim italic"),
            title="[dim]Bot Status[/dim]",
            border_style="dim",
        )

    cycle = bot_state.get("cycle_count", 0)

    # active_order_ids is a dict {token_id: order_id|null}, not a list
    orders = bot_state.get("active_order_ids", {})
    if isinstance(orders, dict):
        order_count = sum(1 for v in orders.values() if v is not None)
    else:
        order_count = 0

    # Derive operational status from file mtime + strategy state + orders
    strategy_status = bot_state.get("strategy_status", "")
    if file_mtime is None:
        status = Text("UNKNOWN", style="dim")
    else:
        age_min = (time.time() - file_mtime) / 60.0
        if age_min >= 10:
            status = Text(f"STOPPED ({age_min:.0f}m ago)", style="bold red")
        elif strategy_status == "POSITION_OPEN":
            status = Text("POSITION_OPEN", style="bold cyan")
        elif order_count > 0:
            status = Text("ORDER_PENDING", style="bold yellow")
        else:
            status = Text("RUNNING", style="bold green")

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=8)
    tbl.add_column(justify="right")
    tbl.add_row("Cycle", Text(str(cycle), style="white"))
    tbl.add_row("Orders", Text(str(order_count), style="white"))
    tbl.add_row("Bot", status)

    return Panel(tbl, title="[dim]Bot Status[/dim]", border_style="dim")


# ── Market Cycle panel ────────────────────────────────────────────────────────

def _resolve_price_to_beat(
    chainlink_feed: Optional[ChainlinkFeed],
    bot_state: Optional[dict] = None,
    current_slot: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    """Resolve the authoritative current-slot price-to-beat."""
    slot_ts = current_slot if current_slot is not None else _current_slot_ts()

    if chainlink_feed is not None:
        slot_open = chainlink_feed.get_slot_open_price()
        if slot_open is not None and slot_open.slot_ts == slot_ts:
            return slot_open.price, "Chainlink (live)"

    if bot_state:
        bot_price = bot_state.get("chainlink_ref_price")
        bot_slot = bot_state.get("chainlink_ref_slot_ts")
        if bot_price is not None and bot_slot == slot_ts:
            return float(bot_price), "Chainlink (bot snapshot)"

    return None, "Waiting for Chainlink"


def _build_market_cycle_panel(
    feed: BtcPriceFeed,
    chainlink_feed: Optional[ChainlinkFeed],
    bot_state: Optional[dict] = None,
    market: Optional[Dict[str, Any]] = None,
) -> Panel:
    """Show current 5-min slot info: countdown, price to beat, BTC delta."""
    if market is None:
        market = _discover_market()

    now = _server_now()
    current_slot = _current_slot_ts(now)
    end_ts = (market or {}).get("end_ts")
    if isinstance(end_ts, (int, float)) and current_slot < end_ts <= current_slot + 300:
        slot_end = float(end_ts)
    else:
        slot_end = current_slot + 300
    remaining = max(0.0, slot_end - now)

    slot_str = datetime.fromtimestamp(current_slot, tz=timezone.utc).strftime("%H:%M UTC")
    rem_m = int(remaining) // 60
    rem_s = int(remaining) % 60
    rem_str = f"{rem_m:02d}:{rem_s:02d}"

    if remaining <= 30:
        rem_style = "bold red"
    elif remaining <= 90:
        rem_style = "bold yellow"
    else:
        rem_style = "bold green"

    price_to_beat, price_source = _resolve_price_to_beat(
        chainlink_feed,
        bot_state=bot_state,
        current_slot=current_slot,
    )

    # Current BTC price from feed
    book = feed.get_latest_book()
    btc_now = book.mid if book is not None else None

    # Chainlink health prefers the dashboard-owned feed.
    cl_healthy = chainlink_feed.is_healthy() if chainlink_feed is not None else False

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=10)
    tbl.add_column(justify="right")

    tbl.add_row("Slot", Text(slot_str, style="dim"))
    tbl.add_row("Remaining", Text(rem_str, style=rem_style))

    if price_to_beat is not None:
        tbl.add_row("Price to beat", Text(f"${price_to_beat:,.2f}", style="cyan"))
    else:
        tbl.add_row("Price to beat", Text("Waiting for Chainlink", style="dim"))

    source_style = "cyan" if price_to_beat is not None else "dim"
    tbl.add_row("Source", Text(price_source, style=source_style))

    # Chainlink feed health
    if cl_healthy:
        tbl.add_row("Chainlink", Text("LIVE", style="bold green"))
    else:
        tbl.add_row("Chainlink", Text("DOWN", style="bold red"))

    if btc_now is not None:
        tbl.add_row("BTC Now", Text(f"${btc_now:,.2f}", style="bold yellow"))
        if price_to_beat is not None:
            delta = btc_now - price_to_beat
            pct = delta / price_to_beat * 100 if price_to_beat > 0 else 0
            arrow = "▲" if delta >= 0 else "▼"
            delta_style = "bold green" if delta >= 0 else "bold red"
            tbl.add_row("Delta", Text(f"{arrow} {delta:+,.2f} ({pct:+.2f}%)", style=delta_style))
        else:
            tbl.add_row("Delta", Text("–", style="dim"))
    else:
        tbl.add_row("BTC Now", Text("–", style="dim"))
        tbl.add_row("Delta", Text("–", style="dim"))

    return Panel(tbl, title="[dim]Market Cycle[/dim]", border_style="dim")


# ── Strategy panel ────────────────────────────────────────────────────────────

def _build_strategy_panel(bot_state: Optional[dict]) -> Panel:
    """Show strategy internal reasoning from bot_state snapshot fields."""
    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=10)
    tbl.add_column(justify="right")

    if bot_state is None:
        return Panel(
            Text("No strategy data", style="dim italic"),
            title="[dim]Strategy[/dim]",
            border_style="dim",
        )

    name = bot_state.get("strategy_name", "") or "–"
    tbl.add_row("Strategy", Text(name, style="white"))

    status = bot_state.get("strategy_status", "") or "–"
    if status == "POSITION_OPEN":
        status_style = "bold cyan"
    elif status == "WATCHING":
        status_style = "bold green"
    elif status == "EXITED":
        status_style = "bold yellow"
    else:
        status_style = "dim"
    tbl.add_row("Status", Text(status, style=status_style))

    bias = bot_state.get("strategy_bias", "")
    if bias:
        bias_style = "bold green" if bias == "LONG" else ("bold red" if bias == "SHORT" else "dim")
        tbl.add_row("Bias", Text(bias, style=bias_style))

    zscore = bot_state.get("strategy_zscore")
    if zscore is not None:
        z_style = "bold red" if abs(zscore) >= 1.8 else "white"
        tbl.add_row("Z-score", Text(f"{zscore:+.3f}", style=z_style))

    momentum_pct = bot_state.get("strategy_momentum_pct")
    if momentum_pct is not None:
        mom_style = "bold green" if momentum_pct >= 0 else "bold red"
        tbl.add_row("Momentum", Text(f"{momentum_pct * 100:+.2f}%", style=mom_style))

    prob_yes = bot_state.get("strategy_prob_yes")
    if prob_yes is not None:
        prob_style = "bold green" if prob_yes >= 0.5 else "bold red"
        tbl.add_row("Prob YES", Text(f"{prob_yes:.3f}", style=prob_style))

    edge_yes = bot_state.get("strategy_edge_yes")
    if edge_yes is not None:
        edge_style = "bold green" if edge_yes >= 0 else "bold red"
        tbl.add_row("Edge YES", Text(f"{edge_yes:+.3f}", style=edge_style))

    edge_no = bot_state.get("strategy_edge_no")
    if edge_no is not None:
        edge_style = "bold green" if edge_no >= 0 else "bold red"
        tbl.add_row("Edge NO", Text(f"{edge_no:+.3f}", style=edge_style))

    model_version = bot_state.get("strategy_model_version", "")
    if model_version:
        tbl.add_row("Model", Text(model_version[-18:], style="white"))

    feature_status = bot_state.get("strategy_feature_status", "")
    if feature_status:
        status_style = "bold green" if feature_status.startswith("ready") else "bold yellow"
        tbl.add_row("Features", Text(feature_status, style=status_style))

    last_sig = bot_state.get("strategy_last_signal", "")
    last_ts = bot_state.get("strategy_last_signal_ts", 0.0)
    if last_sig:
        age_s = int(time.time() - last_ts) if last_ts else 0
        age_str = f"{age_s}s ago"
        tbl.add_row("Last sig", Text(f"{last_sig[:50]}", style="dim"))
        tbl.add_row("", Text(age_str, style="dim"))
    else:
        tbl.add_row("Last sig", Text("–", style="dim"))

    return Panel(tbl, title="[dim]Strategy[/dim]", border_style="dim")


def _build_trade_log_panel(bot_state: Optional[dict]) -> Panel:
    """Show recent BUY/SELL trades from bot_state.trade_log."""
    trades = (bot_state or {}).get("trade_log", [])
    if not trades:
        return Panel(
            Text("No trades yet", style="dim italic"),
            title="[dim]Trades[/dim]",
            border_style="dim",
        )

    tbl = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    tbl.add_column("", width=4, no_wrap=True)       # BUY/SELL
    tbl.add_column("", width=3, no_wrap=True)       # YES/NO
    tbl.add_column("Price", justify="right", no_wrap=True, min_width=6)
    tbl.add_column("Size", justify="right", no_wrap=True, min_width=6)
    tbl.add_column("Age", justify="right", no_wrap=True, min_width=6)

    now = time.time()
    for entry in reversed(trades[-10:]):
        action  = entry.get("action", "")
        outcome = entry.get("outcome", "")
        price   = entry.get("price", 0.0)
        size    = entry.get("size", 0.0)
        age_s   = int(now - entry.get("ts", now))
        age_str = f"{age_s // 60}m{age_s % 60}s" if age_s >= 60 else f"{age_s}s"

        act_style = "bold green" if action == "BUY" else "bold red"
        tbl.add_row(
            Text(action, style=act_style),
            Text(outcome, style="white"),
            Text(f"{price:.2f}", style="white"),
            Text(f"{size:.1f}", style="dim"),
            Text(age_str, style="dim"),
        )

    return Panel(tbl, title="[dim]Trades[/dim]", border_style="dim")


_log_cache: Optional[list] = None
_log_cache_mtime: float = 0.0


def _build_log_panel(log_file: str, n: int = 18) -> Panel:
    """Show last N lines of the bot log file, color-coded by level. Cached by mtime."""
    global _log_cache, _log_cache_mtime
    lines: list = []
    try:
        mtime = os.path.getmtime(log_file)
        if _log_cache is not None and mtime == _log_cache_mtime:
            lines = _log_cache
        else:
            with open(log_file, "rb") as f:
                # Seek near end to avoid reading entire file
                try:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 8192))
                except OSError:
                    f.seek(0)
                tail = f.read().decode("utf-8", errors="replace")
            lines = tail.splitlines(keepends=True)[-n:]
            _log_cache = lines
            _log_cache_mtime = mtime
    except (FileNotFoundError, OSError):
        pass

    text = Text()
    for line in lines:
        line = line.rstrip()
        if " - ERROR - " in line or " - CRITICAL - " in line:
            style = "bold red"
        elif " - WARNING - " in line:
            style = "bold yellow"
        elif " - INFO - " in line:
            style = "cyan"
        else:
            style = "dim"
        text.append(line + "\n", style=style)

    if not lines:
        text = Text("No log entries yet", style="dim italic")

    return Panel(text, title="[bold]Logs[/bold]", border_style="dim", padding=(0, 1))


# ── Dashboard builder ─────────────────────────────────────────────────────────

def _build_panel(
    feed: BtcPriceFeed,
    chainlink_feed: Optional[ChainlinkFeed],
    window_s: int,
    chart_w: int,
    chart_h: int,
    start_time: float,
    state_file: str = "bot_state.json",
    log_file: str = "logs/btc_updown_bot.log",
) -> Panel:
    """Build the complete dashboard as a single Rich Panel."""
    book = feed.get_latest_book()
    age_ms = feed.get_feed_age_ms()
    healthy = feed.is_healthy()
    prices = feed.get_recent_prices(window_s)

    # ── Colours ───────────────────────────────────────────────────────────
    if healthy:
        status_text = Text("● HEALTHY", style="bold green")
        border_style = "green"
        age_style = "cyan"
    else:
        status_text = Text("● STALE", style="bold red")
        border_style = "bright_red"
        age_style = "bold red"

    age_str = f"{age_ms:.0f} ms" if age_ms is not None else "–"
    sn = _server_now()
    slot_elapsed = int(sn - int(math.floor(sn / 300) * 300))
    uptime_str = f"{slot_elapsed // 60}m {slot_elapsed % 60}s"

    # ── Status bar ────────────────────────────────────────────────────────
    reconnects = getattr(feed, "reconnect_count", "–")
    status_bar = Table.grid(padding=(0, 3))
    status_bar.add_column()
    status_bar.add_column()
    status_bar.add_column()
    status_bar.add_column()
    status_bar.add_row(
        status_text,
        Text(f"Age: {age_str}", style=age_style),
        Text(f"Reconnects: {reconnects}", style="dim"),
        Text(f"Slot: {uptime_str}", style="dim"),
    )

    # ── Stats table (left) ────────────────────────────────────────────────
    stats = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    stats.add_column(style="dim", width=9)
    stats.add_column(justify="right", min_width=14, no_wrap=True)

    if book is not None:
        spread = book.ask - book.bid
        stats.add_row("Bid",      Text(f"{book.bid:>14,.2f}", style="white"))
        stats.add_row("Ask",      Text(f"{book.ask:>14,.2f}", style="white"))
        stats.add_row("Spread",   Text(f"{spread:>14,.4f}",   style="dim"))
        stats.add_row("",         "")
        stats.add_row("Mid",      Text(f"{book.mid:>14,.2f}", style="bold yellow"))
        stats.add_row("",         "")
        stats.add_row("Age",      Text(f"{age_str:>14}",      style=age_style))
        stats.add_row("",         "")
        tpm = len(prices) / window_s * 60 if window_s > 0 else 0
        stats.add_row("Ticks/min", Text(f"{tpm:>12,.0f}", style="dim"))
        stats.add_row("Buffer",    Text(f"{len(prices):>12,} pts", style="dim"))

        # Price range over window
        if len(prices) >= 2:
            mids = [p[1] for p in prices]
            lo, hi = min(mids), max(mids)
            stats.add_row("",       "")
            stats.add_row("Hi",     Text(f"{hi:>14,.2f}", style="dim"))
            stats.add_row("Lo",     Text(f"{lo:>14,.2f}", style="dim"))
            stats.add_row("Range",  Text(f"{hi - lo:>14,.2f}", style="dim"))
    else:
        for _ in range(4):
            stats.add_row("", "")
        stats.add_row("", Text("Connecting…", style="dim italic"))

    stats_panel = Panel(stats, title="[dim]Book[/dim]", width=32, border_style="dim")

    # ── Chart (right) ─────────────────────────────────────────────────────
    chart_str = _render_chart(prices, chart_w, chart_h, window_s)
    chart_panel = Panel(
        chart_str,
        title=f"[dim]Mid-price  (last {window_s}s)[/dim]",
        border_style="dim",
    )

    # ── Two-column body ───────────────────────────────────────────────────
    body = Table.grid(expand=True)
    body.add_column(no_wrap=True)
    body.add_column(ratio=1)
    body.add_row(stats_panel, chart_panel)

    # ── Discover market once per render ──────────────────────────────────
    market = _discover_market()

    # ── Polymarket order book (middle row) ────────────────────────────────
    order_book_panel = _build_order_book_panel(market=market)

    # ── Bot state panels ──────────────────────────────────────────────────
    bot_state, file_mtime = _load_bot_state(state_file)
    live_positions = _fetch_live_positions()

    positions_panel = _build_positions_panel(bot_state, live_positions, market=market)
    total_upnl = _compute_total_unrealized(bot_state) if bot_state else None
    pnl_panel = _build_pnl_panel(bot_state, total_upnl=total_upnl)
    bot_status_panel = _build_bot_status_panel(bot_state, file_mtime)

    # Right column: stack positions + pnl + bot status vertically
    right_stack = Table.grid(expand=True)
    right_stack.add_column(ratio=1)
    right_stack.add_column(ratio=1)
    right_stack.add_column(ratio=1)
    right_stack.add_row(positions_panel, pnl_panel, bot_status_panel)

    # Middle row: order book (left) + bot panels (right)
    middle_row = Table.grid(expand=True)
    middle_row.add_column(no_wrap=True)
    middle_row.add_column(ratio=1)
    middle_row.add_row(order_book_panel, right_stack)

    # ── Market Cycle + Strategy + Trade Log panels (bottom row) ──────────
    market_cycle_panel = _build_market_cycle_panel(
        feed,
        chainlink_feed,
        bot_state=bot_state,
        market=market,
    )
    strategy_panel = _build_strategy_panel(bot_state)
    trade_log_panel = _build_trade_log_panel(bot_state)

    bottom_row = Table.grid(expand=True)
    bottom_row.add_column(ratio=1)
    bottom_row.add_column(ratio=1)
    bottom_row.add_column(ratio=1)
    bottom_row.add_row(market_cycle_panel, strategy_panel, trade_log_panel)

    # ── Footer ────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    footer = Text(
        f"  Updated: {now_str}   Ctrl-C to exit",
        style="dim",
    )

    # ── Logs panel ────────────────────────────────────────────────────────
    log_panel = _build_log_panel(log_file)

    # ── Assemble ──────────────────────────────────────────────────────────
    outer = Table.grid(expand=True)
    outer.add_column()
    outer.add_row(status_bar)
    outer.add_row(body)
    outer.add_row(middle_row)
    outer.add_row(bottom_row)
    outer.add_row(log_panel)
    outer.add_row(footer)

    symbol = getattr(feed, "_symbol", "btcusdt").upper()
    return Panel(
        outer,
        title=f"[bold]BTC Feed Monitor[/bold]  [dim]{symbol}[/dim]",
        border_style=border_style,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live BTC/USDT price feed monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol",  default="btcusdt",      help="Binance symbol")
    parser.add_argument("--refresh", type=float, default=4,  help="Refresh rate in Hz")
    parser.add_argument("--window",  type=int, default=300,  help="Chart window (seconds)")
    parser.add_argument("--chart-w", type=int, default=_CHART_W, dest="chart_w",
                        help="Chart width in characters")
    parser.add_argument("--chart-h", type=int, default=_CHART_H, dest="chart_h",
                        help="Chart height in rows")
    parser.add_argument("--state-file", default="bot_state.json", dest="state_file",
                        help="Path to bot_state.json")
    parser.add_argument("--log-file", default="logs/btc_updown_bot.log", dest="log_file",
                        help="Path to bot log file")
    args = parser.parse_args()

    # Suppress feed internal logs so they don't interfere with the TUI
    logging.getLogger("btc_feed").setLevel(logging.WARNING)
    logging.getLogger("chainlink_feed").setLevel(logging.WARNING)

    console = Console()
    console.print(f"\n[bold]BTC Feed Monitor[/bold] — connecting to [cyan]{args.symbol}[/cyan]…")
    console.print("[dim]Starting WebSocket feed…[/dim]\n")

    feed = BtcPriceFeed(symbol=args.symbol)
    feed.start()
    chainlink_feed = ChainlinkFeed(symbol="btc/usd")
    chainlink_feed.start()
    start_time = time.time()

    interval = 1.0 / max(args.refresh, 0.1)

    try:
        with Live(
            console=console,
            screen=True,
            refresh_per_second=args.refresh,
        ) as live:
            while True:
                t0 = time.monotonic()
                panel = _build_panel(
                    feed,
                    chainlink_feed,
                    args.window,
                    args.chart_w,
                    args.chart_h,
                    start_time,
                    args.state_file,
                    args.log_file,
                )
                live.update(panel)
                elapsed = time.monotonic() - t0
                time.sleep(max(0, interval - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        chainlink_feed.stop()
        console.print("\n[dim]Feed stopped.[/dim]")


if __name__ == "__main__":
    main()
