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
import glob as _glob
import json
import logging
import math
import os
import sys
import threading
import time
import traceback
import tty
import termios
from datetime import datetime, timezone

import yaml
from typing import Any, Dict, List, Optional, Tuple

import requests
from src.utils.crypto_feed import CryptoPriceFeed
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.engine.slot_state import SLOT_INTERVAL_S, SlotContext
from src.utils.btc_feed import BtcPriceFeed
from src.utils.chainlink_feed import ChainlinkFeed


# ── Caches ────────────────────────────────────────────────────────────────────

_pos_cache: Optional[List[dict]] = None
_pos_cache_ts: float = 0.0
_POS_CACHE_TTL = 30.0  # seconds

_market_cache: Optional[Dict[str, Any]] = None
_market_cache_ts: float = 0.0
_market_cache_slot: int = 0
_MARKET_CACHE_TTL = 15.0  # re-discover every 15s for fresh outcomePrices

_book_cache: Optional[Dict[str, Any]] = None
_book_cache_ts: float = 0.0
_BOOK_CACHE_TTL = 3.0  # refresh book every 3s
_book_fetch_ok_ts: float = 0.0  # last SUCCESSFUL fetch timestamp

_slot_price_rest_cache: Dict[int, Tuple[float, str]] = {}   # slot_ts → (price, source)
_slot_price_rest_failed: Dict[int, float] = {}             # slot_ts → last failure time

# Resolved market outcome cache per slot
_slot_outcome_cache: Dict[int, str] = {}    # slot_ts → "Up" | "Down"
_slot_outcome_retry_ts: Dict[int, float] = {}
_slot_outcome_lock = threading.Lock()
_SLOT_OUTCOME_RETRY_S = 60.0  # retry unresolved slots every 60s

# Server-clock offset: updated from HTTP Date: headers on each market fetch.
# Compensates for WSL clock drift vs. Polymarket server time.
_clock_offset: float = 0.0

# Multi-asset: slug prefix for market discovery (set at startup from --asset flag)
_slug_prefix: str = "btc-updown-5m"


def _server_now() -> float:
    """Return local time adjusted by the last-observed Polymarket server clock offset."""
    return time.time() + _clock_offset


def _current_slot_ts(now: Optional[float] = None) -> int:
    """Return the active 5-minute slot timestamp."""
    if now is None:
        now = _server_now()
    return SlotContext.slot_for(now)


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
    slug = f"{_slug_prefix}-{slot}"
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
                    # outcomePrices[0] = YES/Up price; updated each time market is re-fetched
                    up_price: Optional[float] = None
                    try:
                        raw_prices = json.loads(m.get("outcomePrices", "[]"))
                        if raw_prices:
                            up_price = float(raw_prices[0])
                    except Exception:
                        pass
                    _market_cache = {
                        "up_token": tids[0],
                        "down_token": tids[1],
                        "outcomes": outcomes,
                        "volume": volume,
                        "title": m.get("question", ""),
                        "end_ts": end_ts,
                        "up_price": up_price,
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

        # Prefer gamma-api outcomePrices (updated every 5s) over CLOB midpoint
        # (which falls back to a fixed 0.50 when the order book is empty).
        gamma_price = market.get("up_price") if market else None
        display_price = gamma_price if gamma_price is not None else midpoint

        prev_mid = book_data.get("_prev_midpoint")
        if display_price and prev_mid:
            if display_price > prev_mid + 1e-6:
                arrow = "\u2191"  # ↑
            elif display_price < prev_mid - 1e-6:
                arrow = "\u2193"  # ↓
            else:
                arrow = "="
        else:
            arrow = ""
        last_str = f"{display_price * 100:.2f}\u00a2{arrow}" if display_price else "–"

        # Book age: seconds since last successful CLOB fetch
        age = time.time() - _book_fetch_ok_ts if _book_fetch_ok_ts > 0 else 0
        if age > 15:
            age_style = "bold red"
            age_str = f"STALE {age:.0f}s"
        elif age > _BOOK_CACHE_TTL * 2:
            age_style = "yellow"
            age_str = f"{age:.0f}s ago"
        else:
            age_style = "dim italic"
            age_str = f"{age:.0f}s ago" if age > 0 else ""
        tbl.add_row(
            Text("", style="dim"),
            Text(f"Last: {last_str}", style="bold yellow"),
            Text(age_str, style=age_style),
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
    is_stale = _book_fetch_ok_ts > 0 and (time.time() - _book_fetch_ok_ts) > 15
    panel_title = "[bold red]Order Book — STALE[/bold red]" if is_stale else "[dim]Order Book — Trade Up[/dim]"
    panel_border = "red" if is_stale else "dim"
    return Panel(
        tbl,
        title=panel_title,
        subtitle=f"[dim]{vol_str}[/dim]" if vol_str else None,
        border_style=panel_border,
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


def _compute_total_unrealized(
    bot_state: dict, market: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """Compute total unrealized PnL from active inventories + CLOB midpoints.

    Restricts to the current market's tokens (if provided) to avoid hammering
    the CLOB API for stale resolved-market positions.
    """
    inventories = bot_state.get("inventories", {})
    current_tokens: set = set()
    if market:
        if market.get("up_token"):
            current_tokens.add(market["up_token"])
        if market.get("down_token"):
            current_tokens.add(market["down_token"])
    active = {
        tid: inv for tid, inv in inventories.items()
        if inv.get("position", 0) != 0 and (not current_tokens or tid in current_tokens)
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

_snapshot_cache: Optional[Dict[str, Any]] = None
_snapshot_cache_mtime: float = 0.0


def _load_snapshot(path: str) -> Optional[Dict[str, Any]]:
    """Read bot_snapshot.json (cycle source of truth). Cached; re-reads on mtime change."""
    global _snapshot_cache, _snapshot_cache_mtime
    try:
        mtime = os.path.getmtime(path)
        if _snapshot_cache is not None and mtime == _snapshot_cache_mtime:
            return _snapshot_cache
        with open(path, "r") as f:
            data = json.load(f)
        _snapshot_cache = data
        _snapshot_cache_mtime = mtime
        return data
    except (OSError, json.JSONDecodeError):
        return _snapshot_cache  # return stale on error


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
    # Filter to non-zero positions, limit to current market tokens to avoid
    # hammering the CLOB API for 100+ stale resolved-market positions.
    current_tokens: set = set()
    if market:
        if market.get("up_token"):
            current_tokens.add(market["up_token"])
        if market.get("down_token"):
            current_tokens.add(market["down_token"])
    active = {
        tid: inv for tid, inv in inventories.items()
        if inv.get("position", 0) != 0 and (not current_tokens or tid in current_tokens)
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
    last_slot_pnl = bot_state.get("last_slot_pnl", 0.0)
    last_slot_outcome = bot_state.get("last_slot_outcome", "")

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=10)
    tbl.add_column(justify="right")
    tbl.add_row("Slot PnL", Text(f"{slot_pnl:+.4f}", style=f"bold {slot_style}"))
    # Show the just-settled slot's PnL so the user sees the result of a
    # hold-to-expiry resolution immediately after rollover, when
    # slot_realized_pnl has already been reset to 0.0 for the new slot.
    if last_slot_pnl != 0.0 or last_slot_outcome:
        ls_style = "green" if last_slot_pnl >= 0 else "red"
        outcome_tag = f" ({last_slot_outcome})" if last_slot_outcome else ""
        tbl.add_row("Last Slot", Text(f"{last_slot_pnl:+.4f}{outcome_tag}", style=ls_style))
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
    snapshot: Optional[dict] = None,
) -> Panel:
    """Build the Bot Status panel.

    Prefers snapshot data (written by bot every cycle) over inferred state.
    """
    if bot_state is None and snapshot is None:
        return Panel(
            Text("Bot offline", style="dim italic"),
            title="[dim]Bot Status[/dim]",
            border_style="dim",
        )

    cycle = (snapshot or bot_state or {}).get("cycle_count", 0)

    # Active orders: snapshot has full Order objects; fallback to order ID count
    if snapshot and snapshot.get("active_orders") is not None:
        order_count = len(snapshot["active_orders"])
    else:
        orders = (bot_state or {}).get("active_order_ids", {})
        order_count = sum(1 for v in orders.values() if v is not None) if isinstance(orders, dict) else 0

    # Bot status: use snapshot's explicit enum when fresh (< 10 min old)
    snap_status = snapshot.get("bot_status") if snapshot else None
    snap_age = (time.time() - float(snapshot.get("updated_at", 0))) if snapshot else float("inf")

    if file_mtime is not None and (time.time() - file_mtime) >= 600:
        status = Text(f"STOPPED ({(time.time() - file_mtime) / 60:.0f}m ago)", style="bold red")
    elif snap_status and snap_age < 600:
        _style_map = {
            "IN_POSITION": "bold cyan",
            "EVALUATING":  "bold green",
            "COOLDOWN":    "bold yellow",
            "ERROR":       "bold red",
            "STOPPED":     "bold red",
            "INIT":        "dim",
        }
        status = Text(snap_status, style=_style_map.get(snap_status, "white"))
    else:
        # Fallback: derive from legacy strategy_status field
        strategy_status = (bot_state or {}).get("strategy_status", "")
        if strategy_status == "POSITION_OPEN":
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

    # Show YES/NO mids from snapshot when available
    if snapshot:
        yes_mid = snapshot.get("yes_mid")
        no_mid = snapshot.get("no_mid")
        if yes_mid is not None:
            tbl.add_row("YES mid", Text(f"{yes_mid:.4f}", style="green"))
        if no_mid is not None:
            tbl.add_row("NO mid", Text(f"{no_mid:.4f}", style="red"))

    return Panel(tbl, title="[dim]Bot Status[/dim]", border_style="dim")


# ── Market Cycle panel ────────────────────────────────────────────────────────

_BINANCE_RETRY_S = 30.0  # seconds before retrying a failed Binance fetch


def _fetch_slot_open_from_binance(slot_ts: int) -> Optional[Tuple[float, str]]:
    """Fetch BTC/USDT 1-minute open price at slot_ts from Binance REST.

    Cached per slot on success. Failures are suppressed for 30s to avoid
    hammering the API on every dashboard refresh cycle.
    """
    if slot_ts in _slot_price_rest_cache:
        return _slot_price_rest_cache[slot_ts]
    last_fail = _slot_price_rest_failed.get(slot_ts)
    if last_fail is not None and (time.time() - last_fail) < _BINANCE_RETRY_S:
        return None
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": slot_ts * 1000,  # ms
                "limit": 1,
            },
            timeout=5,
        )
        resp.raise_for_status()
        klines = resp.json()
        if klines:
            open_price = float(klines[0][1])  # index 1 = open
            result = (open_price, "Binance 1m (est.)")
            _slot_price_rest_cache[slot_ts] = result
            return result
    except Exception as e:
        logging.getLogger(__name__).debug("Binance kline fetch failed for slot %s: %s", slot_ts, e)
        _slot_price_rest_failed[slot_ts] = time.time()
    return None


def _fetch_slot_outcome_bg(slot_ts: int) -> None:
    """Background thread: fetch and cache the resolved outcome for a past slot."""
    slug = f"{_slug_prefix}-{slot_ts}"
    result: Optional[str] = None
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": slug},
            timeout=5,
        )
        resp.raise_for_status()
        events = resp.json()
        for event in (events if isinstance(events, list) else [events]):
            for m in event.get("markets", []):
                raw_prices = m.get("outcomePrices")
                if not raw_prices:
                    continue
                try:
                    prices = json.loads(raw_prices)
                    if len(prices) >= 2:
                        p0, p1 = float(prices[0]), float(prices[1])
                        if p0 >= 0.99:
                            result = "Up"
                        elif p1 >= 0.99:
                            result = "Down"
                except Exception:
                    pass
    except Exception:
        pass

    if result is not None:
        with _slot_outcome_lock:
            _slot_outcome_cache[slot_ts] = result


def _ensure_slot_outcomes(slot_tss: List[int]) -> None:
    """Kick off background fetches for uncached/unretried slots (non-blocking)."""
    now = time.time()
    current_slot = _current_slot_ts(now)
    cutoff = current_slot - 48 * 3600  # evict entries older than 48h

    for slot_ts in slot_tss:
        with _slot_outcome_lock:
            if slot_ts in _slot_outcome_cache:
                continue
            last = _slot_outcome_retry_ts.get(slot_ts, 0)
            # Recently-closed slot: retry every 10s until resolved
            retry_s = 10.0 if slot_ts >= current_slot - SLOT_INTERVAL_S else _SLOT_OUTCOME_RETRY_S
            if now - last < retry_s:
                continue
            _slot_outcome_retry_ts[slot_ts] = now
        threading.Thread(target=_fetch_slot_outcome_bg, args=(slot_ts,), daemon=True).start()

    # Evict stale entries (runs at most once per render cycle, O(n) on cache size)
    with _slot_outcome_lock:
        stale = [s for s in _slot_outcome_cache if s < cutoff]
        for s in stale:
            del _slot_outcome_cache[s]
            _slot_outcome_retry_ts.pop(s, None)


def _resolve_price_to_beat(
    chainlink_feed: Optional[ChainlinkFeed],
    bot_state: Optional[dict] = None,
    current_slot: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    """Resolve the authoritative current-slot price-to-beat."""
    slot_ts = current_slot if current_slot is not None else _current_slot_ts()

    # 1. Live Chainlink slot-open (best)
    if chainlink_feed is not None:
        slot_open = chainlink_feed.get_slot_open_price()
        if slot_open is not None and slot_open.slot_ts == slot_ts:
            return slot_open.price, "Chainlink (live)"

    # 2. Bot state snapshot
    if bot_state:
        bot_price = bot_state.get("chainlink_ref_price")
        bot_slot = bot_state.get("chainlink_ref_slot_ts")
        if bot_price is not None and bot_slot == slot_ts:
            return float(bot_price), "Chainlink (bot snapshot)"

    # 3. ChainlinkFeed buffer — earliest tick this slot (mid-slot reconnect case)
    if chainlink_feed is not None:
        buffered = chainlink_feed.get_earliest_slot_price(slot_ts)
        if buffered is not None:
            return buffered, "Chainlink (buffered)"

    # 4. Binance 1m REST — approximation when feed is completely down
    binance = _fetch_slot_open_from_binance(slot_ts)
    if binance is not None:
        return binance

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
    if isinstance(end_ts, (int, float)) and current_slot < end_ts <= current_slot + SLOT_INTERVAL_S:
        slot_end = float(end_ts)
    else:
        slot_end = current_slot + SLOT_INTERVAL_S
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

    # Chainlink health prefers the dashboard-owned feed.
    cl_healthy = chainlink_feed.is_healthy() if chainlink_feed is not None else False

    # Use Chainlink live price for BTC Now so Delta is consistent with settlement.
    # Fall back to exchange mid only when Chainlink has no data yet.
    cl_latest = chainlink_feed.get_latest() if chainlink_feed is not None else None
    if cl_latest is not None:
        btc_now = cl_latest.price
        btc_now_source = "Chainlink"
    else:
        book = feed.get_latest_book()
        btc_now = book.mid if book is not None else None
        btc_now_source = "Coinbase" if btc_now is not None else None

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

    if cl_healthy:
        tbl.add_row("Chainlink", Text("LIVE", style="bold green"))
    elif chainlink_feed is not None and chainlink_feed.is_connecting():
        tbl.add_row("Chainlink", Text("Connecting...", style="bold yellow"))
    elif chainlink_feed is not None and chainlink_feed.get_latest() is not None:
        age_s = (chainlink_feed.get_feed_age_ms() or 0) / 1000
        tbl.add_row("Chainlink", Text(f"STALE ({age_s:.0f}s)", style="yellow"))
    else:
        tbl.add_row("Chainlink", Text("DOWN", style="bold red"))

    if btc_now is not None:
        now_label = f"BTC Now ({btc_now_source})" if btc_now_source else "BTC Now"
        tbl.add_row(now_label, Text(f"${btc_now:,.2f}", style="bold yellow"))
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

def _build_strategy_panel(
    bot_state: Optional[dict],
    snapshot: Optional[dict] = None,
) -> Panel:
    """Show strategy internal reasoning from state/snapshot telemetry."""
    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=10)
    tbl.add_column(justify="right")

    if bot_state is None and snapshot is None:
        return Panel(
            Text("No strategy data", style="dim italic"),
            title="[dim]Strategy[/dim]",
            border_style="dim",
        )

    def _pick(key: str, default=None):
        if bot_state is not None and bot_state.get(key) is not None:
            return bot_state.get(key)
        if snapshot is not None and snapshot.get(key) is not None:
            return snapshot.get(key)
        return default

    name = _pick("strategy_name", "") or "–"
    tbl.add_row("Strategy", Text(name, style="white"))

    status = _pick("strategy_status", "") or "–"
    if status == "POSITION_OPEN":
        status_style = "bold cyan"
    elif status == "WATCHING":
        status_style = "bold green"
    elif status == "EXITED":
        status_style = "bold yellow"
    else:
        status_style = "dim"
    tbl.add_row("Status", Text(status, style=status_style))

    skip_reason = _pick("strategy_skip_reason", "")
    if skip_reason and status == "WATCHING":
        if skip_reason.startswith("edge_low"):
            skip_style = "yellow"
        elif skip_reason.startswith("spread_wide"):
            skip_style = "bold red"
        elif skip_reason.startswith("tte"):
            skip_style = "cyan"
        elif skip_reason in ("in_position", "no_prediction", "model_not_ready"):
            skip_style = "dim"
        else:
            skip_style = "cyan"
        tbl.add_row("Skip", Text(skip_reason[:50], style=skip_style))

    bias = _pick("strategy_bias", "")
    if bias:
        bias_style = "bold green" if bias == "LONG" else ("bold red" if bias == "SHORT" else "dim")
        tbl.add_row("Bias", Text(bias, style=bias_style))

    zscore = _pick("strategy_zscore")
    if zscore is not None:
        z_style = "bold red" if abs(zscore) >= 1.8 else "white"
        tbl.add_row("Z-score", Text(f"{zscore:+.3f}", style=z_style))

    momentum_pct = _pick("strategy_momentum_pct")
    if momentum_pct is not None:
        mom_style = "bold green" if momentum_pct >= 0 else "bold red"
        tbl.add_row("Momentum", Text(f"{momentum_pct * 100:+.2f}%", style=mom_style))

    prob_yes = _pick("strategy_prob_yes")
    if prob_yes is not None:
        prob_style = "bold green" if prob_yes >= 0.5 else "bold red"
        tbl.add_row("Prob YES", Text(f"{prob_yes:.3f}", style=prob_style))

    prob_no = _pick("strategy_prob_no")
    if prob_no is not None:
        prob_style = "bold green" if prob_no >= 0.5 else "bold red"
        tbl.add_row("Prob NO", Text(f"{prob_no:.3f}", style=prob_style))

    distance_pct = _pick("strategy_distance_to_break_pct")
    distance_bps = _pick("strategy_distance_to_strike_bps")
    if distance_pct is not None:
        distance_style = "bold green" if distance_pct >= 0 else "bold red"
        suffix = ""
        if distance_bps is not None:
            suffix = f" ({distance_bps:+.0f} bps)"
        tbl.add_row("Px vs K", Text(f"{distance_pct * 100:+.2f}%{suffix}", style=distance_style))

    tte_seconds = _pick("strategy_tte_seconds")
    if tte_seconds is None:
        tte_seconds = _pick("tte_seconds")
    if tte_seconds is not None:
        tbl.add_row("TTE", Text(f"{tte_seconds:.0f}s", style="dim"))

    edge_yes = _pick("strategy_edge_yes")
    if edge_yes is not None:
        edge_style = "bold green" if edge_yes >= 0 else "bold red"
        tbl.add_row("Edge YES", Text(f"{edge_yes:+.3f}", style=edge_style))

    edge_no = _pick("strategy_edge_no")
    if edge_no is not None:
        edge_style = "bold green" if edge_no >= 0 else "bold red"
        tbl.add_row("Edge NO", Text(f"{edge_no:+.3f}", style=edge_style))

    net_edge_yes = _pick("strategy_net_edge_yes")
    if net_edge_yes is not None:
        edge_style = "bold green" if net_edge_yes >= 0 else "bold red"
        tbl.add_row("Net YES", Text(f"{net_edge_yes:+.3f}", style=edge_style))

    net_edge_no = _pick("strategy_net_edge_no")
    if net_edge_no is not None:
        edge_style = "bold green" if net_edge_no >= 0 else "bold red"
        tbl.add_row("Net NO", Text(f"{net_edge_no:+.3f}", style=edge_style))

    exp_fill_yes = _pick("strategy_expected_fill_yes")
    if exp_fill_yes is not None:
        tbl.add_row("Fill YES", Text(f"{exp_fill_yes:.3f}", style="dim"))

    exp_fill_no = _pick("strategy_expected_fill_no")
    if exp_fill_no is not None:
        tbl.add_row("Fill NO", Text(f"{exp_fill_no:.3f}", style="dim"))

    req_edge = _pick("strategy_required_edge")
    if req_edge is not None:
        tbl.add_row("Req Edge", Text(f"{req_edge:.3f}", style="dim"))

    breakdown = _pick("strategy_score_breakdown", {}) or {}
    if breakdown:
        tbl.add_row("── score ──", Text("", style="dim"))

        def _contrib_row(label: str, val: float):
            style = "green" if val > 0 else ("red" if val < 0 else "dim")
            return label, Text(f"{val:+.4f}", style=style)

        tbl.add_row(*_contrib_row("Distance", breakdown.get("dist_contrib", 0)))
        tbl.add_row(*_contrib_row("Momentum 1m", breakdown.get("mom1_contrib", 0)))
        tbl.add_row(*_contrib_row("Momentum 3m", breakdown.get("mom3_contrib", 0)))
        tbl.add_row(*_contrib_row("Momentum 5m", breakdown.get("mom5_contrib", 0)))
        td_c = breakdown.get("td_contrib", 0)
        if td_c != 0:
            tbl.add_row(*_contrib_row("TD", td_c))
        tbl.add_row("Time Weight", Text(f"{breakdown.get('time_weight', 1.0):.2f}x", style="dim"))
        raw_score = breakdown.get("score", 0)
        tbl.add_row("Score", Text(f"{raw_score:+.4f}", style="bold green" if raw_score > 0 else "bold red"))

    model_version = _pick("strategy_model_version", "")
    if model_version:
        tbl.add_row("Model", Text(model_version[-18:], style="white"))

    feature_status = _pick("strategy_feature_status", "")
    if feature_status:
        status_style = "bold green" if feature_status.startswith("ready") else "bold yellow"
        tbl.add_row("Features", Text(feature_status, style=status_style))

    last_sig = _pick("strategy_last_signal", "")
    last_ts = _pick("strategy_last_signal_ts", 0.0)
    if last_sig:
        age_s = int(time.time() - last_ts) if last_ts else 0
        age_str = f"{age_s}s ago"
        tbl.add_row("Last sig", Text(f"{last_sig[:50]}", style="dim"))
        tbl.add_row("", Text(age_str, style="dim"))
    else:
        tbl.add_row("Last sig", Text("–", style="dim"))

    return Panel(tbl, title="[dim]Strategy[/dim]", border_style="dim")


# ── Trades panel helpers ──────────────────────────────────────────────────────
#
# The Trades panel renders one row per fill from bot_state.trade_log (schema
# enriched in src/engine/cycle_runner.py::execute_signals, ~line 767). All
# three format helpers are pure functions of a single entry dict so they're
# trivial to unit-test and tweak without touching the builder.

# Narrow-mode breakpoint (panel width below which we drop low-priority cols).
_TRADES_NARROW_WIDTH = 100


def _trade_status(entry: dict, inventories: Optional[dict]) -> tuple[str, str]:
    """Derive (label, style) for the STATUS cell of a trade row.

    Priority:
      1. OPEN      — row's token_id still has nonzero inventory position
      2. SETTLED   — closing fill with a nonzero realized_pnl_delta
      3. FILLED    — entry fill (BUY that opened or SELL that didn't realize)

    CANCELED is a placeholder for a future extension (cycle_runner would need
    to write canceled-order events into trade_log; it doesn't today).
    """
    token_id = entry.get("token_id")
    if token_id and inventories:
        inv = inventories.get(token_id)
        position = 0.0
        if isinstance(inv, dict):
            position = float(inv.get("position", 0.0) or 0.0)
        elif inv is not None:
            position = float(getattr(inv, "position", 0.0) or 0.0)
        if abs(position) > 1e-9:
            return "OPEN", "bold cyan"

    delta = entry.get("realized_pnl_delta")
    if delta is not None and abs(float(delta)) > 1e-9:
        return "SETTLED", "bold white"

    return "FILLED", "dim white"


def _fmt_pnl(delta: Optional[float]) -> Text:
    """Render a realized-PnL delta as a signed dollar string with sign-based color."""
    if delta is None or abs(float(delta)) < 1e-9:
        return Text("—", style="dim")
    d = float(delta)
    sign = "+" if d >= 0 else "-"
    return Text(f"{sign}${abs(d):.2f}", style="bold green" if d >= 0 else "bold red")


def _fmt_tte(seconds: Optional[float]) -> str:
    """Format a time-to-expiry value (seconds) like '73s' or '2m15s'."""
    if seconds is None:
        return "—"
    s = int(max(0.0, float(seconds)))
    return f"{s // 60}m{s % 60}s" if s >= 60 else f"{s}s"


def _fmt_slot(slot_expiry_ts: Optional[float]) -> str:
    """Render a slot expiry timestamp as a local HH:MM label."""
    if not slot_expiry_ts:
        return "—"
    return datetime.fromtimestamp(float(slot_expiry_ts)).strftime("%H:%M")


def _build_trade_log_panel(
    bot_state: Optional[dict],
    panel_width: Optional[int] = None,
) -> Panel:
    """Show recent fills from bot_state.trade_log as a single-row-per-trade table.

    Columns (wide mode, panel width ≥ _TRADES_NARROW_WIDTH):
        ▶  TIME  STRAT  SLOT  TTE  SIDE  OUT  PRICE  SHARES  NOTIONAL  EDGE  STATUS  PnL

    Narrow mode drops TTE, NOTIONAL, and EDGE and shrinks STRAT to 8 chars.
    Open rows (inventory still long in this token) get a leading ▶ marker.
    """
    trades = (bot_state or {}).get("trade_log", [])
    if not trades:
        return Panel(
            Text("No trades yet", style="dim italic"),
            title="[dim]Trades[/dim]",
            border_style="dim",
        )

    # Width detection. Caller may pass an explicit width (tests); otherwise
    # ask Rich for the terminal's current width. Fall back to wide mode if
    # detection fails.
    if panel_width is None:
        try:
            panel_width = Console().width
        except Exception:
            panel_width = 200
    narrow = panel_width < _TRADES_NARROW_WIDTH
    strat_width = 8 if narrow else 10

    inventories = (bot_state or {}).get("inventories") or {}

    tbl = Table(show_header=True, box=None, padding=(0, 1), expand=True,
                header_style="dim")
    tbl.add_column("",         width=1, no_wrap=True)                   # ▶ marker
    tbl.add_column("TIME",     width=8, no_wrap=True)
    tbl.add_column("STRAT",    width=strat_width, no_wrap=True)
    tbl.add_column("SLOT",     width=5, no_wrap=True)
    if not narrow:
        tbl.add_column("TTE",  width=6, justify="right", no_wrap=True)
    tbl.add_column("SIDE",     width=4, no_wrap=True)
    tbl.add_column("OUT",      width=3, no_wrap=True)
    tbl.add_column("PRICE",    width=6, justify="right", no_wrap=True)
    tbl.add_column("SHARES",   width=6, justify="right", no_wrap=True)
    if not narrow:
        tbl.add_column("NOTIONAL", width=8, justify="right", no_wrap=True)
        tbl.add_column("EDGE",     width=7, justify="right", no_wrap=True)
    tbl.add_column("STATUS",   width=9, no_wrap=True)
    tbl.add_column("PnL",      width=8, justify="right", no_wrap=True)

    for entry in reversed(trades[-10:]):
        action  = str(entry.get("action", ""))
        outcome = str(entry.get("outcome", ""))
        price   = float(entry.get("price", 0.0) or 0.0)
        size    = float(entry.get("size", 0.0) or 0.0)
        notional = price * size

        # Settlement rows use a distinct style: show entry→settle with
        # the resolution outcome and realized PnL highlighted.
        is_settle = action == "SETTLE"

        if is_settle:
            resolved = entry.get("resolved_outcome", "")
            entry_px = float(entry.get("entry_price", 0.0) or 0.0)
            status_label = f"{'WIN' if entry.get('realized_pnl_delta', 0) >= 0 else 'LOSS'} ({resolved})"
            status_style = "bold green" if entry.get("realized_pnl_delta", 0) >= 0 else "bold red"
            is_open = False
            side_style = "bold yellow"
            out_style = "green" if outcome == "YES" else "red"
        else:
            status_label, status_style = _trade_status(entry, inventories)
            is_open = status_label == "OPEN"
            side_style = "bold green" if action == "BUY" else "bold red"
            out_style  = "green" if outcome == "YES" else "red"

        edge_val = entry.get("edge")
        edge_str = f"{edge_val:+.3f}" if isinstance(edge_val, (int, float)) else "—"

        strat = str(entry.get("strategy_name") or "—")
        if len(strat) > strat_width:
            strat = strat[: strat_width - 1] + "…"

        ts_val = entry.get("ts") or time.time()
        time_str = datetime.fromtimestamp(float(ts_val)).strftime("%H:%M:%S")

        # For SETTLE rows, show entry_price→settle_price in the PRICE col
        if is_settle:
            entry_px = float(entry.get("entry_price", 0.0) or 0.0)
            price_str = f"{entry_px:.2f}→{price:.2f}"
        else:
            price_str = f"${price:.2f}"

        row = [
            Text("▶" if is_open else " ", style="bold cyan" if is_open else "dim"),
            Text(time_str, style="white"),
            Text(strat, style="white"),
            Text(_fmt_slot(entry.get("slot_expiry_ts")), style="white"),
        ]
        if not narrow:
            row.append(Text(_fmt_tte(entry.get("seconds_to_expiry")), style="dim"))
        row += [
            Text(action, style=side_style),
            Text(outcome, style=out_style),
            Text(price_str, style="white"),
            Text(f"{size:.1f}", style="white"),
        ]
        if not narrow:
            row.append(Text(f"${notional:.2f}", style="white"))
            row.append(Text(edge_str, style="white"))
        row += [
            Text(status_label, style=status_style),
            _fmt_pnl(entry.get("realized_pnl_delta")),
        ]
        tbl.add_row(*row)

    return Panel(tbl, title="[dim]Trades[/dim]", border_style="dim")


def _build_slot_history_panel(
    bot_state: Optional[dict],
    n_slots: int = 12,
) -> Panel:
    """Show last N resolved 5-min slots: market outcome (green=Up, red=Down) and my bet."""
    now = _server_now()
    current_slot = _current_slot_ts(now)

    past_slots = [current_slot - SLOT_INTERVAL_S * i for i in range(n_slots - 1, -1, -1)]

    # Trigger background fetches (non-blocking)
    _ensure_slot_outcomes(past_slots)

    # Build trade lookup: slot_ts → outcome side ("YES"/"NO") from first BUY per slot
    trade_lookup: Dict[int, str] = {}
    if bot_state:
        for entry in (bot_state.get("trade_log") or []):
            if entry.get("action") != "BUY":
                continue
            slot = _current_slot_ts(entry.get("ts") or now)
            if slot not in trade_lookup:
                trade_lookup[slot] = entry.get("outcome", "")

    # Snapshot outcome cache once (avoids per-slot lock acquisition in render loop)
    with _slot_outcome_lock:
        cached_outcomes = {s: _slot_outcome_cache.get(s) for s in past_slots}

    # Build table: label column + one column per slot
    tbl = Table(show_header=True, box=None, padding=(0, 0), expand=True)
    tbl.add_column("", width=7, style="dim", no_wrap=True)
    for slot_ts in past_slots:
        label = datetime.fromtimestamp(slot_ts + SLOT_INTERVAL_S, tz=timezone.utc).strftime("%H:%M")
        tbl.add_column(label, justify="center", min_width=5, no_wrap=True)

    # Build both rows in a single pass
    market_cells: List[Any] = [Text("Market", style="dim")]
    my_cells: List[Any] = [Text("My bet", style="dim")]
    for slot_ts in past_slots:
        outcome = cached_outcomes.get(slot_ts)
        side = trade_lookup.get(slot_ts)

        if outcome == "Up":
            market_cells.append(Text(" ● ", style="bold green"))
        elif outcome == "Down":
            market_cells.append(Text(" ● ", style="bold red"))
        else:
            market_cells.append(Text(" ? ", style="dim"))

        if side is None:
            my_cells.append(Text(" ○ ", style="dim"))
        elif side in ("YES", "Up"):
            my_cells.append(Text(" ● ", style="bold green"))
        else:
            my_cells.append(Text(" ● ", style="bold red"))

    tbl.add_row(*market_cells)
    tbl.add_row(*my_cells)

    return Panel(tbl, title="[dim]Slot History  (past 1h · green=Up · red=Down)[/dim]", border_style="dim")


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


# ── Config panel ──────────────────────────────────────────────────────────────
#
# Runtime config doesn't change mid-session — the bot process reads
# config/config.yaml once at launch and never re-loads it. So the dashboard
# does the same: yaml.safe_load runs exactly once per dashboard process and
# the projected summary is cached in module state. Operator-visible edits
# to the file require a dashboard restart (documented in panel footer).

_runtime_config_cache: Optional[Dict[str, Any]] = None


def _build_runtime_summary(raw: dict, active_strategy: Optional[str]) -> Dict[str, Any]:
    """Project raw config.yaml into the fields the operator cares about during live trading.

    Returns a flat dict; pure function of inputs so it's trivially unit-testable.
    If active_strategy is None or not in cfg.strategies, falls back to the
    first enabled strategy block (or the first strategy at all).
    """
    trading = raw.get("trading", {}) or {}
    risk = raw.get("risk", {}) or {}
    strategies = raw.get("strategies", {}) or {}

    # Resolve the active strategy block.
    strat_cfg: dict = {}
    strat_name = active_strategy or ""
    if strat_name and strat_name in strategies and isinstance(strategies[strat_name], dict):
        strat_cfg = strategies[strat_name]
    else:
        for name, block in strategies.items():
            if isinstance(block, dict) and block.get("enabled", True):
                strat_name, strat_cfg = name, block
                break
        else:
            if strategies:
                strat_name = next(iter(strategies))
                strat_cfg = strategies[strat_name] if isinstance(strategies[strat_name], dict) else {}

    # Position sizing: prefer min-max range, fall back to single cap, then
    # to a min-only floor ("≥$10") when only the floor is configured.
    size_min = strat_cfg.get("min_position_size_usdc")
    size_max = strat_cfg.get("max_position_size_usdc") or strat_cfg.get("position_size_usdc")
    if size_min is not None and size_max is not None:
        max_position: Optional[str] = f"${size_min:g}–${size_max:g}"
    elif size_max is not None:
        max_position = f"${size_max:g}"
    elif size_min is not None:
        max_position = f"≥${size_min:g}"
    else:
        max_position = None

    def _pct(x: Any) -> Optional[str]:
        if x is None:
            return None
        try:
            return f"{float(x) * 100:.0f}%"
        except (TypeError, ValueError):
            return None

    def _money(x: Any) -> Optional[str]:
        if x is None:
            return None
        try:
            return f"${float(x):g}"
        except (TypeError, ValueError):
            return None

    def _fnum(x: Any, fmt: str) -> Optional[str]:
        if x is None:
            return None
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return None

    # Values of None mean "not configured" — _build_config_panel drops those
    # rows entirely rather than rendering a meaningless "—".
    return {
        "mode": "PAPER" if trading.get("paper_trading", True) else "LIVE",
        "interval": f"{int(trading.get('interval', 300))}s "
                    f"({int(trading.get('interval', 300)) // 60}m)",
        "strategy": strat_name or "—",
        "model_dir": strat_cfg.get("model_dir") or None,
        "max_position": max_position,
        "kelly": _fnum(strat_cfg.get("kelly_fraction"), ".2f"),
        "min_edge": _fnum(strat_cfg.get("delta") or strat_cfg.get("min_edge"), ".3f"),
        "min_conf": _fnum(strat_cfg.get("min_confidence"), ".2f"),
        "max_exposure": _pct(risk.get("max_total_exposure")),
        "daily_loss": _pct(risk.get("max_daily_loss")),
        "session_loss": _money(risk.get("max_session_loss_usdc")),
    }


def _load_runtime_config(config_path: str, active_strategy: Optional[str]) -> Dict[str, Any]:
    """Load and project config.yaml exactly once per process, then cache."""
    global _runtime_config_cache
    if _runtime_config_cache is not None:
        return _runtime_config_cache
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    _runtime_config_cache = _build_runtime_summary(raw, active_strategy)
    return _runtime_config_cache


def _build_config_panel(
    bot_state: Optional[dict],
    config_path: str,
) -> Panel:
    """Render a concise runtime summary of the active process's config.

    Reads config.yaml once per dashboard lifetime (see _load_runtime_config).
    Operator edits require a dashboard restart to take effect.
    """
    active_strategy = (bot_state or {}).get("strategy_name") or None
    try:
        summary = _load_runtime_config(config_path, active_strategy)
    except (OSError, yaml.YAMLError) as e:
        return Panel(
            Text(f"Cannot load {config_path}:\n{e}", style="bold red"),
            title="[dim]Config[/dim]",
            border_style="dim",
        )

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    tbl.add_column(style="dim", no_wrap=True)
    tbl.add_column(justify="left", no_wrap=True)

    mode_style = "bold yellow" if summary["mode"] == "PAPER" else "bold green"

    def _add(label: str, key: str, style: str = "white") -> None:
        """Append a row only if the summary has a real value for this key."""
        val = summary.get(key)
        if val is None:
            return
        tbl.add_row(label, Text(str(val), style=style))

    tbl.add_row("mode",      Text(summary["mode"], style=mode_style))
    _add("interval",  "interval")
    tbl.add_row("", "")

    _add("strategy",  "strategy",     style="bold white")
    _add("model",     "model_dir")
    _add("position",  "max_position")
    _add("kelly",     "kelly")
    _add("min edge",  "min_edge")
    _add("min conf",  "min_conf")
    tbl.add_row("", "")

    _add("exposure",  "max_exposure")
    _add("daily cap", "daily_loss")
    _add("session",   "session_loss")

    return Panel(
        tbl,
        title=f"[dim]Config  {os.path.basename(config_path)}[/dim]",
        border_style="dim",
    )


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
    config_path: str = "config/config.yaml",
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
    snapshot = _load_snapshot(state_file.replace(".json", "_snapshot.json"))
    live_positions = _fetch_live_positions()

    # Build token→outcome map from snapshot (avoids extra market API call)
    if snapshot and market is None:
        snap_market = {
            "up_token": snapshot.get("yes_token_id", ""),
            "down_token": snapshot.get("no_token_id", ""),
        }
    else:
        snap_market = market

    positions_panel = _build_positions_panel(
        bot_state, live_positions, market=snap_market or market
    )
    # Use pre-computed unrealized PnL from snapshot (no extra CLOB API calls)
    snap_upnl = snapshot.get("unrealized_pnl") if snapshot else None
    total_upnl = snap_upnl if snap_upnl is not None else (
        _compute_total_unrealized(bot_state, market=snap_market or market) if bot_state else None
    )
    pnl_panel = _build_pnl_panel(bot_state, total_upnl=total_upnl)
    bot_status_panel = _build_bot_status_panel(bot_state, file_mtime, snapshot=snapshot)

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

    # ── Slot history panel (full-width) ───────────────────────────────────
    slot_history_panel = _build_slot_history_panel(bot_state)

    # ── Market Cycle + Strategy + Config panels (bottom row, 3-col) ──────
    # Trades is promoted to its own full-width row below slot history so it
    # has room for strategy/slot/status/PnL without truncation.
    market_cycle_panel = _build_market_cycle_panel(
        feed,
        chainlink_feed,
        bot_state=bot_state,
        market=market,
    )
    strategy_panel = _build_strategy_panel(bot_state, snapshot=snapshot)
    trade_log_panel = _build_trade_log_panel(bot_state)
    config_panel = _build_config_panel(bot_state, config_path)

    bottom_row = Table.grid(expand=True)
    bottom_row.add_column(ratio=1)
    bottom_row.add_column(ratio=1)
    bottom_row.add_column(ratio=1)
    bottom_row.add_row(market_cycle_panel, strategy_panel, config_panel)

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
    outer.add_row(slot_history_panel)
    outer.add_row(trade_log_panel)
    outer.add_row(bottom_row)
    outer.add_row(log_panel)
    outer.add_row(footer)

    symbol = getattr(feed, "_symbol", "btcusdt").upper()
    return Panel(
        outer,
        title=f"[bold]BTC Feed Monitor[/bold]  [dim]{symbol}[/dim]",
        border_style=border_style,
    )


# ── Interactive config picker ─────────────────────────────────────────────────

def _pick_config(default: str) -> str:
    """
    Show an arrow-key menu to pick a config YAML before the TUI starts.
    Returns the chosen path.  If only one file exists, skips the menu.
    Skips when not running in an interactive TTY (e.g. piped / CI).
    """
    config_dir = os.path.dirname(default) or "config"
    candidates = sorted(_glob.glob(os.path.join(config_dir, "*.yaml")))

    # If --config was explicitly set to a non-default path, honour it directly.
    # Also fall through when there's nothing to choose from.
    if not candidates:
        return default
    if len(candidates) == 1:
        return candidates[0]
    if not sys.stdin.isatty():
        return default

    console = Console()
    idx = candidates.index(default) if default in candidates else 0

    def _render(selected: int) -> str:
        lines = ["\n[bold]Select a config[/bold]  (↑/↓ arrow, Enter to confirm)\n"]
        for i, path in enumerate(candidates):
            name = os.path.basename(path)
            if i == selected:
                lines.append(f"  [bold cyan]▶  {name}[/bold cyan]")
            else:
                lines.append(f"     [dim]{name}[/dim]")
        lines.append("")
        return "\n".join(lines)

    # Read a single raw keypress (no echo)
    def _getch() -> bytes:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.buffer.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    console.print(_render(idx))

    while True:
        ch = _getch()
        if ch == b"\r" or ch == b"\n":
            break
        if ch == b"\x1b":
            # ESC sequence — read two more bytes for arrow keys
            ch2 = _getch()
            ch3 = _getch()
            if ch2 == b"[":
                if ch3 == b"A":   # up
                    idx = (idx - 1) % len(candidates)
                elif ch3 == b"B": # down
                    idx = (idx + 1) % len(candidates)
        elif ch == b"q" or ch == b"\x03":  # q or Ctrl-C
            sys.exit(0)
        # Reprint — move cursor up to overwrite previous menu
        lines_printed = len(candidates) + 3
        sys.stdout.write(f"\x1b[{lines_printed}A")
        console.print(_render(idx))

    chosen = candidates[idx]
    console.print(f"\n[dim]Using config:[/dim] [cyan]{chosen}[/cyan]\n")
    return chosen


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live BTC/USDT price feed monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--asset",    default="BTC",           help="Asset to monitor: BTC, ETH, SOL, DOGE, XRP (default: BTC)")
    parser.add_argument("--symbol",   default=None,            help="Exchange symbol (Coinbase: BTC-USD, Binance.US: btcusd). Defaults to exchange's canonical symbol.")
    parser.add_argument("--exchange", default="coinbase",     choices=["coinbase", "binance_us"],
                        help="WebSocket price feed backend: 'coinbase' (wss://ws-feed.exchange.coinbase.com) or 'binance_us' (wss://stream.binance.us:9443)")
    parser.add_argument("--refresh", type=float, default=2,  help="Refresh rate in Hz")
    parser.add_argument("--window",  type=int, default=300,  help="Chart window (seconds)")
    parser.add_argument("--chart-w", type=int, default=_CHART_W, dest="chart_w",
                        help="Chart width in characters")
    parser.add_argument("--chart-h", type=int, default=_CHART_H, dest="chart_h",
                        help="Chart height in rows")
    parser.add_argument("--state-file", default="bot_state.json", dest="state_file",
                        help="Path to bot_state.json")
    parser.add_argument("--log-file", default="logs/btc_updown_bot.log", dest="log_file",
                        help="Path to bot log file")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config.yaml to display in the Config panel")
    args = parser.parse_args()

    # Interactive config picker (skipped when --config was explicitly set
    # to something other than the default, or when not a TTY)
    _default_config = "config/config.yaml"
    if args.config == _default_config:
        args.config = _pick_config(args.config)

    # Set up multi-asset slug prefix from --asset flag
    global _slug_prefix
    asset = args.asset.upper()
    _slug_prefix = f"{asset.lower()}-updown-5m"

    # Suppress feed internal logs so they don't interfere with the TUI
    logging.getLogger("btc_feed").setLevel(logging.WARNING)
    logging.getLogger("crypto_feed").setLevel(logging.WARNING)
    logging.getLogger("chainlink_feed").setLevel(logging.WARNING)

    # Resolve default symbol per exchange if not explicitly provided
    _sym_maps = {"coinbase": CryptoPriceFeed.COINBASE_SYMBOLS, "binance_us": CryptoPriceFeed.BINANCE_SYMBOLS}
    symbol = args.symbol or _sym_maps.get(args.exchange, {}).get(asset, f"{asset}-USD")

    console = Console()
    console.print(f"\n[bold]{asset} Feed Monitor[/bold] — connecting to [cyan]{symbol}[/cyan] via [cyan]{args.exchange}[/cyan]…")
    console.print("[dim]Starting WebSocket feed…[/dim]\n")

    feed = BtcPriceFeed(symbol=symbol, exchange=args.exchange)
    feed.start()
    chainlink_feed = ChainlinkFeed(symbol="btc/usd")
    chainlink_feed.start()

    # Wait for the first tick so the dashboard never opens on a blank Book panel.
    # Coinbase typically delivers within 1–2s; cap at 6s to handle slow networks.
    _deadline = time.time() + 6.0
    while time.time() < _deadline and feed.get_latest_book() is None:
        time.sleep(0.1)
    if feed.get_latest_book() is None:
        console.print("[bold yellow]⚠ Feed not yet connected — dashboard will show 'Connecting…' until data arrives.[/bold yellow]")

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
                try:
                    panel = _build_panel(
                        feed,
                        chainlink_feed,
                        args.window,
                        args.chart_w,
                        args.chart_h,
                        start_time,
                        args.state_file,
                        args.log_file,
                        args.config,
                    )
                except Exception:
                    err_text = Text(traceback.format_exc(), style="bold red")
                    panel = Panel(err_text, title="[bold red]Dashboard Error[/bold red]", border_style="red")
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
