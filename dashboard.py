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
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.utils.btc_feed import BtcPriceFeed


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
    global _market_cache, _market_cache_ts, _market_cache_slot

    now = time.time()
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
        events = resp.json()
        for event in (events if isinstance(events, list) else [events]):
            for m in event.get("markets", []):
                tids = json.loads(m.get("clobTokenIds", "[]"))
                outcomes = json.loads(m.get("outcomes", "[]"))
                if len(tids) >= 2:
                    volume = float(m.get("volume", 0) or 0)
                    _market_cache = {
                        "up_token": tids[0],
                        "down_token": tids[1],
                        "outcomes": outcomes,
                        "volume": volume,
                        "title": m.get("question", ""),
                    }
                    _market_cache_ts = now
                    _market_cache_slot = current_slot
                    return _market_cache
    except Exception:
        pass
    return _market_cache  # return stale on error


def _fetch_clob_book(token_id: str) -> Optional[Dict[str, Any]]:
    """Fetch order book from CLOB for a token. Cached 3s."""
    global _book_cache, _book_cache_ts

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

        # Also fetch midpoint for the "Last" display
        try:
            mr = requests.get(
                "https://clob.polymarket.com/midpoint",
                params={"token_id": token_id},
                timeout=3,
            )
            mr.raise_for_status()
            data["_midpoint"] = float(mr.json().get("mid", 0))
        except Exception:
            data["_midpoint"] = None

        _book_cache = data
        _book_cache_ts = now
        return _book_cache
    except Exception:
        return _book_cache  # stale on error


_OB_LEVELS = 5  # number of price levels to show per side


def _build_order_book_panel() -> Panel:
    """Build a Polymarket-style order book panel (asks on top, bids below)."""
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
        spread_cents = int(round(spread * 100))
        last_str = f"{int(round(midpoint * 100))}\u00a2" if midpoint else "–"
        tbl.add_row(
            Text("", style="dim"),
            Text(f"Last: {last_str}", style="bold yellow"),
            Text("", style="dim"),
            Text(f"Spread: {spread_cents}\u00a2", style="dim"),
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


# ── Bot state & positions helpers ─────────────────────────────────────────────

def _load_bot_state(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Read and parse bot_state.json. Returns (data, file_mtime) or (None, None)."""
    try:
        mtime = os.path.getmtime(path)
        with open(path, "r") as f:
            data = json.load(f)
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

    tbl = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    tbl.add_column("Side", style="bold", no_wrap=True)
    tbl.add_column("Size", justify="right", no_wrap=True)
    tbl.add_column("AvgCost", justify="right", no_wrap=True)
    tbl.add_column("uPnL", justify="right", no_wrap=True)

    for tid, inv in active.items():
        pos_size = inv.get("position", 0)
        avg_cost = inv.get("avg_cost", 0)

        # Determine label from live data or truncated token id
        live = live_lookup.get(tid)
        if live and live.get("outcome"):
            label = live["outcome"]
        else:
            label = tid[:8] + "…" if len(tid) > 8 else tid

        # Unrealized PnL from live data
        if live and "cur_price" in live:
            cur = float(live["cur_price"])
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


def _build_pnl_panel(bot_state: Optional[dict]) -> Panel:
    """Build the PnL panel from bot state."""
    if bot_state is None:
        return Panel(
            Text("No state data", style="dim italic"),
            title="[dim]PnL[/dim]",
            border_style="dim",
        )

    daily_pnl = bot_state.get("daily_realized_pnl", 0.0)
    reset_date = bot_state.get("daily_reset_date", "–")

    pnl_style = "green" if daily_pnl >= 0 else "red"

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=8)
    tbl.add_column(justify="right")
    tbl.add_row("Daily", Text(f"{daily_pnl:+.4f}", style=f"bold {pnl_style}"))
    tbl.add_row("Date", Text(str(reset_date), style="dim"))

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
    orders = bot_state.get("active_order_ids", [])
    order_count = len(orders) if isinstance(orders, list) else 0

    # Liveness: RUNNING if state file updated within last 10 minutes
    if file_mtime is not None:
        age_min = (time.time() - file_mtime) / 60.0
        if age_min < 10:
            status = Text("RUNNING", style="bold green")
        else:
            status = Text(f"STOPPED ({age_min:.0f}m ago)", style="bold red")
    else:
        status = Text("UNKNOWN", style="dim")

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=8)
    tbl.add_column(justify="right")
    tbl.add_row("Cycle", Text(str(cycle), style="white"))
    tbl.add_row("Orders", Text(str(order_count), style="white"))
    tbl.add_row("Bot", status)

    return Panel(tbl, title="[dim]Bot Status[/dim]", border_style="dim")


# ── Market Cycle panel ────────────────────────────────────────────────────────

def _parse_reference_price(question: str) -> Optional[float]:
    """Extract strike price from a Polymarket question like 'Will BTC > $84,500 ...'."""
    m = re.search(r'\$([0-9,]+(?:\.[0-9]+)?)', question)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _build_market_cycle_panel(feed) -> Panel:
    """Show current 5-min slot info: countdown, strike price, BTC delta."""
    market = _discover_market()

    now = time.time()
    current_slot = int(math.floor(now / 300) * 300)
    slot_end = current_slot + 300
    remaining = max(0.0, slot_end - now)

    slot_str = datetime.utcfromtimestamp(current_slot).strftime("%H:%M UTC")
    rem_m = int(remaining) // 60
    rem_s = int(remaining) % 60
    rem_str = f"{rem_m:02d}:{rem_s:02d}"

    if remaining <= 30:
        rem_style = "bold red"
    elif remaining <= 90:
        rem_style = "bold yellow"
    else:
        rem_style = "bold green"

    # Strike price from question text
    strike: Optional[float] = None
    if market:
        strike = _parse_reference_price(market.get("title", ""))

    # Current BTC price from feed
    book = feed.get_latest_book()
    btc_now = book.mid if book is not None else None

    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    tbl.add_column(style="dim", width=10)
    tbl.add_column(justify="right")

    tbl.add_row("Slot", Text(slot_str, style="dim"))
    tbl.add_row("Remaining", Text(rem_str, style=rem_style))

    if strike is not None:
        tbl.add_row("Strike", Text(f"${strike:,.0f}", style="cyan"))
    else:
        tbl.add_row("Strike", Text("–", style="dim"))

    if btc_now is not None:
        tbl.add_row("BTC Now", Text(f"${btc_now:,.2f}", style="bold yellow"))
        if strike is not None:
            delta = btc_now - strike
            pct = delta / strike * 100 if strike > 0 else 0
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


# ── Dashboard builder ─────────────────────────────────────────────────────────

def _build_panel(
    feed: BtcPriceFeed,
    window_s: int,
    chart_w: int,
    chart_h: int,
    start_time: float,
    state_file: str = "bot_state.json",
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
    uptime = int(time.time() - start_time)
    uptime_str = f"{uptime // 60}m {uptime % 60}s"

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
        Text(f"Uptime: {uptime_str}", style="dim"),
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

    # ── Polymarket order book (middle row) ────────────────────────────────
    order_book_panel = _build_order_book_panel()

    # ── Bot state panels ──────────────────────────────────────────────────
    bot_state, file_mtime = _load_bot_state(state_file)
    live_positions = _fetch_live_positions()

    positions_panel = _build_positions_panel(bot_state, live_positions)
    pnl_panel = _build_pnl_panel(bot_state)
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

    # ── Market Cycle + Strategy panels (bottom row) ───────────────────────
    market_cycle_panel = _build_market_cycle_panel(feed)
    strategy_panel = _build_strategy_panel(bot_state)

    bottom_row = Table.grid(expand=True)
    bottom_row.add_column(ratio=1)
    bottom_row.add_column(ratio=1)
    bottom_row.add_row(market_cycle_panel, strategy_panel)

    # ── Footer ────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    footer = Text(
        f"  Updated: {now_str}   Ctrl-C to exit",
        style="dim",
    )

    # ── Assemble ──────────────────────────────────────────────────────────
    outer = Table.grid(expand=True)
    outer.add_column()
    outer.add_row(status_bar)
    outer.add_row(body)
    outer.add_row(middle_row)
    outer.add_row(bottom_row)
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
    args = parser.parse_args()

    # Suppress feed internal logs so they don't interfere with the TUI
    logging.getLogger("btc_feed").setLevel(logging.WARNING)

    console = Console()
    console.print(f"\n[bold]BTC Feed Monitor[/bold] — connecting to [cyan]{args.symbol}[/cyan]…")
    console.print("[dim]Starting WebSocket feed…[/dim]\n")

    feed = BtcPriceFeed(symbol=args.symbol)
    feed.start()
    start_time = time.time()

    interval = 1.0 / max(args.refresh, 0.1)

    try:
        with Live(
            console=console,
            screen=True,
            refresh_per_second=args.refresh,
        ) as live:
            while True:
                panel = _build_panel(feed, args.window, args.chart_w, args.chart_h, start_time, args.state_file)
                live.update(panel)
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        console.print("\n[dim]Feed stopped.[/dim]")


if __name__ == "__main__":
    main()
