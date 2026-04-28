"""Animated per-trade replay as a self-contained HTML.

For each settled trade in the window, builds one Plotly frame showing:
  • BTC price trajectory during the slot (~8 points, every ~30s)
  • Strike level (horizontal line)
  • Entry marker (color = side, vertical position = entry price)
  • YES market price trajectory (right axis)
  • Model probability (p_hat) at decision time
  • Outcome resolution + realized PnL

Frames are stitched into one figure with a play button and a slider, so
the user can scrub through trades in order or watch the full session
auto-play.

Usage::

    python scripts/replay_visual.py \\
      --decision-log data/2026-04-26/decision_log_20260426T055228Z.jsonl \\
      --bot-log logs/btc_updown_bot.log \\
      --since "2026-04-26 00:00:00" \\
      --output data/replay_04_26_27.html

Open the HTML in any browser. Press ▶ to auto-play, drag the slider to
jump to a specific trade.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.diagnose_run import _parse_settlements


_BTC_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
_BTC_BUCKET = "k-polymarket-data"


def _fetch_btc_1s(date: str, hour: str, s3_client) -> pd.DataFrame:
    """Download (or read from cache) one hour of 1s BTC from S3.

    The bot writes 1Hz BTC into ``data/<date>/<HH>/btc_live_1s.csv``.
    Caching across hours keeps the replay-build cost down to one fetch
    per (date, hour) pair, even when many slots span the same hour.
    """
    key = (date, hour)
    if key in _BTC_CACHE:
        return _BTC_CACHE[key]
    s3_key = f"data/{date}/{hour}/btc_live_1s.csv"
    try:
        obj = s3_client.get_object(Bucket=_BTC_BUCKET, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception:
        df = pd.DataFrame(columns=["timestamp", "close"])
    _BTC_CACHE[key] = df
    return df


def _btc_trajectory_for_slot(
    slot_open: float, slot_close: float, s3_client,
) -> Tuple[List[float], List[float]]:
    """Return (xs_seconds_from_open, ys_btc_close) covering the slot.

    Uses the real 1Hz BTC tick stream from S3 instead of the
    decision-log's f_btc_mid values, because the latter freezes after a
    fill (the strategy short-circuits on `in_position` before
    recomputing features). Empty result if S3 is unreachable.
    """
    open_dt = datetime.fromtimestamp(slot_open, tz=timezone.utc)
    close_dt = datetime.fromtimestamp(slot_close, tz=timezone.utc)
    hours_to_pull = {(open_dt.strftime("%Y-%m-%d"), open_dt.strftime("%H"))}
    if close_dt.strftime("%Y-%m-%d %H") != open_dt.strftime("%Y-%m-%d %H"):
        hours_to_pull.add((close_dt.strftime("%Y-%m-%d"), close_dt.strftime("%H")))

    frames = [_fetch_btc_1s(d, h, s3_client) for d, h in hours_to_pull]
    nonempty = [f for f in frames if not f.empty]
    if not nonempty:
        return [], []
    df = pd.concat(nonempty, ignore_index=True)
    if "timestamp" not in df.columns:
        return [], []
    price_col = "close" if "close" in df.columns else df.select_dtypes("number").columns[-1]
    df = df[(df["timestamp"] >= slot_open) & (df["timestamp"] <= slot_close)] \
        .sort_values("timestamp").reset_index(drop=True)
    xs = (df["timestamp"] - slot_open).tolist()
    ys = df[price_col].astype(float).tolist()
    return xs, ys


def _load_decisions_by_slot(path: Path, since_ts: float) -> Dict[int, List[dict]]:
    """Group every decision-log record by slot_ts so we can rebuild
    each slot's intra-cycle trajectory."""
    by_slot: Dict[int, List[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ts", 0) < since_ts:
                continue
            slot_ts = r.get("slot_ts")
            if slot_ts is None:
                continue
            by_slot[int(slot_ts)].append(r)
    for slot_ts, recs in by_slot.items():
        recs.sort(key=lambda r: r.get("ts", 0))
    return by_slot


def _trade_records(slot_records: List[dict]) -> List[dict]:
    """The intra-slot records where the strategy actually fired."""
    return [r for r in slot_records if r.get("action")]


def _attach_settlements_by_match(
    settlements: List[dict],
    trades_by_slot: Dict[int, List[dict]],
) -> Dict[int, dict]:
    """Match settlements to slot_ts via the trade-entry tuple (side, price).

    The derive-slot-from-settlement-ts approach fails when slots without
    trades sit between traded slots — settlement.ts // 300 doesn't always
    align to the slot the trade opened in. Matching by side + entry price
    is what diagnose_run uses; it's robust to gaps.
    """
    settle_by_slot: Dict[int, dict] = {}
    used = set()  # avoid double-claiming a settlement
    for slot_ts, recs in trades_by_slot.items():
        # First record in this slot that actually has an action.
        first_trade = next((r for r in recs if r.get("action")), None)
        if first_trade is None:
            continue
        side = first_trade["action"].get("side")
        price = float(first_trade["action"].get("price"))
        # Find a settlement on the same side, similar price, AFTER the trade.
        cands = [
            (i, s) for i, s in enumerate(settlements)
            if i not in used
            and s["side"] == side
            and abs(s["entry"] - price) < 0.001
            and s["ts"] >= first_trade["ts"]
            and s["ts"] - first_trade["ts"] < 700  # within ~12 min
        ]
        if cands:
            # Take the earliest matching settlement.
            cands.sort(key=lambda kv: kv[1]["ts"])
            i, s = cands[0]
            settle_by_slot[slot_ts] = s
            used.add(i)
    return settle_by_slot


def _build_frame(
    trade_idx: int,
    trade: dict,
    slot_recs: List[dict],
    settlement: Optional[dict],
    strike: float,
    s3_client=None,
) -> dict:
    """Build one Plotly frame's worth of trace data for a single trade."""
    slot_open = float(trade["slot_ts"])
    slot_close = slot_open + 300.0
    entry_ts = float(trade["ts"])

    # Real BTC trajectory from the S3 1Hz tick stream — does not freeze
    # after fill the way decision_log f_btc_mid does. Falls back to the
    # decision-log values if S3 isn't reachable.
    btc_xs: List[float] = []
    btc_y: List[float] = []
    if s3_client is not None:
        btc_xs, btc_y = _btc_trajectory_for_slot(slot_open, slot_close, s3_client)
    if not btc_xs:
        btc_xs = [r["ts"] - slot_open for r in slot_recs]
        btc_y = [r.get("f_btc_mid") for r in slot_recs]

    # YES market price: keep the decision_log values BUT only the
    # records up to and including the entry tick. After the fill,
    # `in_position` short-circuits the feature recompute and the
    # f_yes_mid value freezes. Drawing those would mislead.
    pre_entry = [r for r in slot_recs if r["ts"] <= entry_ts + 1]
    yes_xs = [r["ts"] - slot_open for r in pre_entry]
    yes_y = [r.get("f_yes_mid") for r in pre_entry]

    side = trade["action"].get("side")
    entry_x = entry_ts - slot_open
    # Pull entry BTC from the real trajectory (closest tick) rather than
    # the recorded f_btc_mid, which is fine here but stays consistent.
    entry_btc = trade.get("f_btc_mid")
    if btc_xs and btc_y:
        # Snap to the closest real tick around entry_x.
        idx = min(range(len(btc_xs)), key=lambda i: abs(btc_xs[i] - entry_x))
        entry_btc = btc_y[idx]
    entry_yes = trade["action"].get("price")
    p_hat = trade.get("prob_yes")

    win = settlement is not None and settlement.get("pnl", 0) > 0
    outcome = settlement.get("outcome") if settlement else "?"
    pnl = settlement.get("pnl") if settlement else None
    exit_btc = btc_y[-1] if btc_y else entry_btc

    # Entry marker color
    entry_color = "#26a69a" if side == "YES" else "#ef5350"
    outcome_color = "#26a69a" if win else "#ef5350"

    btc_trace = go.Scatter(
        x=btc_xs, y=btc_y, mode="lines", name="BTC mid (1Hz)",
        line=dict(color="#42a5f5", width=2),
        hovertemplate="t+%{x:.0f}s<br>BTC=$%{y:,.2f}<extra></extra>",
    )
    strike_trace = go.Scatter(
        x=[0, 300],
        y=[strike, strike],
        mode="lines", name="strike",
        line=dict(color="#ffa726", width=1, dash="dash"),
        hovertemplate=f"strike $%{{y:,.2f}}<extra></extra>",
    )
    entry_marker = go.Scatter(
        x=[entry_x], y=[entry_btc],
        mode="markers", name=f"ENTRY {side}",
        marker=dict(symbol="diamond", size=14, color=entry_color,
                    line=dict(color="white", width=1.5)),
        hovertemplate=(
            f"ENTRY {side}<br>"
            f"t+%{{x:.0f}}s<br>"
            f"yes_ask=%{{customdata[0]:.3f}}<br>"
            f"p_hat=%{{customdata[1]:.3f}}<extra></extra>"
        ),
        customdata=[[entry_yes, p_hat]],
    )
    # Vertical line at entry time so the user can see exactly when we
    # decided. Drawn from y_lo to y_hi of the BTC range.
    if btc_y:
        y_lo, y_hi = min(btc_y), max(btc_y)
        # pad 5% top/bottom so the line spans the visible area
        pad = (y_hi - y_lo) * 0.1 + 1.0
        entry_line = go.Scatter(
            x=[entry_x, entry_x], y=[y_lo - pad, y_hi + pad],
            mode="lines", name="entry tick",
            line=dict(color=entry_color, width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        )
    else:
        entry_line = go.Scatter(x=[], y=[], mode="lines", showlegend=False)

    yes_trace = go.Scatter(
        x=yes_xs, y=yes_y, mode="lines+markers",
        name="YES market (pre-entry)",
        line=dict(color="#ab47bc", width=1.5, dash="dot"),
        marker=dict(size=4),
        yaxis="y2",
        hovertemplate="t+%{x:.0f}s<br>YES=%{y:.3f}<extra></extra>",
    )
    p_hat_trace = go.Scatter(
        x=[entry_x], y=[p_hat],
        mode="markers", name="p_hat",
        marker=dict(symbol="star", size=12, color="#fff176",
                    line=dict(color="#f57f17", width=1)),
        yaxis="y2",
        hovertemplate=f"p_hat=%{{y:.3f}}<extra></extra>",
    )

    title = (
        f"Trade {trade_idx + 1} • slot {datetime.fromtimestamp(slot_open, tz=timezone.utc).strftime('%m-%d %H:%M')}Z • "
        f"<b style='color:{entry_color}'>{side}</b> @ {entry_yes:.3f} • "
        f"p_hat=<b>{p_hat:.3f}</b> • "
        f"outcome=<b style='color:{outcome_color}'>{outcome}</b>" +
        (f" • PnL <b style='color:{outcome_color}'>${pnl:+.2f}</b>" if pnl is not None else "")
    )

    return {
        "data": [btc_trace, strike_trace, entry_line, entry_marker,
                 yes_trace, p_hat_trace],
        "name": str(trade_idx),
        "layout": {"title": {"text": title}},
    }


def _parse_strike_from_question(q: str) -> Optional[float]:
    m = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)", q or "")
    return float(m.group(1).replace(",", "")) if m else None


def build_replay_figure(
    decision_log: Path,
    bot_log: Path,
    since_ts: float,
    use_s3_btc: bool = True,
) -> go.Figure:
    by_slot = _load_decisions_by_slot(decision_log, since_ts=since_ts)
    settlements = _parse_settlements(bot_log, since_ts=since_ts)
    # Group decisions by slot, keeping only those that contain a TRADE.
    trades_by_slot = {
        slot_ts: recs for slot_ts, recs in by_slot.items()
        if any(r.get("action") for r in recs)
    }
    settle_by_slot = _attach_settlements_by_match(settlements, trades_by_slot)

    # Real BTC ticks come from S3 1Hz CSVs. The decision-log's f_btc_mid
    # freezes after fill (in_position short-circuits feature recompute),
    # so without S3 the post-entry BTC line goes flat. With S3 we get a
    # 300-point trajectory per slot.
    s3_client = None
    if use_s3_btc:
        try:
            import boto3
            s3_client = boto3.client("s3", region_name="us-west-2")
        except Exception as exc:
            print(f"warning: S3 client unavailable ({exc}); "
                  "falling back to decision_log f_btc_mid (frozen post-fill).")

    # Build one frame per (settled) trade.
    frames: List[dict] = []
    for slot_ts in sorted(by_slot.keys()):
        recs = by_slot[slot_ts]
        trades = _trade_records(recs)
        if not trades:
            continue
        settlement = settle_by_slot.get(slot_ts)
        if settlement is None:
            continue
        strike = recs[0].get("strike") or recs[0].get("f_strike_price") or 0.0
        if not strike:
            # Try to derive from question text in any record
            for r in recs:
                s = _parse_strike_from_question(r.get("question", ""))
                if s:
                    strike = s
                    break
        if not strike:
            continue
        # Use the first BUY decision as "the trade" for this slot
        trade = trades[0]
        frame = _build_frame(
            trade_idx=len(frames), trade=trade,
            slot_recs=recs, settlement=settlement, strike=float(strike),
            s3_client=s3_client,
        )
        frames.append(frame)

    if not frames:
        raise SystemExit("No trades to visualize.")

    # Build the figure with the FIRST frame's traces as initial state.
    initial = frames[0]
    fig = make_subplots(
        rows=1, cols=1, specs=[[{"secondary_y": True}]],
    )
    for tr in initial["data"]:
        # Plotly expects yaxis="y2" via secondary_y arg in add_trace
        secondary = (tr.yaxis == "y2") if hasattr(tr, "yaxis") else False
        fig.add_trace(tr, secondary_y=secondary)

    # Frames
    plotly_frames = []
    for f in frames:
        plotly_frames.append(go.Frame(
            data=f["data"], name=f["name"], layout=f["layout"],
        ))
    fig.frames = plotly_frames

    # Layout: animation controls + initial title
    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Trade #", "visible": True},
        "pad": {"t": 50},
        "steps": [{
            "method": "animate",
            "label": str(i + 1),
            "args": [[str(i)], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        } for i in range(len(frames))],
    }]
    updatemenus = [{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.15, "xanchor": "left", "yanchor": "top",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 1500, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                }],
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                }],
            },
        ],
    }]

    fig.update_layout(
        title={"text": initial["layout"]["title"]["text"]},
        template="plotly_dark",
        height=620, width=1200,
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        sliders=sliders,
        updatemenus=updatemenus,
        margin=dict(l=60, r=60, t=80, b=120),
    )
    fig.update_xaxes(title_text="seconds into 5-min slot", range=[0, 300])
    fig.update_yaxes(title_text="BTC price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="market YES price / p_hat", secondary_y=True,
                     range=[0, 1])
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--bot-log", type=Path, required=True)
    parser.add_argument("--since", required=True,
                        help="UTC start, 'YYYY-MM-DD HH:MM:SS'.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HTML. Default: data/replay_<UTC ts>.html.",
    )
    args = parser.parse_args()

    since_ts = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S").timestamp()
    fig = build_replay_figure(args.decision_log, args.bot_log, since_ts)

    out = args.output or (
        _ROOT / "data" / f"replay_{int(datetime.now().timestamp())}.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", auto_play=False)
    print(f"wrote {out}")
    print(f"frames: {len(fig.frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
