"""Build local training datasets from probability tick logs and resolved slots."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.api.types import OrderBook, OrderBookEntry
from src.models import DEFAULT_FEATURE_VALUES, build_live_features, parse_strike_price


# 4 hours — see src/backtest/s3_snapshot_loader.py for rationale. Must match
# the live strategy's BTC window so training and inference see the same
# long-horizon features.
DEFAULT_BTC_WINDOW_SECONDS = 14400
_BOOK_SIZE = 100.0


def load_market_history(csv_path: str) -> pd.DataFrame:
    """Load resolved BTC Up/Down markets keyed by slot timestamp."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    df["slot_ts"] = pd.to_numeric(df["slot_ts"], errors="coerce")
    df = df.dropna(subset=["slot_ts"]).copy()
    df["slot_ts"] = df["slot_ts"].astype(int)
    df["question"] = df["question"].fillna("").astype(str) if "question" in df.columns else ""
    df["outcome"] = df["outcome"].fillna("").astype(str) if "outcome" in df.columns else ""

    if "strike_price" in df.columns:
        df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    else:
        df["strike_price"] = pd.NA

    parsed = df["question"].map(parse_strike_price)
    df["strike_price"] = df["strike_price"].fillna(parsed)
    df["label"] = df["outcome"].map(lambda value: 1 if str(value).strip().lower() == "up" else 0)
    return df.drop_duplicates(subset=["slot_ts"], keep="last").sort_values("slot_ts").reset_index(drop=True)


def load_probability_ticks(jsonl_path: str) -> pd.DataFrame:
    """Load the probability tick JSONL log written by the bot."""
    rows: List[Dict[str, object]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["ts", "slot_ts", "yes_bid", "yes_ask", "no_bid", "no_ask", "yes_mid", "no_mid", "yes_spread", "no_spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ts", "slot_ts"]).copy()
    df["slot_ts"] = df["slot_ts"].astype(int)

    df["yes_mid"] = df.get("yes_mid", pd.Series(dtype=float))
    df["no_mid"] = df.get("no_mid", pd.Series(dtype=float))
    df["yes_mid"] = df["yes_mid"].fillna(_mid_from_bid_ask(df.get("yes_bid"), df.get("yes_ask")))
    df["no_mid"] = df["no_mid"].fillna(_mid_from_bid_ask(df.get("no_bid"), df.get("no_ask")))
    df["yes_spread"] = df.get("yes_spread", pd.Series(dtype=float)).fillna(
        (df.get("yes_ask") - df.get("yes_bid")).astype(float)
    )
    df["no_spread"] = df.get("no_spread", pd.Series(dtype=float)).fillna(
        (df.get("no_ask") - df.get("no_bid")).astype(float)
    )
    return df.sort_values(["slot_ts", "ts"]).reset_index(drop=True)


def load_btc_prices(path: str) -> pd.DataFrame:
    """Load local BTC history from CSV or JSONL.

    Supported schemas are intentionally loose:
    - timestamp columns: ts | timestamp | time
    - price columns: price | mid | btc_mid | close
    - or bid+ask, where price = (bid + ask) / 2
    """
    if path.lower().endswith(".jsonl"):
        rows: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)

    if df.empty:
        return df

    ts_col = _first_existing(df.columns, ("ts", "timestamp", "time"))
    if ts_col is None:
        raise ValueError("BTC file must contain one of: ts, timestamp, time")

    price_col = _first_existing(df.columns, ("price", "mid", "btc_mid", "close"))
    if price_col is not None:
        price = pd.to_numeric(df[price_col], errors="coerce")
    elif "bid" in df.columns and "ask" in df.columns:
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        price = (bid + ask) / 2.0
    else:
        raise ValueError("BTC file must contain price-like columns or bid+ask")

    out = pd.DataFrame(
        {
            "ts": pd.to_numeric(df[ts_col], errors="coerce"),
            "price": pd.to_numeric(price, errors="coerce"),
        }
    )
    out = out.dropna(subset=["ts", "price"]).copy()
    out = out[out["price"] > 0].sort_values("ts").reset_index(drop=True)
    return out


def build_snapshot_dataset(
    markets_df: pd.DataFrame,
    prob_ticks_df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    btc_window_seconds: int = DEFAULT_BTC_WINDOW_SECONDS,
    cost_per_share: float = 0.0,
) -> pd.DataFrame:
    """Build a training/backtest frame from local market and tick logs."""
    if markets_df.empty or prob_ticks_df.empty:
        return pd.DataFrame()

    markets = markets_df.copy()
    markets["slot_expiry_ts"] = markets["slot_ts"] + 300
    markets_by_slot = {int(row.slot_ts): row for row in markets.itertuples(index=False)}

    btc_ts: Sequence[float] = ()
    btc_price: Sequence[float] = ()
    if btc_df is not None and not btc_df.empty:
        btc_ts = btc_df["ts"].astype(float).to_numpy()
        btc_price = btc_df["price"].astype(float).to_numpy()

    rows: List[Dict[str, object]] = []
    for slot_ts, group in prob_ticks_df.groupby("slot_ts", sort=True):
        market = markets_by_slot.get(int(slot_ts))
        if market is None:
            continue
        if str(market.outcome).strip().lower() not in {"up", "down"}:
            continue

        slot_ticks = group.sort_values("ts").reset_index(drop=True)
        for idx, tick in slot_ticks.iterrows():
            snapshot_ts = float(tick["ts"])
            if snapshot_ts < float(slot_ts):
                continue

            strike_price = market.strike_price
            if pd.isna(strike_price) and len(btc_ts) > 0:
                strike_price = _asof_price(btc_ts, btc_price, float(slot_ts))

            features, feature_status = _build_feature_row(
                tick=tick,
                slot_ticks=slot_ticks.iloc[: idx + 1],
                snapshot_ts=snapshot_ts,
                slot_ts=float(slot_ts),
                question=str(market.question),
                strike_price=None if pd.isna(strike_price) else float(strike_price),
                btc_ts=btc_ts,
                btc_price=btc_price,
                btc_window_seconds=btc_window_seconds,
            )

            sim_yes_ask = _safe_fill_price(tick.get("yes_ask"), tick.get("yes_mid"), tick.get("yes_bid"))
            sim_no_ask = _safe_fill_price(tick.get("no_ask"), tick.get("no_mid"), tick.get("no_bid"))

            label = int(market.label)
            edge_yes = label - sim_yes_ask - cost_per_share
            edge_no = (1 - label) - sim_no_ask - cost_per_share

            row: Dict[str, object] = {
                "snapshot_ts": snapshot_ts,
                "slot_ts": int(slot_ts),
                "slot_expiry_ts": int(slot_ts) + 300,
                "question": str(market.question),
                "outcome": str(market.outcome),
                "label": label,
                "feature_status": feature_status,
                "sim_yes_ask": sim_yes_ask,
                "sim_no_ask": sim_no_ask,
                "edge_yes": edge_yes,
                "edge_no": edge_no,
                "profitable_yes": int(edge_yes > 0),
                "profitable_no": int(edge_no > 0),
            }
            row.update(features)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("snapshot_ts").reset_index(drop=True)


def build_btc_decision_dataset(
    markets_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    row_interval_sec: int = 15,
    window_seconds: int = DEFAULT_BTC_WINDOW_SECONDS,
) -> pd.DataFrame:
    """Build a training dataset by expanding each contract into decision rows.

    Step 3: For each resolved contract, create one row every *row_interval_sec*
    seconds within its 5-minute window.
    Step 4: Attach BTC features (spot, returns, vol, MA, RSI) using only data
    available up to each row's timestamp.
    Step 5: Add normalised features (dist_to_strike, ma_12_gap).

    Args:
        markets_df: Resolved market history from :func:`load_market_history`.
            Required columns: slot_ts, outcome.  Optional: strike_price.
        btc_df: BTC price series from :func:`load_btc_prices`.
            Required columns: ts, price.
        row_interval_sec: Seconds between decision rows (default 15).
        window_seconds: Lookback window for BTC features (default 300).

    Returns:
        DataFrame with one row per (contract, decision timestamp).
    """
    if markets_df.empty or btc_df.empty:
        return pd.DataFrame()

    btc_ts = btc_df["ts"].astype(float).to_numpy()
    btc_price = btc_df["price"].astype(float).to_numpy()

    rows: List[Dict[str, object]] = []
    for market in markets_df.itertuples(index=False):
        slot_ts = int(market.slot_ts)
        expiry_ts = slot_ts + 300
        outcome = str(market.outcome).strip().lower()
        if outcome not in {"up", "down"}:
            continue
        target_up = 1 if outcome == "up" else 0

        strike = getattr(market, "strike_price", None)
        if strike is not None and (pd.isna(strike) or float(strike) <= 0):
            strike = None
        if strike is None:
            strike_val = _asof_price(btc_ts, btc_price, float(slot_ts))
        else:
            strike_val = float(strike)
        if strike_val is None:
            continue

        # Step 3 — expand into decision timestamps
        t = slot_ts
        while t <= expiry_ts:
            tte = expiry_ts - t
            spot = _asof_price(btc_ts, btc_price, float(t))
            if spot is None:
                t += row_interval_sec
                continue

            # Step 4 — BTC features
            ret_15s = _ts_return(btc_ts, btc_price, float(t), 15)
            ret_30s = _ts_return(btc_ts, btc_price, float(t), 30)
            ret_60s = _ts_return(btc_ts, btc_price, float(t), 60)
            vol_60s = _rolling_vol(btc_ts, btc_price, float(t), 60)
            ma_12 = _rolling_mean(btc_ts, btc_price, float(t), 12)
            rsi_14 = _rsi(btc_ts, btc_price, float(t), 14)

            # Step 5 — normalised features
            dist_to_strike = (spot - strike_val) / strike_val if strike_val > 0 else 0.0
            ma_12_gap = (spot - ma_12) / ma_12 if ma_12 > 0 else 0.0

            rows.append({
                "contract_id": slot_ts,
                "timestamp": t,
                "expiry_ts": expiry_ts,
                "time_to_expiry_sec": tte,
                "reference_price": strike_val,
                "target_up": target_up,
                "btc_spot": spot,
                "ret_15s": ret_15s,
                "ret_30s": ret_30s,
                "ret_60s": ret_60s,
                "rolling_vol_60s": vol_60s,
                "ma_12": ma_12,
                "rsi_14": rsi_14,
                "dist_to_strike": dist_to_strike,
                "ma_12_gap": ma_12_gap,
            })
            t += row_interval_sec

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["contract_id", "timestamp"]).reset_index(drop=True)


# -- helpers for build_btc_decision_dataset ------------------------------------

def _ts_return(
    btc_ts: Sequence[float], btc_price: Sequence[float], now: float, lookback_sec: int
) -> float:
    """Simple return over *lookback_sec* seconds ending at *now*."""
    current = _asof_price(btc_ts, btc_price, now)
    prev = _asof_price(btc_ts, btc_price, now - lookback_sec)
    if current is None or prev is None or prev <= 0:
        return 0.0
    return (current - prev) / prev


def _rolling_vol(
    btc_ts: Sequence[float], btc_price: Sequence[float], now: float, lookback_sec: int
) -> float:
    """Standard deviation of 1-second returns in the lookback window."""
    end_idx = _bisect_right_ts(btc_ts, now)
    start_idx = _bisect_right_ts(btc_ts, now - lookback_sec)
    if end_idx - start_idx < 2:
        return 0.0
    arr = np.array(btc_price[start_idx:end_idx], dtype=float)
    rets = np.diff(arr) / arr[:-1]
    return float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0


def _rolling_mean(
    btc_ts: Sequence[float], btc_price: Sequence[float], now: float, n: int
) -> float:
    """Mean of the last *n* BTC prices up to *now*."""
    idx = _bisect_right_ts(btc_ts, now)
    if idx == 0:
        return 0.0
    start = max(0, idx - n)
    segment = btc_price[start:idx]
    if len(segment) == 0:
        return 0.0
    return float(sum(segment)) / len(segment)


def _rsi(
    btc_ts: Sequence[float], btc_price: Sequence[float], now: float, period: int
) -> float:
    """RSI computed from the last *period+1* price ticks up to *now*."""
    idx = _bisect_right_ts(btc_ts, now)
    need = period + 1
    if idx < need:
        return 50.0  # neutral default
    segment = [float(btc_price[i]) for i in range(idx - need, idx)]
    gains = 0.0
    losses = 0.0
    for i in range(1, len(segment)):
        delta = segment[i] - segment[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _bisect_right_ts(btc_ts: Sequence[float], target: float) -> int:
    """Return insertion point for *target* in sorted *btc_ts* (bisect_right)."""
    lo, hi = 0, len(btc_ts)
    while lo < hi:
        mid = (lo + hi) // 2
        if btc_ts[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def save_snapshot_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Write the built dataset to CSV or Parquet based on extension."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def _build_feature_row(
    tick: pd.Series,
    slot_ticks: pd.DataFrame,
    snapshot_ts: float,
    slot_ts: float,
    question: str,
    strike_price: Optional[float],
    btc_ts: Sequence[float],
    btc_price: Sequence[float],
    btc_window_seconds: int,
) -> Tuple[Dict[str, float], str]:
    slot_expiry_ts = slot_ts + 300.0
    yes_book = _build_book("yes", tick.get("yes_bid"), tick.get("yes_ask"), tick.get("yes_mid"))
    no_book = _build_book("no", tick.get("no_bid"), tick.get("no_ask"), tick.get("no_mid"))
    yes_history = [(float(row.ts), float(row.yes_mid)) for row in slot_ticks.itertuples(index=False) if pd.notna(row.yes_mid)]
    no_history = [(float(row.ts), float(row.no_mid)) for row in slot_ticks.itertuples(index=False) if pd.notna(row.no_mid)]

    btc_window = _btc_window(btc_ts, btc_price, snapshot_ts, btc_window_seconds)
    if strike_price is not None and len(btc_window) >= 2:
        snapshot = {
            "btc_prices": btc_window,
            "yes_book": yes_book,
            "no_book": no_book,
            "yes_history": yes_history,
            "no_history": no_history,
            "question": question,
            "strike_price": strike_price,
            "slot_expiry_ts": slot_expiry_ts,
            "now_ts": snapshot_ts,
        }
        built = build_live_features(snapshot)
        return built.features, built.status

    features = dict(DEFAULT_FEATURE_VALUES)
    _fill_market_features(features, tick, slot_ticks, snapshot_ts)
    features["seconds_to_expiry"] = max(0.0, slot_expiry_ts - snapshot_ts)

    current_btc = _asof_price(btc_ts, btc_price, snapshot_ts) if len(btc_ts) > 0 else None
    if current_btc is not None:
        features["btc_mid"] = current_btc
    if strike_price is not None:
        features["strike_price"] = float(strike_price)
        if current_btc is not None and strike_price > 0:
            features["moneyness"] = (current_btc - strike_price) / strike_price
            features["distance_to_strike_bps"] = features["moneyness"] * 10_000.0

    if strike_price is None and current_btc is None:
        status = "missing_btc_history_and_strike"
    elif strike_price is None:
        status = "missing_strike"
    elif current_btc is None:
        status = "missing_btc_history"
    else:
        status = "insufficient_btc_history"
    return features, status


def _fill_market_features(
    features: Dict[str, float],
    tick: pd.Series,
    slot_ticks: pd.DataFrame,
    snapshot_ts: float,
) -> None:
    yes_bid = _to_float(tick.get("yes_bid"))
    yes_ask = _to_float(tick.get("yes_ask"))
    yes_mid = _to_float(tick.get("yes_mid")) or _first_positive(yes_bid, yes_ask, 0.5)
    yes_spread = _positive_or_default(_to_float(tick.get("yes_spread")), max(0.0, yes_ask - yes_bid))

    no_bid = _to_float(tick.get("no_bid"))
    no_ask = _to_float(tick.get("no_ask"))
    no_mid = _to_float(tick.get("no_mid")) or _first_positive(no_bid, no_ask, 0.5)
    no_spread = _positive_or_default(_to_float(tick.get("no_spread")), max(0.0, no_ask - no_bid))

    features["yes_bid"] = yes_bid
    features["yes_ask"] = yes_ask
    features["yes_mid"] = yes_mid
    features["yes_spread"] = yes_spread
    features["yes_spread_pct"] = yes_spread / yes_mid if yes_mid > 0 else 0.0
    features["yes_book_imbalance"] = 0.0
    features["yes_ret_30s"] = _history_return(slot_ticks, snapshot_ts, "yes_mid")

    features["no_bid"] = no_bid
    features["no_ask"] = no_ask
    features["no_mid"] = no_mid
    features["no_spread"] = no_spread
    features["no_spread_pct"] = no_spread / no_mid if no_mid > 0 else 0.0
    features["no_book_imbalance"] = 0.0
    features["no_ret_30s"] = _history_return(slot_ticks, snapshot_ts, "no_mid")


def _btc_window(
    btc_ts: Sequence[float],
    btc_price: Sequence[float],
    snapshot_ts: float,
    window_seconds: int,
) -> List[Tuple[float, float]]:
    if len(btc_ts) == 0:
        return []
    left = 0
    right = len(btc_ts)
    while left < right and btc_ts[left] < snapshot_ts - window_seconds:
        left += 1
    while right > left and btc_ts[right - 1] > snapshot_ts:
        right -= 1
    return [(float(btc_ts[idx]), float(btc_price[idx])) for idx in range(left, right)]


def _asof_price(btc_ts: Sequence[float], btc_price: Sequence[float], target_ts: float) -> Optional[float]:
    if len(btc_ts) == 0:
        return None
    left = 0
    right = len(btc_ts)
    while left < right:
        mid = (left + right) // 2
        if btc_ts[mid] <= target_ts:
            left = mid + 1
        else:
            right = mid
    idx = left - 1
    if idx < 0:
        return None
    return float(btc_price[idx])


def _build_book(token_id: str, bid_raw: object, ask_raw: object, mid_raw: object) -> OrderBook:
    bid = _to_float(bid_raw)
    ask = _to_float(ask_raw)
    mid = _to_float(mid_raw)
    if bid <= 0 and mid is not None:
        bid = mid
    if ask <= 0 and mid is not None:
        ask = mid
    if bid <= 0 and ask > 0:
        bid = ask
    if ask <= 0 and bid > 0:
        ask = bid
    if bid <= 0:
        bid = 0.5
    if ask <= 0:
        ask = 0.5
    return OrderBook(
        market_id="",
        token_id=token_id,
        bids=[OrderBookEntry(price=bid, size=_BOOK_SIZE)],
        asks=[OrderBookEntry(price=ask, size=_BOOK_SIZE)],
        tick_size=0.001,
    )


def _history_return(slot_ticks: pd.DataFrame, snapshot_ts: float, column: str) -> float:
    current = _to_float(slot_ticks.iloc[-1].get(column))
    if current <= 0:
        return 0.0
    target = snapshot_ts - 30.0
    previous = None
    for row in slot_ticks.itertuples(index=False):
        if float(row.ts) <= target:
            value = _to_float(getattr(row, column))
            if value > 0:
                previous = value
        else:
            break
    if previous is None or previous <= 0:
        return 0.0
    return (current - previous) / previous


def _safe_fill_price(*values: object) -> float:
    for value in values:
        v = _to_float(value)
        if v > 0:
            return v
    return 0.5


def _mid_from_bid_ask(bid: Optional[pd.Series], ask: Optional[pd.Series]) -> pd.Series:
    if bid is None or ask is None:
        return pd.Series(dtype=float)
    bid_num = pd.to_numeric(bid, errors="coerce")
    ask_num = pd.to_numeric(ask, errors="coerce")
    return (bid_num + ask_num) / 2.0


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def _positive_or_default(value: Optional[float], fallback: float) -> float:
    if value is not None and value >= 0:
        return float(value)
    return float(fallback)


def _first_positive(*values: object) -> float:
    for value in values:
        v = _to_float(value)
        if v > 0:
            return v
    return 0.0


def _to_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(out):
        return 0.0
    return out
