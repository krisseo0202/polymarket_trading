"""Load collected BTC Up/Down snapshots from S3 into a training DataFrame.

Data layout (written by ``scripts/collect_snapshots.py`` +
``scripts/data_collector.py``):

    s3://<bucket>/data/<YYYY-MM-DD>/<HH>/snapshots_btc.jsonl   # per-5s snapshot
    s3://<bucket>/data/<YYYY-MM-DD>/<HH>/btc_live_1s.csv       # 1s OHLCV BTC
    s3://<bucket>/data/<YYYY-MM-DD>/<HH>/slot_key_map.csv      # slot → outcome

Snapshot records include full depth dicts (``yes_bids``/``yes_asks`` etc.) and
an outcome sentinel ``{"type": "outcome", "slot_ts": N, "outcome": "Up"|"Down"}``.

Parsing is split from S3 IO so unit tests can feed synthetic records without
network or boto3.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.api.types import OrderBook, OrderBookEntry
from src.models import build_live_features
from src.models.slot_path_state import SlotPathState


DEFAULT_BUCKET = "k-polymarket-data"
DEFAULT_PREFIX = "data"
SNAPSHOT_FILENAME = "snapshots_btc.jsonl"
BTC_FILENAME = "btc_live_1s.csv"
SLOT_MAP_FILENAME = "slot_key_map.csv"

_SLOT_SECONDS = 300
# 4 hours of BTC 1s history. Long enough to:
#   - compute btc_ret_*s up to 3600s (1h)
#   - warm up multi-TF RSI on 1m / 3m / 5m / 15m (14 bars × 15m = 12600s)
# rsi_30m/60m/240m still dead without per-TF bar buffers — deferred.
_BTC_WINDOW_SECONDS = 14400
_SNAPSHOT_TICK_SIZE = 0.001


# ---------------------------------------------------------------------------
# Pure parsing helpers (no IO)
# ---------------------------------------------------------------------------


def parse_book_dict(
    levels: Optional[Mapping[str, object]],
    side: str,
) -> List[OrderBookEntry]:
    """Convert a ``{price: size}`` depth dict into sorted OrderBookEntry list.

    Bids are returned descending (best bid first); asks ascending (best ask
    first), matching the live ``ClobClient`` convention. Non-numeric entries
    are silently dropped.
    """
    if not levels:
        return []
    parsed: List[Tuple[float, float]] = []
    for price_raw, size_raw in levels.items():
        try:
            price = float(price_raw)
            size = float(size_raw)
        except (TypeError, ValueError):
            continue
        if price <= 0 or size <= 0:
            continue
        parsed.append((price, size))
    parsed.sort(key=lambda pair: pair[0], reverse=(side == "bids"))
    return [OrderBookEntry(price=price, size=size) for price, size in parsed]


def snapshot_to_order_books(
    snap: Mapping[str, object],
) -> Tuple[OrderBook, OrderBook]:
    """Build full-depth YES and NO books from one snapshot record."""
    yes_bids = parse_book_dict(snap.get("yes_bids"), "bids")
    yes_asks = parse_book_dict(snap.get("yes_asks"), "asks")
    no_bids = parse_book_dict(snap.get("no_bids"), "bids")
    no_asks = parse_book_dict(snap.get("no_asks"), "asks")

    yes_book = OrderBook(
        market_id="",
        token_id=str(snap.get("up_token") or ""),
        bids=yes_bids,
        asks=yes_asks,
        tick_size=_SNAPSHOT_TICK_SIZE,
    )
    no_book = OrderBook(
        market_id="",
        token_id=str(snap.get("down_token") or ""),
        bids=no_bids,
        asks=no_asks,
        tick_size=_SNAPSHOT_TICK_SIZE,
    )
    return yes_book, no_book


def iter_jsonl_records(lines: Iterable[str]) -> Iterator[dict]:
    """Parse JSONL lines, skipping empty/invalid rows."""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


@dataclass
class SnapshotBatch:
    """Parsed jsonl records for one S3 key, split into snapshots + outcomes."""
    snapshots: List[dict]
    outcomes: Dict[int, str]  # slot_ts -> "Up"|"Down"

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "SnapshotBatch":
        snapshots: List[dict] = []
        outcomes: Dict[int, str] = {}
        for rec in iter_jsonl_records(lines):
            if rec.get("type") == "outcome":
                try:
                    slot_ts = int(rec["slot_ts"])
                    outcomes[slot_ts] = str(rec["outcome"])
                except (KeyError, TypeError, ValueError):
                    continue
            elif "slot_ts" in rec and "snapshot_ts" in rec:
                snapshots.append(rec)
        return cls(snapshots=snapshots, outcomes=outcomes)


# ---------------------------------------------------------------------------
# Feature-frame builder
# ---------------------------------------------------------------------------


def build_features_frame(
    snapshots: Sequence[Mapping[str, object]],
    outcomes: Mapping[int, str],
    btc_1s_df: Optional[pd.DataFrame] = None,
    btc_window_seconds: int = _BTC_WINDOW_SECONDS,
    logger: Optional[logging.Logger] = None,
    add_random_sentinel: bool = False,
    random_seed: int = 0,
    n_random_sentinels: int = 1,
    last_snap_only: bool = False,
    min_tte_seconds: float = 0.0,
) -> pd.DataFrame:
    """Assemble a feature DataFrame from parsed snapshot records.

    Parameters
    ----------
    snapshots : iterable of snapshot dicts (non-sentinel).
    outcomes : slot_ts -> "Up"|"Down" (from outcome sentinels OR slot_key_map).
    btc_1s_df : optional OHLCV frame with columns [timestamp, close]; when
        provided, the rolling BTC history inside the feature builder is
        sourced from it rather than the short in-snapshot return window.
    btc_window_seconds : rolling window passed to the feature builder.
    last_snap_only : when True, emit only the final snapshot row per slot.
        State (slot_path, yes/no history) is still accumulated across all
        snapshots; only the build_live_features() call is deferred to the
        last valid snapshot. ~60x faster for backtest-only use cases.
    min_tte_seconds : when > 0 and last_snap_only is True, only consider
        snapshots with at least this many seconds remaining as the candidate
        decision point. Eliminates last-snapshot bias where market prices
        have converged to near-certain outcomes near expiry.
    """
    log = logger or logging.getLogger(__name__)

    btc_ts: np.ndarray
    btc_price: np.ndarray
    if btc_1s_df is not None and not btc_1s_df.empty:
        btc_ts = btc_1s_df["timestamp"].to_numpy(dtype=float)
        btc_price = btc_1s_df["close"].to_numpy(dtype=float)
    else:
        btc_ts = np.empty(0, dtype=float)
        btc_price = np.empty(0, dtype=float)

    # Pre-group snapshots by slot so within-slot histories (yes_mid, no_mid)
    # can be built incrementally — consistent with how the live path sees them.
    snaps_by_slot: Dict[int, List[Mapping[str, object]]] = {}
    for snap in snapshots:
        try:
            slot_ts = int(snap["slot_ts"])
        except (KeyError, TypeError, ValueError):
            continue
        snaps_by_slot.setdefault(slot_ts, []).append(snap)

    rows: List[Dict[str, object]] = []
    skipped_no_strike = 0
    skipped_no_outcome = 0

    for slot_ts in sorted(snaps_by_slot):
        outcome = outcomes.get(slot_ts)
        if outcome is None:
            skipped_no_outcome += 1
            continue
        label = 1 if str(outcome).strip().lower() == "up" else 0

        ordered = sorted(
            snaps_by_slot[slot_ts], key=lambda r: float(r.get("snapshot_ts") or 0.0)
        )
        yes_history: List[Tuple[float, float]] = []
        no_history: List[Tuple[float, float]] = []

        # BTC ticks restricted to this slot's window; used to drive
        # SlotPathState as we walk the snapshots in time order.
        slot_tick_ts, slot_tick_price = _slice_slot_ticks(
            btc_ts, btc_price, slot_ts
        )
        slot_state = SlotPathState()
        slot_state.reset(slot_ts)
        tick_idx = 0

        # When last_snap_only, we accumulate state across all snapshots but
        # defer build_live_features() to the last valid one. Use a dict that
        # gets overwritten each iteration; we call the builder once after.
        pending: Optional[Dict] = None

        for snap in ordered:
            snapshot_ts = float(snap.get("snapshot_ts") or 0.0)
            strike_raw = snap.get("strike")
            if strike_raw in (None, "", 0):
                skipped_no_strike += 1
                continue
            try:
                strike = float(strike_raw)
            except (TypeError, ValueError):
                skipped_no_strike += 1
                continue

            # Fold every slot tick that arrived at or before this snapshot.
            while (
                tick_idx < slot_tick_ts.size
                and slot_tick_ts[tick_idx] <= snapshot_ts
            ):
                slot_state.update(
                    float(slot_tick_ts[tick_idx]),
                    float(slot_tick_price[tick_idx]),
                    strike,
                )
                tick_idx += 1

            btc_now_raw = snap.get("btc_now")
            try:
                btc_now_value = float(btc_now_raw) if btc_now_raw is not None else 0.0
            except (TypeError, ValueError):
                btc_now_value = 0.0
            # Fold the snapshot's own btc_now reading. The 1s CSV and the
            # snapshot's live feed diverge slightly (different sample moments),
            # so without this, slot_max could miss a snapshot-only excursion —
            # breaking the invariant slot_high_excursion_bps >= slot_drift_bps.
            if btc_now_value > 0:
                slot_state.update(snapshot_ts, btc_now_value, strike)
            slot_path_features = slot_state.to_features(
                snapshot_ts, btc_now_value, strike
            )

            yes_book, no_book = snapshot_to_order_books(snap)
            yes_mid = snap.get("yes_mid")
            no_mid = snap.get("no_mid")
            if yes_mid is not None:
                try:
                    yes_history.append((snapshot_ts, float(yes_mid)))
                except (TypeError, ValueError):
                    pass
            if no_mid is not None:
                try:
                    no_history.append((snapshot_ts, float(no_mid)))
                except (TypeError, ValueError):
                    pass

            btc_prices = _btc_window_from_arrays(
                btc_ts, btc_price, snapshot_ts, btc_window_seconds
            )
            if len(btc_prices) < 2:
                if btc_now_value > 0:
                    # Synthesize a two-point history so the feature builder can
                    # at least compute moneyness/TTE; returns will be zero.
                    btc_prices = [
                        (snapshot_ts - 1.0, btc_now_value),
                        (snapshot_ts, btc_now_value),
                    ]

            # Recent slot outcomes BEFORE this slot — feeds the
            # recent_up_rate_{5,10,20} features. No peeking ahead: the
            # current slot's outcome is excluded.
            prior_outcomes = sorted(
                ((s_ts, o) for s_ts, o in outcomes.items() if s_ts < slot_ts),
                key=lambda kv: kv[0],
            )
            recent_slot_outcomes = [o for _, o in prior_outcomes[-20:]]

            snapshot_for_features = {
                "btc_prices": btc_prices,
                "yes_book": yes_book,
                "no_book": no_book,
                "yes_history": list(yes_history),
                "no_history": list(no_history),
                "question": "",
                "strike_price": strike,
                "slot_expiry_ts": slot_ts + _SLOT_SECONDS,
                "now_ts": snapshot_ts,
                "slot_path_features": slot_path_features,
                "recent_slot_outcomes": recent_slot_outcomes,
            }

            if last_snap_only:
                tte = (slot_ts + _SLOT_SECONDS) - snapshot_ts
                if min_tte_seconds <= 0 or tte >= min_tte_seconds:
                    # Overwrite pending each iteration; build only after the loop.
                    pending = {"args": snapshot_for_features, "ts": snapshot_ts}
            else:
                built = build_live_features(snapshot_for_features)
                row: Dict[str, object] = {
                    "slot_ts": slot_ts,
                    "snapshot_ts": snapshot_ts,
                    "slot_expiry_ts": slot_ts + _SLOT_SECONDS,
                    "label": label,
                    "outcome": outcome,
                    "feature_status": built.status,
                }
                row.update(built.features)
                rows.append(row)

        if last_snap_only and pending is not None:
            built = build_live_features(pending["args"])
            row = {
                "slot_ts": slot_ts,
                "snapshot_ts": pending["ts"],
                "slot_expiry_ts": slot_ts + _SLOT_SECONDS,
                "label": label,
                "outcome": outcome,
                "feature_status": built.status,
            }
            row.update(built.features)
            rows.append(row)

    if skipped_no_strike:
        log.info("Skipped %d snapshots with missing strike", skipped_no_strike)
    if skipped_no_outcome:
        log.info("Skipped %d slots with no outcome mapping", skipped_no_outcome)

    if not rows:
        return pd.DataFrame()
    frame = (
        pd.DataFrame(rows)
        .sort_values(["slot_ts", "snapshot_ts"])
        .reset_index(drop=True)
    )
    if add_random_sentinel:
        _inject_random_sentinels(
            frame, seed=random_seed, n=max(1, int(n_random_sentinels))
        )
    return frame


def _inject_random_sentinels(frame: pd.DataFrame, seed: int, n: int) -> None:
    """Add n standard-normal ``random_value_k`` columns (in-place).

    Sentinels act as a noise floor during feature-importance analysis: any real
    feature whose importance sits below a sentinel is worse than random.
    """
    rng = np.random.default_rng(seed)
    if n == 1:
        frame["random_value"] = rng.standard_normal(len(frame))
        return
    for idx in range(n):
        frame[f"random_value_{idx}"] = rng.standard_normal(len(frame))


def _slice_slot_ticks(
    btc_ts: np.ndarray,
    btc_price: np.ndarray,
    slot_ts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return BTC ticks within ``[slot_ts, slot_ts + _SLOT_SECONDS]``."""
    if btc_ts.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    lo = float(slot_ts)
    hi = float(slot_ts + _SLOT_SECONDS)
    mask = (btc_ts >= lo) & (btc_ts <= hi)
    return btc_ts[mask], btc_price[mask]


def _btc_window_from_arrays(
    btc_ts: np.ndarray,
    btc_price: np.ndarray,
    snapshot_ts: float,
    window_seconds: int,
) -> List[Tuple[float, float]]:
    if btc_ts.size == 0:
        return []
    lo = snapshot_ts - window_seconds
    mask = (btc_ts >= lo) & (btc_ts <= snapshot_ts)
    if not np.any(mask):
        return []
    window_ts = btc_ts[mask]
    window_price = btc_price[mask]
    return [(float(t), float(p)) for t, p in zip(window_ts, window_price) if p > 0]


# ---------------------------------------------------------------------------
# S3 IO (thin wrapper around boto3)
# ---------------------------------------------------------------------------


def _get_s3_client():  # pragma: no cover — trivial import wrapper
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 access (install with `pip install boto3`)"
        ) from exc
    return boto3.client("s3")


def list_hour_prefixes(
    bucket: str,
    date: str,
    prefix_root: str = DEFAULT_PREFIX,
    client=None,
) -> List[str]:
    """Return sorted hour prefixes under ``s3://bucket/<prefix_root>/<date>/``."""
    client = client or _get_s3_client()
    prefix = f"{prefix_root}/{date}/"
    paginator = client.get_paginator("list_objects_v2")
    hours: set = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes") or []:
            p = item.get("Prefix", "")
            parts = p.strip("/").split("/")
            if len(parts) >= 3:
                hours.add(parts[-1])
    return sorted(hours)


def read_s3_text(bucket: str, key: str, client=None) -> str:
    """Read an S3 object as UTF-8 text. Raises on missing key."""
    client = client or _get_s3_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    return body.decode("utf-8", errors="replace")


def load_hour_batch(
    bucket: str,
    date: str,
    hour: str,
    prefix_root: str = DEFAULT_PREFIX,
    client=None,
) -> Tuple[SnapshotBatch, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch the three files for a single hour prefix.

    Returns (snapshots, btc_1s_df | None, slot_map_df | None).
    Missing files are returned as None rather than raised, since a running hour
    may legitimately be missing slot_key_map.csv.
    """
    client = client or _get_s3_client()
    base = f"{prefix_root}/{date}/{hour}"

    snap_text = read_s3_text(bucket, f"{base}/{SNAPSHOT_FILENAME}", client=client)
    batch = SnapshotBatch.from_lines(snap_text.splitlines())

    btc_df: Optional[pd.DataFrame] = None
    try:
        btc_text = read_s3_text(bucket, f"{base}/{BTC_FILENAME}", client=client)
        btc_df = pd.read_csv(io.StringIO(btc_text))
    except Exception:
        btc_df = None

    slot_df: Optional[pd.DataFrame] = None
    try:
        slot_text = read_s3_text(bucket, f"{base}/{SLOT_MAP_FILENAME}", client=client)
        slot_df = pd.read_csv(io.StringIO(slot_text))
    except Exception:
        slot_df = None

    return batch, btc_df, slot_df


def load_dates(
    dates: Sequence[str],
    bucket: str = DEFAULT_BUCKET,
    prefix_root: str = DEFAULT_PREFIX,
    client=None,
    logger: Optional[logging.Logger] = None,
    add_random_sentinel: bool = False,
    random_seed: int = 0,
    n_random_sentinels: int = 1,
    last_snap_only: bool = False,
    min_tte_seconds: float = 0.0,
) -> pd.DataFrame:
    """Load one or more date partitions and build the training frame.

    ``dates`` are ISO strings like ``"2026-04-18"``. All hours available under
    each date are fetched and concatenated. Outcomes from jsonl sentinels take
    precedence; ``slot_key_map.csv`` fills in anything missing.
    """
    client = client or _get_s3_client()
    log = logger or logging.getLogger(__name__)

    all_snapshots: List[dict] = []
    all_outcomes: Dict[int, str] = {}
    btc_frames: List[pd.DataFrame] = []

    for date in dates:
        hours = list_hour_prefixes(bucket, date, prefix_root=prefix_root, client=client)
        log.info("Date %s has %d hour prefixes", date, len(hours))
        for hour in hours:
            try:
                batch, btc_df, slot_df = load_hour_batch(
                    bucket, date, hour, prefix_root=prefix_root, client=client
                )
            except Exception as exc:
                log.warning("Failed to load %s/%s: %s", date, hour, exc)
                continue
            all_snapshots.extend(batch.snapshots)
            all_outcomes.update(batch.outcomes)
            if slot_df is not None:
                for slot_ts, outcome in _outcomes_from_slot_map(slot_df):
                    all_outcomes.setdefault(slot_ts, outcome)
            if btc_df is not None and not btc_df.empty:
                btc_frames.append(btc_df)

    btc_combined: Optional[pd.DataFrame] = None
    if btc_frames:
        btc_combined = (
            pd.concat(btc_frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    return build_features_frame(
        all_snapshots,
        all_outcomes,
        btc_1s_df=btc_combined,
        logger=log,
        add_random_sentinel=add_random_sentinel,
        random_seed=random_seed,
        n_random_sentinels=n_random_sentinels,
        last_snap_only=last_snap_only,
        min_tte_seconds=min_tte_seconds,
    )


def _outcomes_from_slot_map(slot_df: pd.DataFrame) -> Iterator[Tuple[int, str]]:
    if "slot_ts" not in slot_df.columns or "outcome" not in slot_df.columns:
        return
    for row in slot_df.itertuples(index=False):
        try:
            slot_ts = int(row.slot_ts)
        except (TypeError, ValueError):
            continue
        outcome = str(row.outcome or "").strip()
        if outcome.lower() in {"up", "down"}:
            yield slot_ts, outcome
