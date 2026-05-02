"""Shared period-based train/validation/test split for the ML pipeline.

Replaces ad-hoc ``split_by_slot(df, val_ratio)`` copies that were duplicated
across training and backtest scripts. Callers now pass explicit date ranges
for each split so reproducibility is guaranteed across different data
windows.

Accepted period-spec formats (each side of the ``:`` separator):
  * ``YYYY-MM-DD``                    — interpreted as 00:00:00 UTC that day
  * ``YYYY-MM-DDTHH:MM:SSZ``          — ISO 8601 UTC (trailing ``Z`` mandatory)
  * Unix seconds as int or float      — e.g. ``1776549600``

Examples:
  * ``"2026-03-01:2026-04-01"``        — month of March 2026 UTC
  * ``"2026-03-01T12:00:00Z:2026-03-02T12:00:00Z"`` — sub-day precision
  * ``"1776000000:1776600000"``        — raw seconds (7-day window)

End is *exclusive*. A row matches a period iff ``start_ts <= ts < end_ts``.

Backward compatibility
----------------------
When callers don't pass any period flags, ``resolve_split_from_args`` falls
back to the legacy ``--val-ratio`` behavior: sort unique slot timestamps and
take the last ``val_ratio`` slots as the validation set. This keeps existing
probe/train pipelines working without modification.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Canonical period labels. The order matters: it's used for reporting and
# for chronological validation (each label must start at or after the prior).
PERIOD_LABELS: Tuple[str, str, str] = ("training", "validation", "test")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodBounds:
    """Half-open time window ``[start_ts, end_ts)`` in UTC seconds.

    ``label`` is one of ``"training"``, ``"validation"``, ``"test"``.
    """

    label: str
    start_ts: float
    end_ts: float

    def __post_init__(self) -> None:
        if self.label not in PERIOD_LABELS:
            raise ValueError(f"label must be one of {PERIOD_LABELS}, got {self.label!r}")
        if self.end_ts <= self.start_ts:
            raise ValueError(
                f"Period {self.label!r}: end_ts ({self.end_ts}) must be > start_ts ({self.start_ts})"
            )

    def contains(self, ts: float) -> bool:
        """``True`` iff ``start_ts <= ts < end_ts``."""
        return self.start_ts <= ts < self.end_ts

    def mask(self, series: pd.Series) -> pd.Series:
        """Vectorised boolean mask over a Series of timestamps."""
        return (series >= self.start_ts) & (series < self.end_ts)

    def humanize(self) -> str:
        """Return a friendly UTC string for logs/reports."""
        a = datetime.fromtimestamp(self.start_ts, tz=timezone.utc).isoformat()
        b = datetime.fromtimestamp(self.end_ts, tz=timezone.utc).isoformat()
        return f"{a} → {b}"

    def to_dict(self) -> Dict[str, object]:
        """Serializable form for meta.json / audit logs."""
        return {
            "label": self.label,
            "start_ts": float(self.start_ts),
            "end_ts": float(self.end_ts),
            "start_utc": datetime.fromtimestamp(self.start_ts, tz=timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(self.end_ts, tz=timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class PeriodConfig:
    """Full split configuration. Training is always required; val and test
    are optional (callers may have no held-out test set, for example)."""

    training: PeriodBounds
    validation: Optional[PeriodBounds] = None
    test: Optional[PeriodBounds] = None

    def periods(self) -> List[PeriodBounds]:
        """Return the configured periods, in chronological label order."""
        out: List[PeriodBounds] = [self.training]
        if self.validation is not None:
            out.append(self.validation)
        if self.test is not None:
            out.append(self.test)
        return out

    def validate(self) -> None:
        """Enforce ordering (training < validation < test) and no overlaps.

        This is redundant when configs are built via ``parse_period_config``
        (which calls this internally) but provides a safety net for callers
        that construct configs programmatically.
        """
        periods = self.periods()
        # Chronological order of START times must match label order.
        for earlier, later in zip(periods, periods[1:]):
            if later.start_ts < earlier.start_ts:
                raise ValueError(
                    f"Periods out of chronological order: "
                    f"{earlier.label} starts {earlier.humanize()} but "
                    f"{later.label} starts {later.humanize()}"
                )
        # No overlap — each period's start must be >= prior period's end.
        for earlier, later in zip(periods, periods[1:]):
            if later.start_ts < earlier.end_ts:
                raise ValueError(
                    f"Periods overlap: {earlier.label} ends {earlier.humanize()} "
                    f"but {later.label} starts before that"
                )

    def to_dict(self) -> Dict[str, object]:
        """Serializable for meta.json."""
        return {p.label: p.to_dict() for p in self.periods()}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_period_spec(spec: str, label: str) -> PeriodBounds:
    """Parse a single ``"START:END"`` (or ``"START..END"``) string.

    Each side may be ``YYYY-MM-DD``, ``YYYY-MM-DDTHH:MM:SSZ``, or Unix seconds.
    ISO datetimes themselves contain ``:``, so for the ``:`` separator form
    we try every candidate split position and accept the first pair that
    parses on both sides. Using ``..`` avoids this ambiguity entirely.

    Raises:
        ValueError: on malformed input, inverted range, or bad label.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"Period spec for {label!r} must be a non-empty string")

    left, right = _split_spec(spec, label)
    start_ts = _parse_timestamp(left.strip())
    end_ts = _parse_timestamp(right.strip())
    return PeriodBounds(label=label, start_ts=start_ts, end_ts=end_ts)


def _split_spec(spec: str, label: str) -> Tuple[str, str]:
    """Split a period spec into (start_token, end_token).

    Prefers the unambiguous ``..`` separator. For ``:``, probes every
    candidate position and accepts the first where both halves parse as
    timestamps.
    """
    if ".." in spec:
        parts = spec.split("..", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Period spec {spec!r} for {label!r} must look like 'START..END'"
            )
        return parts[0], parts[1]

    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Period spec {spec!r} for {label!r} must look like 'START:END' or 'START..END'"
        )
    # Try every possible split index; pick the first where both sides parse.
    for i in range(1, len(parts)):
        left = ":".join(parts[:i])
        right = ":".join(parts[i:])
        if not left or not right:
            continue
        try:
            _parse_timestamp(left)
            _parse_timestamp(right)
        except ValueError:
            continue
        return left, right
    raise ValueError(
        f"Period spec {spec!r} for {label!r} must look like 'START:END' or 'START..END'; "
        "if both sides are ISO datetimes use the '..' separator to disambiguate."
    )


def parse_period_config(
    training: Optional[str],
    validation: Optional[str] = None,
    test: Optional[str] = None,
) -> Optional[PeriodConfig]:
    """Parse the 3 CLI strings into a validated PeriodConfig.

    Returns ``None`` when no training period is provided (signals to callers
    that they should use the ``--val-ratio`` fallback).

    Raises:
        ValueError: on malformed specs or overlapping / out-of-order periods.
    """
    if not training:
        if validation or test:
            raise ValueError("--valid-period / --test-period require --training-period")
        return None

    cfg = PeriodConfig(
        training=parse_period_spec(training, "training"),
        validation=parse_period_spec(validation, "validation") if validation else None,
        test=parse_period_spec(test, "test") if test else None,
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_by_periods(
    df: pd.DataFrame,
    config: PeriodConfig,
    ts_col: str = "slot_ts",
) -> Dict[str, pd.DataFrame]:
    """Filter rows into ``{label: sub_df}`` by each period's time window.

    Args:
        df: source frame containing ``ts_col``.
        config: validated PeriodConfig.
        ts_col: timestamp column name. Defaults to ``"slot_ts"`` which is the
            5-minute slot start used across this repo.

    Returns:
        Dict with keys that mirror the configured periods (always
        ``"training"``; ``"validation"`` and ``"test"`` only when present).
        Row order within each sub-frame is preserved.

    Raises:
        KeyError: if ``ts_col`` is missing.
        ValueError: if a period yields an empty sub-frame (caller likely
            passed a period that doesn't intersect the data).
    """
    if ts_col not in df.columns:
        raise KeyError(f"Timestamp column {ts_col!r} not found in dataframe")

    ts = df[ts_col]
    out: Dict[str, pd.DataFrame] = {}
    for period in config.periods():
        mask = period.mask(ts)
        sub = df.loc[mask].reset_index(drop=True)
        if sub.empty:
            raise ValueError(
                f"Period {period.label!r} ({period.humanize()}) matched 0 rows in "
                f"column {ts_col!r}. Check the period bounds vs. data range "
                f"[{ts.min()}, {ts.max()}]."
            )
        out[period.label] = sub
    return out


def split_by_val_ratio(
    df: pd.DataFrame,
    val_ratio: float,
    ts_col: str = "slot_ts",
) -> Dict[str, pd.DataFrame]:
    """Legacy walk-forward split: last ``val_ratio`` of unique timestamps → val.

    Produces the same train/val frames as the pre-existing copy-pasted
    ``split_by_slot()`` helpers. Provided so scripts can be refactored to
    use ``resolve_split_from_args`` uniformly.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1); got {val_ratio}")
    if ts_col not in df.columns:
        raise KeyError(f"Timestamp column {ts_col!r} not found in dataframe")

    ts_unique = np.sort(df[ts_col].unique())
    cutoff = max(1, int(round(len(ts_unique) * (1.0 - val_ratio))))
    train_ts = set(ts_unique[:cutoff].tolist())
    val_ts = set(ts_unique[cutoff:].tolist())
    return {
        "training": df[df[ts_col].isin(train_ts)].reset_index(drop=True),
        "validation": df[df[ts_col].isin(val_ts)].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def add_period_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``--training-period``, ``--valid-period``, ``--test-period``.

    Kept deliberately as three separate flags rather than a single
    ``--periods`` blob so shell history stays legible.
    """
    group = parser.add_argument_group("period split")
    group.add_argument(
        "--training-period", default=None, metavar="START:END",
        help=(
            "Date range for the training set (e.g. '2026-03-01:2026-04-01' "
            "or '2026-03-01T00:00:00Z:2026-04-01T00:00:00Z'). End is exclusive. "
            "When omitted, falls back to --val-ratio."
        ),
    )
    group.add_argument(
        "--valid-period", default=None, metavar="START:END",
        help="Optional validation period; must follow --training-period.",
    )
    group.add_argument(
        "--test-period", default=None, metavar="START:END",
        help="Optional held-out test period; must follow --valid-period.",
    )


@dataclass(frozen=True)
class ResolvedSplit:
    """Output of ``resolve_split_from_args`` — carries the period config (if
    any) alongside the sub-frames so callers can log audit metadata."""

    frames: Dict[str, pd.DataFrame]
    config: Optional[PeriodConfig]
    used_fallback: bool  # True when we split via val_ratio instead of periods


def resolve_split_from_args(
    args: argparse.Namespace,
    df: pd.DataFrame,
    val_ratio: float,
    ts_col: str = "slot_ts",
    logger: Optional[logging.Logger] = None,
) -> ResolvedSplit:
    """Unified entrypoint used by training/backtest scripts.

    If any period flag was passed, build a ``PeriodConfig`` and split by it.
    Otherwise fall back to ``split_by_val_ratio`` with the caller's default.

    The caller supplies ``val_ratio`` explicitly so existing scripts can keep
    their own defaults (some use 0.20, others differ).
    """
    log = logger or logging.getLogger(__name__)

    training = getattr(args, "training_period", None)
    validation = getattr(args, "valid_period", None)
    test = getattr(args, "test_period", None)

    config = parse_period_config(training=training, validation=validation, test=test)

    if config is None:
        frames = split_by_val_ratio(df, val_ratio=val_ratio, ts_col=ts_col)
        log.info(
            "Period split: val_ratio=%.2f fallback (train=%d rows, val=%d rows)",
            val_ratio, len(frames["training"]), len(frames["validation"]),
        )
        return ResolvedSplit(frames=frames, config=None, used_fallback=True)

    frames = split_by_periods(df, config, ts_col=ts_col)
    for period in config.periods():
        n = len(frames[period.label])
        log.info("Period split: %s %s → %d rows", period.label, period.humanize(), n)
    return ResolvedSplit(frames=frames, config=config, used_fallback=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(token: str) -> float:
    """Coerce a single start/end token to UTC seconds.

    Order of attempts: numeric → ``YYYY-MM-DDTHH:MM:SSZ`` → ``YYYY-MM-DD``.
    """
    # Numeric (Unix seconds).
    try:
        return float(token)
    except ValueError:
        pass

    # ISO datetime with trailing Z (always UTC).
    if token.endswith("Z"):
        try:
            dt = datetime.strptime(token, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass

    # Date-only form, interpreted as start-of-day UTC.
    try:
        dt = datetime.strptime(token, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError as exc:
        raise ValueError(
            f"Could not parse timestamp token {token!r}. "
            "Expected 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SSZ', or Unix seconds."
        ) from exc
