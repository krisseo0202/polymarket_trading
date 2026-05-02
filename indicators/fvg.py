"""FVG (Fair Value Gap) Indicator Implementation"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .base import (
    Indicator,
    IndicatorResult,
    Signal,
    SignalType,
    SignalSeverity
)


@dataclass
class FVGRecord:
    """Record of a detected Fair Value Gap"""
    start_index: int
    is_bullish: bool
    min_level: float
    max_level: float
    mitigated: bool = False
    mitigation_index: Optional[int] = None
    timestamp: Optional[pd.Timestamp] = None


def _compute_threshold_series(high: np.ndarray, low: np.ndarray, auto: bool, threshold_percent: float):
    """
    Utility for dynamic per-bar threshold for gap size.
    Matches Pine behavior: if auto, avg bar size, else flat line.
    """
    n = len(high)
    if not auto:
        return np.full(n, threshold_percent, dtype=float)
    avg_bar_size = np.mean(high - low) / np.mean(high)
    return np.full(n, avg_bar_size, dtype=float)


class FVGIndicator(Indicator):
    """Fair Value Gap (FVG) indicator
    
    Detects bullish and bearish fair value gaps and tracks their mitigation.
    
    Parameters:
        threshold_percent: Minimum gap size as percentage (default: 0.0)
        auto: Use automatic threshold calculation (default: False)
        time_column: Optional time column name
    """

    def _detect_fvgs_with_counts(
        self,
        ohlc: pd.DataFrame,
        threshold_percent: float,
        auto: bool
    ):
        n = len(ohlc)
        empty_int = np.zeros(n, dtype=int)
        empty_float = np.full(n, np.nan, dtype=float)
        sentinel_int = np.full(n, 999, dtype=int)
        if n < 3:
            return (
                [],
                empty_int.copy(), empty_int.copy(),
                empty_int.copy(), empty_int.copy(),
                empty_int.copy(), empty_int.copy(),
                empty_float.copy(), empty_float.copy(),
                sentinel_int.copy(),
            )

        high = ohlc["high"].astype(float).to_numpy()
        low  = ohlc["low"].astype(float).to_numpy()
        close= ohlc["close"].astype(float).to_numpy()

        # timestamps (optional)
        times = list(ohlc.index) if isinstance(ohlc.index, pd.DatetimeIndex) else [None]*n

        threshold_series = _compute_threshold_series(high, low, auto, threshold_percent)

        records: List[FVGRecord] = []
        active:  List[FVGRecord] = []

        # bar-by-bar counters to match Pine
        bull_count       = np.zeros(n, dtype=int)
        bear_count       = np.zeros(n, dtype=int)
        bull_mitigated   = np.zeros(n, dtype=int)
        bear_mitigated   = np.zeros(n, dtype=int)

        # new per-bar continuous arrays
        bull_count_active  = np.zeros(n, dtype=int)
        bear_count_active  = np.zeros(n, dtype=int)
        dist_to_nearest_bull_fvg = np.full(n, np.nan, dtype=float)
        dist_to_nearest_bear_fvg = np.full(n, np.nan, dtype=float)
        nearest_fvg_age_bars     = np.full(n, 999, dtype=int)

        last_new_time = None  # Pine prevents duplicate adds on same bar/time

        for i in range(2, n):
            thr = threshold_series[i]

            # --- Detect bullish FVG (Pine)
            is_bull = (
                low[i] > high[i-2] and
                close[i-1] > high[i-2] and
                (low[i] - high[i-2]) / high[i-2] > thr
            )
            if is_bull and times[i] != last_new_time:
                rec = FVGRecord(
                    start_index=i,
                    is_bullish=True,
                    # Pine: fvg.new(low, high[2], true) => max=low, min=high[2]
                    min_level=float(high[i-2]),
                    max_level=float(low[i]),
                    mitigated=False,
                    timestamp=times[i]
                )
                records.append(rec)
                active.append(rec)
                bull_count[i] = bull_count[i-1] + 1
                bear_count[i] = bear_count[i-1]
                last_new_time = times[i]
            # --- Detect bearish FVG (Pine)
            is_bear = (
                high[i] < low[i-2] and
                close[i-1] < low[i-2] and
                (low[i-2] - high[i]) / high[i] > thr
            )
            if is_bear and times[i] != last_new_time:
                rec = FVGRecord(
                    start_index=i,
                    is_bullish=False,
                    # Pine: fvg.new(low[2], high, false) => max=low[2], min=high
                    min_level=float(high[i]),
                    max_level=float(low[i-2]),
                    mitigated=False,
                    timestamp=times[i]
                )
                records.append(rec)
                active.append(rec)
                bear_count[i] = bear_count[i-1] + 1
                bull_count[i] = max(bull_count[i], bull_count[i-1])  # carry over if not set
                last_new_time = times[i]

            # carry forward counts if no new FVG this bar
            if not is_bull and not is_bear:
                bull_count[i] = bull_count[i-1]
                bear_count[i] = bear_count[i-1]

            # --- Mitigation checks on the **current** bar (Pine semantics)
            still_active = []
            for rec in active:
                if rec.is_bullish:
                    if close[i] < rec.min_level:
                        rec.mitigated = True
                        rec.mitigation_index = i
                        bull_mitigated[i] = bull_mitigated[i-1] + 1
                    else:
                        still_active.append(rec)
                else:
                    if close[i] > rec.max_level:
                        rec.mitigated = True
                        rec.mitigation_index = i
                        bear_mitigated[i] = bear_mitigated[i-1] + 1
                    else:
                        still_active.append(rec)
            active = still_active

            # carry forward mitigated counts if nothing changed this bar
            if bull_mitigated[i] == 0 and i > 0:
                bull_mitigated[i] = bull_mitigated[i-1]
            if bear_mitigated[i] == 0 and i > 0:
                bear_mitigated[i] = bear_mitigated[i-1]

            # --- Per-bar continuous features (computed from still_active after mitigation)
            active_bull = [r for r in active if r.is_bullish]
            active_bear = [r for r in active if not r.is_bullish]

            bull_count_active[i] = len(active_bull)
            bear_count_active[i] = len(active_bear)

            if active_bull:
                mids = [(r.min_level + r.max_level) / 2.0 for r in active_bull]
                nearest_mid = min(mids, key=lambda m: abs(m - close[i]))
                dist_to_nearest_bull_fvg[i] = (nearest_mid - close[i]) / close[i]

            if active_bear:
                mids = [(r.min_level + r.max_level) / 2.0 for r in active_bear]
                nearest_mid = min(mids, key=lambda m: abs(m - close[i]))
                dist_to_nearest_bear_fvg[i] = (nearest_mid - close[i]) / close[i]

            # Age of the most-recently-formed FVG (bull or bear)
            recent_records = [r for r in records if not r.mitigated]
            if recent_records:
                newest = max(recent_records, key=lambda r: r.start_index)
                nearest_fvg_age_bars[i] = i - newest.start_index

        return (
            records,
            bull_count, bear_count, bull_mitigated, bear_mitigated,
            bull_count_active, bear_count_active,
            dist_to_nearest_bull_fvg, dist_to_nearest_bear_fvg,
            nearest_fvg_age_bars,
        )

    def compute(self, ohlc: pd.DataFrame, timeframe: str = "unknown") -> IndicatorResult:
        self.validate_ohlc(ohlc)

        threshold_percent = self.params.get("threshold_percent", 0.0)
        auto = self.params.get("auto", False)

        (
            recs, bull_cnt, bear_cnt, bull_mit, bear_mit,
            bull_cnt_active, bear_cnt_active,
            dist_bull_fvg, dist_bear_fvg,
            fvg_age_bars,
        ) = self._detect_fvgs_with_counts(ohlc, threshold_percent, auto)

        n = len(ohlc)
        signals: List[Signal] = []
        ts_last = ohlc.index[-1] if isinstance(ohlc.index, pd.DatetimeIndex) else None

        # Pine: alertcondition(bull_count > bull_count[1], 'Bullish FVG', ...)
        if n >= 2 and bull_cnt[-1] > bull_cnt[-2]:
            signals.append(Signal(
                type=SignalType.BULL_FVG,
                severity=SignalSeverity.MEDIUM,   # choose your mapping; Pine doesn’t grade severity
                value=1.0,
                message="Bullish FVG detected",
                timestamp=ts_last
            ))

        if n >= 2 and bear_cnt[-1] > bear_cnt[-2]:
            signals.append(Signal(
                type=SignalType.BEAR_FVG,
                severity=SignalSeverity.MEDIUM,
                value=1.0,
                message="Bearish FVG detected",
                timestamp=ts_last
            ))

        # Mitigation alerts
        if n >= 2 and bull_mit[-1] > bull_mit[-2]:
            signals.append(Signal(
                type=SignalType.BULL_FVG_MITIGATED,
                severity=SignalSeverity.LOW,
                value=1.0,
                message="Bullish FVG mitigated",
                timestamp=ts_last
            ))

        if n >= 2 and bear_mit[-1] > bear_mit[-2]:
            signals.append(Signal(
                type=SignalType.BEAR_FVG_MITIGATED,
                severity=SignalSeverity.LOW,
                value=1.0,
                message="Bearish FVG mitigated",
                timestamp=ts_last
            ))

        # Optional: include latest unmitigated record for your own UI, but don’t use it for alerts matching Pine
        latest_unmitigated = next((r for r in reversed(recs) if not r.mitigated), None)

        return IndicatorResult(
            indicator_name=self.name,
            timeframe=timeframe,
            values={
                "bull_count": int(bull_cnt[-1]) if n else 0,
                "bear_count": int(bear_cnt[-1]) if n else 0,
                "bull_mitigated": int(bull_mit[-1]) if n else 0,
                "bear_mitigated": int(bear_mit[-1]) if n else 0,
                "latest_gap": latest_unmitigated,
                # per-bar continuous arrays
                "bull_count_active": bull_cnt_active,
                "bear_count_active": bear_cnt_active,
                "dist_to_nearest_bull_fvg": dist_bull_fvg,
                "dist_to_nearest_bear_fvg": dist_bear_fvg,
                "nearest_fvg_age_bars": fvg_age_bars,
            },
            signals=signals,
            metadata={
                "all_records": recs,
                "threshold_percent": threshold_percent,
                "auto_threshold": auto
            }
        )

