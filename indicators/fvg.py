"""FVG (Fair Value Gap) Indicator Implementation"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .base import (
    Indicator,
    IndicatorConfig,
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
        if n < 3:
            return [], np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int)

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

        return records, bull_count, bear_count, bull_mitigated, bear_mitigated

    def compute(self, ohlc: pd.DataFrame, timeframe: str = "unknown") -> IndicatorResult:
        self.validate_ohlc(ohlc)

        threshold_percent = self.params.get("threshold_percent", 0.0)
        auto = self.params.get("auto", False)

        recs, bull_cnt, bear_cnt, bull_mit, bear_mit = self._detect_fvgs_with_counts(
            ohlc, threshold_percent, auto
        )

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
                "latest_gap": latest_unmitigated
            },
            signals=signals,
            metadata={
                "all_records": recs,
                "threshold_percent": threshold_percent,
                "auto_threshold": auto
            }
        )

