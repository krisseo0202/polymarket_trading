from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict
from .base import Indicator, IndicatorResult, IndicatorConfig

class TDSequentialIndicator(Indicator):
    """
    TDSequential indicator implementation.
    Computes the TDSequential for an OHLC DataFrame.
    """
    def __init__(self, config: IndicatorConfig):
        super().__init__(config)
        self.setup_len: int = self.params.get("setup_len", 9)
        self.setup_lookback: int = self.params.get("setup_lookback", 4) 
        self.countdown_len: int = self.params.get("countdown_len", 13)
        self.countdown_lookback: int = self.params.get("countdown_lookback", 2) 
    
    @staticmethod
    def _compute_tdst_levels(
        ohlc: pd.DataFrame,
        bullish_setup_complete: np.ndarray,  # buy_9 in your core
        bearish_setup_complete: np.ndarray,  # sell_9 in your core
        mode: str = "pine_like",
        return_state: bool = False,
    ) -> Dict[str, np.ndarray]:

        if mode not in {"pine_like", "dual"}:
            raise ValueError("mode must be one of {'pine_like','dual'}")
        n = len(ohlc)
        high = ohlc['high'].astype(float).to_numpy()
        low = ohlc['low'].astype(float).to_numpy()

    # Active Pine-like levels: only one side active at a time
        tdst_support = np.full(n, np.nan, dtype=float)
        tdst_resistance = np.full(n, np.nan, dtype=float)
        tdst_side = np.zeros(n, dtype=int)

        active_support = np.nan
        active_resistance = np.nan
        active_side = 0  # +1 support, -1 resistance, 0 none

        for i in range(n):
            # When buy_9 triggers: activate support, clear resistance (Pine behavior)
            if bool(bullish_setup_complete[i]):
                active_support = low[i]
                if mode == "pine_like":
                    active_resistance = np.nan
                    active_side = 1
                else:
                    active_side = 1 if np.isnan(active_resistance) else active_side

            # When sell_9 triggers: activate resistance, clear support (Pine behavior)
            if bool(bearish_setup_complete[i]):
                active_resistance = high[i]
                if mode == "pine_like":
                    active_support = np.nan
                    active_side = -1
                else:
                    active_side = -1 if np.isnan(active_support) else active_side

            # Persist forward
            tdst_support[i] = active_support
            tdst_resistance[i] = active_resistance

            if mode == "pine_like":
                tdst_side[i] = active_side
            else:
                # in dual mode we can mark side by where close is relative, but keep simple:
                tdst_side[i] = (1 if not np.isnan(active_support) else 0) + (-1 if not np.isnan(active_resistance) else 0)

        return {
            "tdst_support": tdst_support,
            "tdst_resistance": tdst_resistance,
            "tdst_side": tdst_side,
        }
              
    @staticmethod
    def _compute_td_core_part(
        ohlc: pd.DataFrame,
        setup_len: int,
        setup_lookback: int,
        countdown_len: int,
        countdown_lookback: int,
        return_state: bool = False,
    ) -> dict:
        """
        Core TD Sequential-style computation on numpy arrays.

        Returns a dict of numpy arrays:
        - bullish_setup_count, bearish_setup_count
        - bullish_setup_complete, bearish_setup_complete (bool)
        - buy_cd_count, sell_cd_count
        - buy_9, sell_9, buy_13, sell_13, buy_9_13, sell_9_13 (bool)
        """
        n = len(ohlc)
        
        # Convert to numpy arrays to avoid pandas indexing deprecation warnings
        high = ohlc['high'].astype(float).to_numpy()
        low = ohlc['low'].astype(float).to_numpy()
        close = ohlc['close'].astype(float).to_numpy()

        if n == 0:
            raise ValueError("Empty price series")
        
        # Validate lookback parameters
        if setup_lookback <= 0 or countdown_lookback <= 0:
            raise ValueError("setup_lookback and countdown_lookback must be positive")
        if setup_lookback >= n or countdown_lookback >= n:
            raise ValueError("lookback parameters must be less than data length")

        # --- 1) Setup counts (consecutive 9-style) -------------------------

        bullish_setup_count = np.zeros(n, dtype=int)
        bearish_setup_count = np.zeros(n, dtype=int)

        for i in range(n):
            if i >= setup_lookback:
                # Bullish setup: close < close[4] → weakness
                bullish_setup_count[i] = bullish_setup_count[i - 1] + 1 if close[i] < close[i - setup_lookback] else 0
                bearish_setup_count[i] = bearish_setup_count[i - 1] + 1 if close[i] > close[i - setup_lookback] else 0
            else:
                bullish_setup_count[i] = 0
                bearish_setup_count[i] = 0

        bullish_setup_complete = bullish_setup_count == setup_len
        bearish_setup_complete = bearish_setup_count == setup_len

        buy_9 = bullish_setup_complete
        sell_9 = bearish_setup_complete

        # --- 2) Countdown (non-consecutive 13-style) -----------------------

        buy_cd_count = np.zeros(n, dtype=int)
        sell_cd_count = np.zeros(n, dtype=int)

        buy_cd_active = False
        sell_cd_active = False

        last_bullish_setup_idx = None
        last_bearish_setup_idx = None

        for i in range(n):
            # Priority 1: Cancel opposite countdown when setup completes
            # Priority 2: Cancel same-type countdown when new setup starts
            # Priority 3: Activate countdown when setup completes
            # Priority 4: Increment countdown if active
            
            # Check for new setup start (transition from 0 to >0) - cancel same-type countdown
            # Note: This handles the case where a new setup starts after a previous one completed
            # and countdown was active. The countdown will be cancelled here, then reactivated
            # below if this new setup also completes.
            if i > 0:
                # New bullish setup started - cancel buy countdown
                if bullish_setup_count[i-1] == 0 and bullish_setup_count[i] > 0:
                    buy_cd_active = False
                    buy_cd_count[i] = 0
                
                # New bearish setup started - cancel sell countdown
                if bearish_setup_count[i-1] == 0 and bearish_setup_count[i] > 0:
                    sell_cd_active = False
                    sell_cd_count[i] = 0
            
            # Start / reset BUY countdown on new bullish setup completion
            if bullish_setup_complete[i]:
                buy_cd_active = True
                buy_cd_count[i] = 0  # Reset countdown to 0 when new setup completes
                last_bullish_setup_idx = i

                # Cancel any active SELL countdown
                sell_cd_active = False
                sell_cd_count[i] = 0

            # Start / reset SELL countdown on new bearish setup completion
            if bearish_setup_complete[i]:
                sell_cd_active = True
                sell_cd_count[i] = 0  # Reset countdown to 0 when new setup completes
                last_bearish_setup_idx = i

                # Cancel any active BUY countdown
                buy_cd_active = False
                buy_cd_count[i] = 0

            # BUY countdown: after bullish setup, non-consecutive closes <= low[2]
            if buy_cd_active:
                if last_bullish_setup_idx is not None and i > last_bullish_setup_idx:
                    # Only increment after setup completion bar
                    if i >= countdown_lookback:
                        buy_cd_count[i] = buy_cd_count[i - 1] + 1 if close[i] <= low[i - countdown_lookback] else buy_cd_count[i - 1]
                    else:
                        # Not enough bars for lookback - keep at 0
                        buy_cd_count[i] = 0
                else:
                    # Setup bar itself - keep at 0
                    buy_cd_count[i] = 0
            else:
                # Countdown not active - reset to 0 (don't carry forward old values)
                buy_cd_count[i] = 0

            # SELL countdown: after bearish setup, non-consecutive closes >= high[2]
            if sell_cd_active:
                if last_bearish_setup_idx is not None and i > last_bearish_setup_idx:
                    # Only increment after setup completion bar
                    if i >= countdown_lookback:
                        sell_cd_count[i] = sell_cd_count[i - 1] + 1 if close[i] >= high[i - countdown_lookback] else sell_cd_count[i - 1]
                    else:
                        # Not enough bars for lookback - keep at 0
                        sell_cd_count[i] = 0
                else:
                    # Setup bar itself - keep at 0
                    sell_cd_count[i] = 0
            else:
                # Countdown not active - reset to 0 (don't carry forward old values)
                sell_cd_count[i] = 0

            # Optional: stop countdown once 13 reached
            # if buy_cd_count[i] >= countdown_len:
            #     buy_cd_active = False
            # if sell_cd_count[i] >= countdown_len:
            #     sell_cd_active = False

        buy_13 = buy_cd_count == countdown_len
        sell_13 = sell_cd_count == countdown_len
        if return_state:
            return {
                "bullish_setup_count": bullish_setup_count,
                "bearish_setup_count": bearish_setup_count,
                "bullish_setup_complete": bullish_setup_complete,
                "bearish_setup_complete": bearish_setup_complete,
                "buy_cd_count": buy_cd_count,
                "sell_cd_count": sell_cd_count,
                "buy_9": buy_9,
                "sell_9": sell_9,
                "buy_13": buy_13,
                "sell_13": sell_13,
            }
        else:
            return {
                "buy_9": buy_9,
                "sell_9": sell_9,
                "buy_13": buy_13,
                "sell_13": sell_13,
            }

 
        
    
    def _bars_since(self,cond: pd.Series | np.ndarray) -> np.ndarray:
        """
        Pine: barssince(condition)
        Returns number of bars since condition was last true.
        - 0 on the bar where cond is true
        - increments by 1 on following bars
        - NaN before it is ever true
        """
        cond = np.asarray(cond, dtype=bool)
        n = len(cond)
        out = np.full(n, np.nan, dtype=float)
        last_true = None

        for i in range(n):
            if cond[i]:
                last_true = i
                out[i] = 0.0
            elif last_true is None:
                out[i] = np.nan
            else:
                out[i] = float(i - last_true)
        return out

    @staticmethod
    def _value_when_last(cond: pd.Series | np.ndarray,
                         series: pd.Series | np.ndarray) -> np.ndarray:
        """
        Pine: valuewhen(condition, series, 0)
        Returns the value of the series when the condition was last true.
        - NaN if condition was never true
        - the value of the series when the condition was last true
        """
        cond = np.asarray(cond, dtype=bool)
        series = np.asarray(series, dtype=float)
        n = len(series)
        out = np.full(n, np.nan, dtype=float)
        last_val = np.nan
        for i in range(n):
            if cond[i]:
                last_val = series[i]
            out[i] = last_val
        
        return out
        
    def compute(self, ohlc: pd.DataFrame, timeframe: str = "unknown") -> "IndicatorResult":
        td_result = self._compute_td_core_part(ohlc,
                            setup_len=self.setup_len,
                            setup_lookback=self.setup_lookback,
                            countdown_len=self.countdown_len,
                            countdown_lookback=self.countdown_lookback,
                            return_state=True  # Need full state for visualization
                        )
        td_st_result = self._compute_tdst_levels(ohlc,
                            bullish_setup_complete=td_result['buy_9'],
                            bearish_setup_complete=td_result['sell_9'],
                            mode="pine_like",
                            return_state=False
                        )
        td_result.update(td_st_result)
        
        # Calculate TD Sequential sequence numbers for visualization
        # TDUp = Sell Setup = counts strength (bearish_setup_count) = Red triangles down (above bars)
        # TDDn = Buy Setup = counts weakness (bullish_setup_count) = Green triangles up (below bars)
        
        bullish_setup_count = td_result['bullish_setup_count']
        bearish_setup_count = td_result['bearish_setup_count']
        buy_cd_count = td_result['buy_cd_count']
        sell_cd_count = td_result['sell_cd_count']
        bullish_setup_complete = td_result['bullish_setup_complete']
        bearish_setup_complete = td_result['bearish_setup_complete']
        
        # TDUp: Sell Setup (bearish_setup_count) - Red triangles down above bars
        # VALIDATION: Only show valid sequences
        # Setup numbers: Show when setup_count > 0 (already consecutive from core computation)
        # Countdown: Only show after setup completes (9) and before new setup starts
        td_up = np.zeros(len(ohlc), dtype=int)
        countdown_allowed = False
        
        for i in range(len(ohlc)):
            current_setup = int(bearish_setup_count[i])
            prev_setup = int(bearish_setup_count[i-1]) if i > 0 else 0
            
            # Priority 1: If setup count is active, show it
            if current_setup > 0:
                td_up[i] = current_setup
                # When setup completes (reaches 9), allow countdown to start on next bar
                if current_setup == self.setup_len:
                    countdown_allowed = True  # Will be used on next iteration
                else:
                    # Setup is active but not complete - countdown not allowed
                    countdown_allowed = False
            # Priority 2: Show countdown ONLY if:
            # - Setup is 0 on current bar (no active setup)
            # - Previous bar had setup 9 (we just completed setup)
            # - Countdown value is valid and > 0
            elif prev_setup == self.setup_len and sell_cd_count[i] > 0 and sell_cd_count[i] <= self.countdown_len:
                # We're in countdown phase (setup just completed on previous bar)
                td_up[i] = 9 + int(sell_cd_count[i])  # 10-13 (A-D)
                countdown_allowed = True  # Keep countdown allowed if continuing
            elif countdown_allowed and sell_cd_count[i] > 0 and sell_cd_count[i] <= self.countdown_len:
                # Continue countdown (previous bar had countdown, this bar continues it)
                td_up[i] = 9 + int(sell_cd_count[i])  # 10-13 (A-D)
            else:
                # No setup, no valid countdown - reset
                countdown_allowed = False
        
        # TDDn: Buy Setup (bullish_setup_count) - Green triangles up below bars  
        # Same validation logic as TDUp
        td_dn = np.zeros(len(ohlc), dtype=int)
        buy_countdown_allowed = False
        
        for i in range(len(ohlc)):
            current_setup = int(bullish_setup_count[i])
            prev_setup = int(bullish_setup_count[i-1]) if i > 0 else 0
            
            # Priority 1: If setup count is active, show it
            if current_setup > 0:
                td_dn[i] = current_setup
                # When setup completes (reaches 9), allow countdown to start on next bar
                if current_setup == self.setup_len:
                    buy_countdown_allowed = True  # Will be used on next iteration
                else:
                    # Setup is active but not complete - countdown not allowed
                    buy_countdown_allowed = False
            # Priority 2: Show countdown ONLY if all conditions are met
            elif prev_setup == self.setup_len and buy_cd_count[i] > 0 and buy_cd_count[i] <= self.countdown_len:
                # We're in countdown phase (setup just completed on previous bar)
                td_dn[i] = 9 + int(buy_cd_count[i])  # 10-13 (A-D)
                buy_countdown_allowed = True  # Keep countdown allowed if continuing
            elif buy_countdown_allowed and buy_cd_count[i] > 0 and buy_cd_count[i] <= self.countdown_len:
                # Continue countdown (previous bar had countdown, this bar continues it)
                td_dn[i] = 9 + int(buy_cd_count[i])  # 10-13 (A-D)
            else:
                # No setup, no valid countdown - reset
                buy_countdown_allowed = False
        
        td_result['td_up'] = td_up
        td_result['td_dn'] = td_dn

        # Continuous distance features
        close_arr = ohlc['close'].astype(float).to_numpy()
        tdst_support    = td_st_result['tdst_support']
        tdst_resistance = td_st_result['tdst_resistance']

        # (close - support) / close; positive when price is above support
        dist_support = np.where(
            np.isnan(tdst_support),
            np.nan,
            np.clip((close_arr - tdst_support) / close_arr, -0.20, 0.20),
        )
        # (resistance - close) / close; positive when price is below resistance
        dist_resistance = np.where(
            np.isnan(tdst_resistance),
            np.nan,
            np.clip((tdst_resistance - close_arr) / close_arr, -0.20, 0.20),
        )

        td_result['dist_to_tdst_support']    = dist_support
        td_result['dist_to_tdst_resistance'] = dist_resistance

        return IndicatorResult(
            indicator_name="TDSequential",
            timeframe=timeframe,
            values=td_result,
            signals=[],
            metadata={"ohlc": ohlc}
        )