"""Trading strategies module"""

from .base import Strategy, Signal
from .btc_updown import BTCUpDownStrategy
from .btc_updown_xgb import BTCUpDownXGBStrategy
from .btc_vol_reversion import BTCVolatilityReversionStrategy
from .coin_toss import CoinTossStrategy
from .prob_edge import ProbEdgeStrategy
from .td_rsi import TDRSIStrategy

__all__ = [
    "Strategy",
    "Signal",
    "BTCUpDownStrategy",
    "BTCUpDownXGBStrategy",
    "BTCVolatilityReversionStrategy",
    "CoinTossStrategy",
    "ProbEdgeStrategy",
    "TDRSIStrategy",
]

