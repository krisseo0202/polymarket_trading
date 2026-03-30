"""Data models and types for Polymarket API"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

Side = Literal["BUY", "SELL"]


@dataclass
class MarketData:
    """Market information"""
    market_id: str
    question: str
    condition_id: str
    outcome_tokens: Dict[str, str]  # outcome -> token_id
    end_date: Optional[datetime] = None
    tags: List[str] = None
    volume: float = 0.0
    liquidity: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class OrderBookEntry:
    """Single order book entry"""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book for a market"""
    market_id: str
    token_id: str
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    last_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    min_order_size: Optional[int] = None
    tick_size: Optional[float] = None


@dataclass
class Position:
    """User position in a market"""
    market_id: str
    token_id: str
    outcome: str  # "YES" or "NO"
    size: float
    average_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None


@dataclass
class Order:
    """Trading order"""
    order_id: Optional[str] = None
    market_id: str = ""
    token_id: str = ""
    outcome: str = ""  # "YES" or "NO"
    side: str = ""  # "BUY" or "SELL"
    price: float = 0.0
    size: float = 0.0
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
    timestamp: Optional[datetime] = None
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    """Represents a filled order"""
    order_id: str
    side: Side
    price: float
    size: float
    timestamp: datetime
    token_id: str

