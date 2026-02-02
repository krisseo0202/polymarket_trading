"""Historical data loading for backtesting"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from ..api.types import MarketData, OrderBook, OrderBookEntry


EVENT_URL = "https://gamma-api.polymarket.com/events"
class DataLoader:
    """
    Load historical market data for backtesting
    
    In a real implementation, this would fetch data from:
    - Polymarket API historical endpoints
    - External data providers
    - Local database/cache
    """
    
    def __init__(self, data_source: str = "mock"):
        """
        Initialize data loader
        
        Args:
            data_source: Source of data ("mock", "api", "file")
        """
        self.data_source = data_source
    
    def filter_markets(self, tag: str) -> List[MarketData]:
        """
        Filter markets by tag
        """
        return self.client.get_markets(tag=tag)
    
    def load_market_data(
        self,
        token_id: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Load historical market data
        
        Args:
            token_id: Event Token ID to load
            start_date: Start date ('YYYY-MM-DD')
            end_date: End date ('YYYY-MM-DD')
            interval: Data interval  ("1m", "1h", "6h", "1d", "1w", "max")
            
        Returns:
            DataFrame with columns: timestamp, price_yes, price_no, volume
        """
        if self.data_source == "mock":
            return self._generate_mock_data(start_date, end_date, interval)
        
        params = {"market": token_id, 'interval': interval}
        
        url = "https://clob.polymarket.com/prices-history"
        response = requests.get(url, params=params)
        data = response.json()
        if not data.get('history'):
            return pd.DataFrame()
        # data = response.json()

        raise NotImplementedError("load_market_data: Only 'mock' data source is implemented.")
        return data
    
   
    
    def load_order_book_snapshots(
        self,
        token_id: str,
        timestamps: List[datetime]
    ) -> Dict[datetime, OrderBook]:
        """
        Load order book snapshots at specific timestamps
        
        Args:
            token_id: Token ID
            timestamps: List of timestamps to load
            
        Returns:
            Dictionary mapping timestamp to OrderBook
        """
        snapshots = {}
        
        for ts in timestamps:
            # In real implementation, fetch historical order book
            # For now, generate mock data
            snapshots[ts] = self._generate_mock_order_book(token_id)
        
        return snapshots
    
    def _generate_mock_order_book(self, token_id: str) -> OrderBook:
        """Generate mock order book"""
        import random
        
        mid_price = random.uniform(0.3, 0.7)
        spread = random.uniform(0.01, 0.05)
        
        bids = [
            OrderBookEntry(price=mid_price - spread/2 - i*0.01, size=random.uniform(50, 200))
            for i in range(5)
        ]
        asks = [
            OrderBookEntry(price=mid_price + spread/2 + i*0.01, size=random.uniform(50, 200))
            for i in range(5)
        ]
        
        return OrderBook(
            market_id="mock",
            token_id=token_id,
            bids=bids,
            asks=asks,
            last_price=mid_price,
            timestamp=datetime.now()
        )

