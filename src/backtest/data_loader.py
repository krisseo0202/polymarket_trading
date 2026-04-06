"""Historical data loading for backtesting"""

import os
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from ..api.types import MarketData, OrderBook, OrderBookEntry


EVENT_URL = "https://gamma-api.polymarket.com/events"
COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# Default paths relative to project root
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_DEFAULT_HISTORY_CSV = os.path.join(_DATA_DIR, "btc_updown_5m.csv")
_DEFAULT_BTC_1M_CACHE = os.path.join(_DATA_DIR, "btc_1m_cache.csv")


class DataLoader:
    """Load historical market data for backtesting."""

    def __init__(self, data_source: str = "mock"):
        """
        Args:
            data_source: "mock" for synthetic data, "file" for CSV-based real data
        """
        self.data_source = data_source

    def load_market_data(
        self,
        token_id: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Load historical market data.

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

        if self.data_source == "file":
            return self._load_from_csv(start_date, end_date)

        # Fallback: try CLOB prices-history (sparse, not useful for intra-window)
        params = {"market": token_id, "interval": interval}
        url = "https://clob.polymarket.com/prices-history"
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if not data.get("history"):
            return pd.DataFrame()

        history = data["history"]
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
        df["price_yes"] = df["p"].astype(float)
        df["price_no"] = 1.0 - df["price_yes"]
        df["volume"] = 0.0
        return df[["timestamp", "price_yes", "price_no", "volume"]]

    def _load_from_csv(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load signal backtest data by joining slot outcomes with BTC 1m candles.

        Returns one row per resolved slot with BTC OHLC context and outcome.
        This is used by the backtester for signal-only evaluation.
        """
        slots = self.load_btc_updown_history()
        slots = slots.dropna(subset=["outcome"])
        slots = slots[slots["outcome"].isin(["Up", "Down"])].copy()

        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")
        slots = slots[(slots["slot_utc"] >= start) & (slots["slot_utc"] <= end)]

        if slots.empty:
            return pd.DataFrame()

        up_won = slots["outcome"] == "Up"
        return pd.DataFrame({
            "timestamp": slots["slot_utc"].values,
            "slot_ts": slots["slot_ts"].astype(int).values,
            "price_yes": up_won.astype(float).values,
            "price_no": (~up_won).astype(float).values,
            "volume": slots["volume"].fillna(0.0).values,
            "outcome": slots["outcome"].values,
        })

    def load_btc_1m_candles(
        self, start_ts: int, end_ts: int, use_cache: bool = True
    ) -> pd.DataFrame:
        """Load BTC 1-minute candles, fetching from Coinbase if not cached.

        Args:
            start_ts: Start unix timestamp
            end_ts: End unix timestamp
            use_cache: If True, try loading from btc_1m_cache.csv first

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close
        """
        if use_cache and os.path.exists(_DEFAULT_BTC_1M_CACHE):
            df = pd.read_csv(_DEFAULT_BTC_1M_CACHE, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            t_start = pd.Timestamp(start_ts, unit="s", tz="UTC")
            t_end = pd.Timestamp(end_ts, unit="s", tz="UTC")
            subset = df[(df.index >= t_start) & (df.index <= t_end)]
            if not subset.empty:
                return subset

        return self._fetch_coinbase_1m(start_ts, end_ts)

    @staticmethod
    def _fetch_coinbase_1m(start_ts: int, end_ts: int) -> pd.DataFrame:
        """Fetch 1-minute candles from Coinbase Exchange API."""
        from datetime import timezone
        CHUNK_SEC = 300 * 60  # 300 bars max per request
        chunks = []
        cursor = start_ts
        while cursor < end_ts:
            chunk_end = min(cursor + CHUNK_SEC, end_ts)
            start_iso = datetime.fromtimestamp(cursor, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = datetime.fromtimestamp(chunk_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                r = requests.get(
                    COINBASE_URL,
                    params={"granularity": 60, "start": start_iso, "end": end_iso},
                    timeout=15,
                )
                r.raise_for_status()
                raw = r.json()
                if raw:
                    df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
                    df.sort_values("timestamp", inplace=True)
                    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                    for c in ("open", "high", "low", "close"):
                        df[c] = df[c].astype(float)
                    chunks.append(df[["open", "high", "low", "close"]])
            except requests.RequestException:
                pass
            cursor = chunk_end
            if cursor < end_ts:
                time.sleep(0.35)

        if not chunks:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        combined = pd.concat(chunks).sort_index()
        return combined[~combined.index.duplicated(keep="last")]

    def load_btc_updown_history(
        self, csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Load collected BTC Up/Down 5-min history from CSV.

        Args:
            csv_path: Path to CSV file. Defaults to data/btc_updown_5m.csv

        Returns:
            DataFrame with columns: slot_ts, slot_utc, question, up_token,
            down_token, outcome, volume, up_price_start, up_price_end,
            down_price_start, down_price_end, strike_price
        """
        path = csv_path or _DEFAULT_HISTORY_CSV
        df = pd.read_csv(path, parse_dates=["slot_utc"])
        # Coerce numeric columns
        for col in [
            "volume", "up_price_start", "up_price_end",
            "down_price_start", "down_price_end", "strike_price",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

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

