"""Polymarket API client wrapper"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams, AssetType, OpenOrderParams
from py_clob_client.order_builder.constants import BUY, SELL
from .types import MarketData, OrderBook, OrderBookEntry, Position, Order


class PolymarketClient:
    """Wrapper around py-clob-client for Polymarket API access"""
    
    def __init__(
        self,
        private_key: Optional[str] = None,
        funder_address: Optional[str] = None,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        signature_type: int = 1,
        paper_trading: bool = False
    ):
        """
        Initialize Polymarket client
        
        Args:
            private_key: Private key for signing transactions
            funder_address: Address to fund orders
            host: API host URL
            chain_id: Blockchain chain ID (137 for Polygon)
            paper_trading: If True, simulate trades without executing
        """
        self.paper_trading = paper_trading
        self.host = host
        self.chain_id = chain_id
        
        if ClobClient is None:
            raise ImportError(
                "py-clob-client is not installed. "
                "Install it with: pip install py-clob-client"
            )
        
        if not paper_trading:
            self.client = ClobClient(
                host=host,
                key=private_key or os.getenv("PRIVATE_KEY"),
                chain_id=chain_id,
                signature_type=signature_type,
                funder=funder_address or os.getenv("PROXY_FUNDER"),
            )
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
            logging.getLogger(__name__).info(f"Polymarket Client Address: {self.client.get_address()}")
        else:
            self.client = None
            self._paper_positions: Dict[str, Position] = {}
            self._paper_orders: List[Order] = []
    
    def get_markets(
        self,
        tags: Optional[List[str]] = None,
        active: bool = True
    ) -> List[MarketData]:
        """
        Fetch markets from Polymarket
        
        Args:
            tags: Filter by tags
            active: Only return active markets
            
        Returns:
            List of MarketData objects
        """
        if self.paper_trading:
            # Return mock data for paper trading
            return self._get_mock_markets()
        
        try:
            markets = self.client.get_markets()
            market_data_list = []
            
            for m in markets:
                if active and m.get('active', True) is False:
                    continue
                
                if tags and not any(tag in m.get('tags', []) for tag in tags):
                    continue
                
                outcome_tokens = {}
                if 'tokens' in m:
                    for token in m['tokens']:
                        outcome_tokens[token.get('outcome', '')] = token.get('token_id', '')
                
                market_data = MarketData(
                    market_id=m.get('id', ''),
                    question=m.get('question', ''),
                    condition_id=m.get('condition_id', ''),
                    outcome_tokens=outcome_tokens,
                    tags=m.get('tags', []),
                    volume=float(m.get('volume', 0)),
                    liquidity=float(m.get('liquidity', 0))
                )
                market_data_list.append(market_data)
            
            return market_data_list
        except Exception as e:
            raise Exception(f"Error fetching markets: {e}")
    
    def get_order_book(self, token_id: str) -> OrderBook:
        """
        Get order book for a token.
        Paper trading still reads REAL market data — only order placement is mocked.

        Args:
            token_id: Token ID to get order book for

        Returns:
            OrderBook object
        """
        if self.paper_trading:
            return self._fetch_real_order_book(token_id)

        try:
            book = self.client.get_order_book(token_id)

            bids = [
                OrderBookEntry(price=float(bid.price), size=float(bid.size))
                for bid in book.bids
            ]
            asks = [
                OrderBookEntry(price=float(ask.price), size=float(ask.size))
                for ask in book.asks
            ]
            min_order_size = int(book.min_order_size)
            tick_size = float(book.tick_size)
            return OrderBook(
                market_id="",
                token_id=token_id,
                bids=bids,
                asks=asks,
                min_order_size=min_order_size,
                tick_size=tick_size,
                last_price=float(book.last_price) if hasattr(book, 'last_price') and book.last_price else None,
                timestamp=datetime.now()
            )
        except Exception as e:
            raise Exception(f"Error fetching order book: {e}")

    def _fetch_real_order_book(self, token_id: str) -> OrderBook:
        """Fetch real order book from CLOB REST API (no auth required)."""
        last_exc = None
        for _ in range(2):
            try:
                response = requests.get(
                    f"{self.host}/book",
                    params={"token_id": token_id},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()

                bids = [
                    OrderBookEntry(
                        price=float(entry.get("price") or entry.get("p")),
                        size=float(entry.get("size") or entry.get("s")),
                    )
                    for entry in data.get("bids", [])
                ]
                asks = [
                    OrderBookEntry(
                        price=float(entry.get("price") or entry.get("p")),
                        size=float(entry.get("size") or entry.get("s")),
                    )
                    for entry in data.get("asks", [])
                ]
                return OrderBook(
                    market_id="",
                    token_id=token_id,
                    bids=bids,
                    asks=asks,
                    min_order_size=int(data.get("min_order_size", 0)),
                    tick_size=float(data.get("tick_size", 0.01)),
                    timestamp=datetime.now(),
                )
            except Exception as e:
                last_exc = e
        raise Exception(f"Error fetching real order book: {last_exc}")
    
    def get_midpoint(self, token_id):
        """Get current midpoint price or best effort"""
        try:
            mid = self.client.get_midpoint(token_id)
            return float(mid["mid"])
        except Exception as e:
            raise Exception(f"Failed to get midpoint for token {token_id}: {e}")
    
    def place_order(
        self,
        market_id: str,
        token_id: str,
        outcome: str,
        side: str,
        price: float,
        size: float
    ) -> Order:
        """
        Place a trading order
        
        Args:
            market_id: Market ID
            token_id: Token ID to trade
            outcome: "YES" or "NO"
            side: "BUY" or "SELL"
            price: Order price (0.0 to 1.0)
            size: Order size
            
        Returns:
            Order object
        """
        if self.paper_trading:
            return self._place_paper_order(market_id, token_id, outcome, side, price, size)
        
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY if side.upper() == "BUY" else SELL,
            )
            
            # create_and_post_order creates GTC (Good Till Cancel) orders by default
            # These orders stay on the book until filled or cancelled (perfect for market making)
            response = self.client.create_and_post_order(order_args)
            
            return Order(
                order_id=response['orderID'],
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                side=side.upper(),
                price=price,
                size=size,
                status=response['status'],
                timestamp=datetime.now()
            )
        except Exception as e:
            raise Exception(f"Error placing order: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if self.paper_trading:
            for order in self._paper_orders:
                if order.order_id == order_id:
                    order.status = "CANCELLED"
                    return True
            return False
        
        try:
            response = self.client.cancel(order_id)
            # cancel() may return a dict with status or just True/None
            if isinstance(response, dict):
                return response.get('status') == 'CANCELLED'
            return True  # Assume success if no exception
        except Exception as e:
            raise Exception(f"Error cancelling order: {e}")
    
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        if self.paper_trading:
            return list(self._paper_positions.values())
        
        try:
            url = "https://data-api.polymarket.com/positions"
            params = {
                "user":os.getenv("PROXY_FUNDER"),
                "sizeThreshold": 1,
                "limit": 100,
                "sortBy": "TOKENS",
                "sortDirection": "DESC"
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Handle both list and dict responses
            if isinstance(data, list):
                positions = data
            else:
                positions = data.get("positions", [])
            
            position_list = []
            for pos in positions:
                position_list.append(
                    Position(
                        market_id=pos.get("conditionId", ""),
                        token_id=pos.get("asset", ""),
                        outcome=pos.get("outcome", ""),
                        size=float(pos.get("size", 0)),
                        average_price=float(pos.get("avgPrice", 0)),
                        current_price=float(pos.get("curPrice", 0)) if pos.get("curPrice") is not None else None,
                        unrealized_pnl=float(pos.get("cashPnl", 0)) if pos.get("cashPnl") is not None else None,
                        # Add extra fields as needed, but main ones mapped above
                    )
                )
            return position_list
        except Exception as e:
            raise Exception(f"Error fetching positions: {e}")
    


    def get_open_orders(self, token_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders
        
        Args:
            token_id: Optional token ID to filter orders
            
        Returns:
            List of open order dictionaries
        """
        if self.paper_trading:
            return [
                {
                    'id': order.order_id,
                    'side': order.side,
                    'price': str(order.price),
                    'size': str(order.size),
                    'token_id': order.token_id,
                    'status': order.status
                }
                for order in self._paper_orders if order.status == "PENDING"
            ]
        
        try:
            params = OpenOrderParams()
            if token_id:
                params.token_id = token_id
            orders = self.client.get_orders(params)
            return orders if orders else []
        except Exception as e:
            raise Exception(f"Error fetching open orders: {e}")
    
    def get_recent_fills(self, token_id: str, since_ts: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get recent fills/executions for a token
        
        Args:
            token_id: Token ID to get fills for
            since_ts: Only return fills since this timestamp (if None, returns recent fills)
            
        Returns:
            List of fill dictionaries with: id, order_id, side, price, size, timestamp
        """
        if self.paper_trading:
            # Return mock fills for paper trading
            return []
        
        try:
            # TODO: Implement actual fill fetching from Polymarket API
            # The py-clob-client may not have a direct fills endpoint
            # You may need to:
            # 1. Query order history and filter for filled orders
            # 2. Use a separate fills/trades endpoint if available
            # 3. Use WebSocket for real-time fills (future enhancement)
            
            # For now, we'll try to get fills by checking order history
            # This is a placeholder - you'll need to implement based on actual API
            url = f"{self.host}/fills"
            params = {"token_id": token_id}
            if since_ts:
                params["since"] = int(since_ts.timestamp())
            
            try:
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        return data
                    return data.get("fills", [])
            except:
                # If fills endpoint doesn't exist, return empty list
                # In production, you'd implement proper fill fetching
                pass
            
            return []
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error fetching fills: {e}")
            return []
    def get_balance(self) -> float:
        """Get account balance"""
        if self.paper_trading:
            return 10000.0  # Mock balance
        
        try:
            balance = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            return float(balance['balance'])/ 1e6 if balance else 0.0
        except Exception as e:
            raise Exception(f"Error fetching balance: {e}")
    
    # Paper trading helper methods
    def _get_mock_markets(self) -> List[MarketData]:
        """Return mock markets for paper trading"""
        return [
            MarketData(
                market_id="mock_1",
                question="Will Bitcoin reach $100k by 2025?",
                condition_id="cond_1",
                outcome_tokens={"YES": "token_yes_1", "NO": "token_no_1"},
                tags=["Crypto"],
                volume=100000.0,
                liquidity=50000.0
            )
        ]
    
    def _get_mock_order_book(self, token_id: str) -> OrderBook:
        """Return mock order book for paper trading"""
        return OrderBook(
            market_id="mock_1",
            token_id=token_id,
            bids=[
                OrderBookEntry(price=0.45, size=100.0),
                OrderBookEntry(price=0.44, size=200.0),
            ],
            asks=[
                OrderBookEntry(price=0.46, size=150.0),
                OrderBookEntry(price=0.47, size=180.0),
            ],
            last_price=0.455,
            timestamp=datetime.now()
        )
    
    def _place_paper_order(
        self,
        market_id: str,
        token_id: str,
        outcome: str,
        side: str,
        price: float,
        size: float
    ) -> Order:
        """Simulate order placement for paper trading"""
        order = Order(
            order_id=f"paper_{len(self._paper_orders)}",
            market_id=market_id,
            token_id=token_id,
            outcome=outcome,
            side=side,
            price=price,
            size=size,
            status="FILLED",  # Auto-fill in paper trading
            timestamp=datetime.now()
        )
        self._paper_orders.append(order)
        
        # Update paper positions
        position_key = f"{market_id}_{outcome}"
        if position_key in self._paper_positions:
            pos = self._paper_positions[position_key]
            if side == "BUY":
                total_cost = pos.average_price * pos.size + price * size
                pos.size += size
                pos.average_price = total_cost / pos.size
            else:
                pos.size -= size
                if pos.size <= 0:
                    del self._paper_positions[position_key]
        else:
            if side == "BUY":
                self._paper_positions[position_key] = Position(
                    market_id=market_id,
                    token_id=token_id,
                    outcome=outcome,
                    size=size,
                    average_price=price
                )
        
        return order

