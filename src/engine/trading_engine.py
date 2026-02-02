"""Main trading engine for executing strategies"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..api.client import PolymarketClient
from ..api.types import MarketData, OrderBook, Position, Order
from ..strategies.base import Strategy, Signal
from .risk_manager import RiskManager, RiskLimits


class TradingEngine:
    """
    Main trading engine that orchestrates strategy execution
    
    Responsibilities:
    - Manage active strategies
    - Execute signals from strategies
    - Integrate with risk manager
    - Handle order lifecycle
    - Support live and paper trading
    """
    
    def __init__(
        self,
        client: PolymarketClient,
        risk_manager: Optional[RiskManager] = None,
        paper_trading: bool = False
    ):
        """
        Initialize trading engine
        
        Args:
            client: Polymarket API client
            risk_manager: Risk manager instance
            paper_trading: If True, simulate trades without executing
        """
        self.client = client
        self.paper_trading = paper_trading
        self.risk_manager = risk_manager or RiskManager()
        self.strategies: List[Strategy] = []
        self.active_orders: Dict[str, Order] = {}
        self.positions: List[Position] = []
        self.running = False
        self.price_history: Dict[str, List[float]] = {}  # token_id -> price history
    
    def add_strategy(self, strategy: Strategy):
        """Add a strategy to the engine"""
        self.strategies.append(strategy)
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy by name"""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
    
    def start(self, interval: int = 60):
        """
        Start the trading engine loop
        
        Args:
            interval: Seconds between each trading cycle
        """
        self.running = True
        self.risk_manager.reset_daily()
        
        while self.running:
            try:
                self._trading_cycle()
                time.sleep(interval)
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                print(f"Error in trading cycle: {e}")
                time.sleep(interval)
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        # Cancel all pending orders
        for order_id, order in list(self.active_orders.items()):
            if order.status == "PENDING":
                try:
                    self.client.cancel_order(order_id)
                except:
                    pass
    
    def _trading_cycle(self):
        """Execute one trading cycle"""
        # Update positions and orders
        self._update_positions()
        self._update_orders()
        
        # Check circuit breaker
        balance = self.client.get_balance()
        if self.risk_manager.check_circuit_breaker(balance):
            print("Circuit breaker active - pausing trading")
            return
        
        # Fetch market data
        markets = self.client.get_markets(active=True)
        if not markets:
            return
        
        # Get order books for all markets
        order_books = {}
        for market in markets:
            for outcome, token_id in market.outcome_tokens.items():
                try:
                    book = self.client.get_order_book(token_id)
                    order_books[token_id] = book
                    
                    # Update price history
                    if book.last_price:
                        if token_id not in self.price_history:
                            self.price_history[token_id] = []
                        self.price_history[token_id].append(book.last_price)
                        # Keep only last 100 prices
                        if len(self.price_history[token_id]) > 100:
                            self.price_history[token_id] = self.price_history[token_id][-100:]
                except Exception as e:
                    print(f"Error fetching order book for {token_id}: {e}")
        
        # Prepare market data for strategies
        market_data = {
            "markets": markets,
            "order_books": order_books,
            "price_history": self.price_history,
            "positions": self.positions,
            "balance": balance
        }
        
        # Run all strategies
        all_signals: List[Signal] = []
        for strategy in self.strategies:
            try:
                signals = strategy.analyze(market_data)
                for signal in signals:
                    if strategy.should_enter(signal):
                        all_signals.append(signal)
            except Exception as e:
                print(f"Error in strategy {strategy.name}: {e}")
        
        # Execute signals
        for signal in all_signals:
            self._execute_signal(signal, market_data)
        
        # Check for exit signals
        for position in self.positions:
            for strategy in self.strategies:
                try:
                    if strategy.should_exit(
                        {
                            "market_id": position.market_id,
                            "outcome": position.outcome,
                            "size": position.size,
                            "average_price": position.average_price
                        },
                        market_data
                    ):
                        self._exit_position(position, market_data)
                        break
                except Exception as e:
                    print(f"Error checking exit for {strategy.name}: {e}")
    
    def _execute_signal(self, signal: Signal, market_data: Dict[str, Any]):
        """Execute a trading signal"""
        # Validate with risk manager
        is_valid, reason = self.risk_manager.validate_signal(
            signal,
            market_data["balance"],
            self.positions,
            list(self.active_orders.values())
        )
        
        if not is_valid:
            print(f"Signal rejected: {reason}")
            return
        
        # Get market and token info
        market = next((m for m in market_data["markets"] if m.market_id == signal.market_id), None)
        if not market:
            print(f"Market {signal.market_id} not found")
            return
        
        token_id = market.outcome_tokens.get(signal.outcome)
        if not token_id:
            print(f"Token for {signal.outcome} not found in market {signal.market_id}")
            return
        
        # Get order book for price
        order_book = market_data["order_books"].get(token_id)
        if not order_book:
            print(f"Order book not available for {token_id}")
            return
        
        # Calculate safe position size
        current_price = order_book.asks[0].price if signal.action == "BUY" else order_book.bids[0].price
        safe_size = self.risk_manager.calculate_position_size(
            signal,
            market_data["balance"],
            self.positions,
            current_price
        )
        
        if safe_size <= 0:
            print(f"Position size too small: {safe_size}")
            return
        
        # Adjust signal price to market price
        execution_price = order_book.asks[0].price if signal.action == "BUY" else order_book.bids[0].price
        
        # Place order
        try:
            order = self.client.place_order(
                market_id=signal.market_id,
                token_id=token_id,
                outcome=signal.outcome,
                side=signal.action,
                price=execution_price,
                size=safe_size
            )
            
            self.active_orders[order.order_id] = order
            print(f"Placed {signal.action} order: {safe_size} @ {execution_price} for {signal.outcome} in {signal.market_id}")
        except Exception as e:
            print(f"Error placing order: {e}")
    
    def _exit_position(self, position: Position, market_data: Dict[str, Any]):
        """Exit a position"""
        # Get market info
        market = next((m for m in market_data["markets"] if m.market_id == position.market_id), None)
        if not market:
            return
        
        token_id = market.outcome_tokens.get(position.outcome)
        if not token_id:
            return
        
        order_book = market_data["order_books"].get(token_id)
        if not order_book or not order_book.bids:
            return
        
        # Place sell order
        try:
            order = self.client.place_order(
                market_id=position.market_id,
                token_id=token_id,
                outcome=position.outcome,
                side="SELL",
                price=order_book.bids[0].price,
                size=position.size
            )
            
            self.active_orders[order.order_id] = order
            print(f"Exiting position: {position.size} @ {order_book.bids[0].price}")
        except Exception as e:
            print(f"Error exiting position: {e}")
    
    def _update_positions(self):
        """Update current positions"""
        try:
            self.positions = self.client.get_positions()
        except Exception as e:
            print(f"Error updating positions: {e}")
    
    def _update_orders(self):
        """Update order status"""
        # In a real implementation, you'd fetch order status from the API
        # For now, we'll assume orders are filled immediately in paper trading
        if self.paper_trading:
            # Mark all pending orders as filled after a delay
            for order_id, order in list(self.active_orders.items()):
                if order.status == "PENDING":
                    # Simulate order fill
                    order.status = "FILLED"
                    # Update positions would happen automatically in paper trading mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "running": self.running,
            "strategies": [s.name for s in self.strategies],
            "active_orders": len([o for o in self.active_orders.values() if o.status == "PENDING"]),
            "positions": len(self.positions),
            "balance": self.client.get_balance(),
            "risk_metrics": self.risk_manager.get_risk_metrics()
        }

