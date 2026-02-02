"""Backtesting framework for trading strategies"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..strategies.base import Strategy
from ..api.types import MarketData, OrderBook, OrderBookEntry
from .data_loader import DataLoader
from ..engine.risk_manager import RiskManager, RiskLimits


class BacktestResult:
    """Results from a backtest"""
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        self.initial_balance: float = 0.0
        self.final_balance: float = 0.0
        self.total_return: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.win_rate: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.equity_curve:
            return
        
        # Total return
        if self.initial_balance > 0:
            self.total_return = (self.final_balance - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (annualized)
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Assume daily returns, annualize
                annualized_return = returns.mean() * 252
                annualized_std = returns.std() * np.sqrt(252)
                self.sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        self.max_drawdown = abs(drawdown.min())
        
        # Win rate
        self.total_trades = len(self.trades)
        self.winning_trades = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        self.losing_trades = sum(1 for t in self.trades if t.get("pnl", 0) < 0)
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
    
    def print_summary(self):
        """Print backtest summary"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.final_balance:,.2f}")
        print(f"Total Return: {self.total_return:.2%}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print("="*50 + "\n")


class Backtester:
    """
    Backtesting framework for trading strategies
    
    Simulates strategy execution on historical data
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        data_loader: Optional[DataLoader] = None,
        risk_limits: Optional[RiskLimits] = None
    ):
        """
        Initialize backtester
        
        Args:
            initial_balance: Starting balance for backtest
            data_loader: Data loader instance
            risk_limits: Risk limits to apply
        """
        self.initial_balance = initial_balance
        self.data_loader = data_loader or DataLoader()
        self.risk_manager = RiskManager(risk_limits)
    
    def run(
        self,
        strategy: Strategy,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h"
    ) -> BacktestResult:
        """
        Run backtest for a strategy
        
        Args:
            strategy: Strategy to backtest
            market_id: Market to test on
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            BacktestResult with performance metrics
        """
        result = BacktestResult()
        result.initial_balance = self.initial_balance
        
        # Load historical data
        data = self.data_loader.load_market_data(market_id, start_date, end_date, interval)
        
        if data.empty:
            print("No data available for backtest")
            return result
        
        # Initialize state
        balance = self.initial_balance
        positions: Dict[str, Dict[str, Any]] = {}  # token_id -> position info
        price_history: Dict[str, List[float]] = {}
        
        # Simulate trading
        for idx, row in data.iterrows():
            timestamp = row["timestamp"]
            price_yes = row["price_yes"]
            price_no = row["price_no"]
            
            # Update price history
            yes_token = f"{market_id}_YES"
            no_token = f"{market_id}_NO"
            
            if yes_token not in price_history:
                price_history[yes_token] = []
            if no_token not in price_history:
                price_history[no_token] = []
            
            price_history[yes_token].append(price_yes)
            price_history[no_token].append(price_no)
            
            # Create market data for strategy
            market_data = self._create_market_data(
                market_id, yes_token, no_token, price_yes, price_no, price_history
            )
            
            # Run strategy
            try:
                signals = strategy.analyze(market_data)
                
                for signal in signals:
                    if strategy.should_enter(signal):
                        # Execute signal
                        trade = self._execute_signal(
                            signal, yes_token, no_token, price_yes, price_no,
                            balance, positions, timestamp
                        )
                        if trade:
                            result.trades.append(trade)
                            balance = trade["balance_after"]
            except Exception as e:
                print(f"Error in strategy at {timestamp}: {e}")
            
            # Check for exits
            for token_id, position in list(positions.items()):
                current_price = price_yes if "YES" in token_id else price_no
                
                exit_signal = strategy.should_exit(
                    position,
                    market_data
                )
                
                if exit_signal:
                    trade = self._exit_position(
                        token_id, position, current_price, balance, timestamp
                    )
                    if trade:
                        result.trades.append(trade)
                        balance = trade["balance_after"]
                        del positions[token_id]
            
            # Update equity curve
            equity = balance
            for pos in positions.values():
                # Add unrealized PnL
                current_price = price_yes if "YES" in pos["token_id"] else price_no
                if pos["outcome"] == "YES":
                    pnl = (current_price - pos["entry_price"]) * pos["size"]
                else:
                    pnl = ((1.0 - current_price) - (1.0 - pos["entry_price"])) * pos["size"]
                equity += pnl
            
            result.equity_curve.append(equity)
            result.timestamps.append(timestamp)
        
        # Calculate final balance
        result.final_balance = balance
        for pos in positions.values():
            # Close remaining positions at last price
            last_price = data.iloc[-1]["price_yes"] if "YES" in pos["token_id"] else data.iloc[-1]["price_no"]
            if pos["outcome"] == "YES":
                pnl = (last_price - pos["entry_price"]) * pos["size"]
            else:
                pnl = ((1.0 - last_price) - (1.0 - pos["entry_price"])) * pos["size"]
            result.final_balance += pnl
        
        # Calculate metrics
        result.calculate_metrics()
        
        return result
    
    def _create_market_data(
        self,
        market_id: str,
        yes_token: str,
        no_token: str,
        price_yes: float,
        price_no: float,
        price_history: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Create market data dictionary for strategy"""
        # Create mock order book
        def create_order_book(token_id: str, price: float) -> Dict[str, Any]:
            spread = 0.01
            bids = [{"price": price - spread/2 - i*0.01, "size": 100.0} for i in range(3)]
            asks = [{"price": price + spread/2 + i*0.01, "size": 100.0} for i in range(3)]
            return {"bids": bids, "asks": asks, "last_price": price}
        
        market = MarketData(
            market_id=market_id,
            question="Test Market",
            condition_id="test_condition",
            outcome_tokens={"YES": yes_token, "NO": no_token}
        )
        
        return {
            "markets": [market],
            "order_books": {
                yes_token: create_order_book(yes_token, price_yes),
                no_token: create_order_book(no_token, price_no)
            },
            "price_history": price_history,
            "balance": self.initial_balance
        }
    
    def _execute_signal(
        self,
        signal,
        yes_token: str,
        no_token: str,
        price_yes: float,
        price_no: float,
        balance: float,
        positions: Dict[str, Dict[str, Any]],
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Execute a trading signal in backtest"""
        token_id = yes_token if signal.outcome == "YES" else no_token
        current_price = price_yes if signal.outcome == "YES" else price_no
        
        # Check risk limits
        is_valid, reason = self.risk_manager.validate_signal(
            signal, balance, [], []
        )
        if not is_valid:
            return None
        
        # Calculate position size
        size = self.risk_manager.calculate_position_size(
            signal, balance, [], current_price
        )
        
        if size <= 0 or size * current_price > balance:
            return None
        
        # Execute trade
        cost = size * current_price
        balance_after = balance - cost
        
        # Record position
        positions[token_id] = {
            "token_id": token_id,
            "outcome": signal.outcome,
            "size": size,
            "entry_price": current_price,
            "entry_time": timestamp
        }
        
        return {
            "timestamp": timestamp,
            "action": signal.action,
            "outcome": signal.outcome,
            "size": size,
            "price": current_price,
            "cost": cost,
            "balance_before": balance,
            "balance_after": balance_after,
            "pnl": 0.0  # Will be calculated on exit
        }
    
    def _exit_position(
        self,
        token_id: str,
        position: Dict[str, Any],
        current_price: float,
        balance: float,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Exit a position in backtest"""
        entry_price = position["entry_price"]
        size = position["size"]
        outcome = position["outcome"]
        
        # Calculate PnL
        if outcome == "YES":
            pnl = (current_price - entry_price) * size
        else:
            pnl = ((1.0 - current_price) - (1.0 - entry_price)) * size
        
        balance_after = balance + size * current_price
        
        self.risk_manager.record_trade(pnl)
        
        return {
            "timestamp": timestamp,
            "action": "SELL",
            "outcome": outcome,
            "size": size,
            "price": current_price,
            "entry_price": entry_price,
            "pnl": pnl,
            "balance_before": balance,
            "balance_after": balance_after
        }

