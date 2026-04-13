"""Risk management for trading operations"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..strategies.base import Signal
from ..api.types import Position, Order


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float = 1000.0
    max_position_pct: float = 0.1  # 10% of balance per position
    max_total_exposure: float = 0.5  # 50% of balance total
    max_daily_loss: float = 0.1  # 10% daily loss limit
    max_loss_per_trade: float = 0.05  # 5% loss per trade
    max_exposure_per_market: float = 0.2  # 20% per market
    stop_loss_pct: float = 0.1  # 10% stop loss
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = 0.15  # 15% drawdown triggers circuit breaker
    # Absolute session loss cap in USDC. When cumulative realized PnL
    # (tracked via record_trade) drops below -max_session_loss_usdc,
    # both validate_signal and calculate_position_size reject all new
    # entries. This is the "stop trading" hard floor.
    max_session_loss_usdc: float = float("inf")


class RiskManager:
    """
    Manages risk for trading operations
    
    Enforces:
    - Position sizing limits
    - Stop-loss mechanisms
    - Daily loss limits
    - Maximum exposure per market
    - Circuit breakers
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager
        
        Args:
            limits: Risk limit configuration
        """
        self.limits = limits or RiskLimits()
        self.daily_pnl: float = 0.0
        self.daily_start: datetime = datetime.now()
        self.trade_count: int = 0
        self.circuit_breaker_active: bool = False
        self.max_drawdown: float = 0.0
        self.peak_balance: float = 0.0
    
    def reset_daily(self):
        """Reset daily tracking (call at start of each day)"""
        self.daily_pnl = 0.0
        self.daily_start = datetime.now()
        self.trade_count = 0
    
    def check_circuit_breaker(self, current_balance: float) -> bool:
        """
        Check if circuit breaker should be triggered
        
        Args:
            current_balance: Current account balance
            
        Returns:
            True if circuit breaker is active
        """
        if not self.limits.circuit_breaker_enabled:
            return False
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            if drawdown >= self.limits.circuit_breaker_threshold:
                self.circuit_breaker_active = True
                return True
        
        return self.circuit_breaker_active
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)"""
        self.circuit_breaker_active = False
        self.max_drawdown = 0.0
    
    def validate_signal(
        self,
        signal: Signal,
        balance: float,
        positions: List[Position],
        orders: List[Order]
    ) -> Tuple[bool, str]:
        """
        Validate if a signal can be executed based on risk limits

        Args:
            signal: Trading signal to validate
            balance: Current account balance
            positions: Current positions
            orders: Pending orders

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker is active"

        # Check absolute session loss cap (USDC). This fires before the
        # percentage-based daily loss check so a $50 cap means $50, not
        # "10% of whatever the balance is right now".
        if self.daily_pnl <= -self.limits.max_session_loss_usdc:
            return False, f"Session loss cap reached: PnL ${self.daily_pnl:.2f} < -${self.limits.max_session_loss_usdc:.2f}"

        # Check daily loss limit (percentage of balance)
        if self.daily_pnl < -abs(self.limits.max_daily_loss * balance):
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        # Check position size limit.
        # max_position_size is a hard share cap; max_position_pct produces
        # a USDC notional that must be converted to shares via signal.price
        # so both terms in min() share the same unit.
        notional_cap = balance * self.limits.max_position_pct
        shares_from_notional = notional_cap / signal.price if signal.price > 0 else 0
        max_size = min(
            self.limits.max_position_size,
            shares_from_notional
        )
        if signal.size > max_size:
            return False, f"Position size {signal.size} exceeds limit {max_size:.1f} shares (notional cap ${notional_cap:.2f})"
        
        # Check total exposure
        total_exposure = sum(abs(pos.size * (pos.current_price or pos.average_price or 0)) for pos in positions)
        total_exposure += sum(abs(order.size * order.price) for order in orders if order.status == "PENDING")
        total_exposure += signal.size * signal.price
        
        if total_exposure > balance * self.limits.max_total_exposure:
            return False, f"Total exposure {total_exposure} exceeds limit {balance * self.limits.max_total_exposure}"
        
        # Check exposure per market
        market_exposure = sum(
            abs(pos.size * (pos.current_price or pos.average_price))
            for pos in positions
            if pos.market_id == signal.market_id
        )
        market_exposure += signal.size * signal.price
        
        if market_exposure > balance * self.limits.max_exposure_per_market:
            return False, f"Market exposure {market_exposure} exceeds limit {balance * self.limits.max_exposure_per_market}"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
        positions: List[Position],
        current_price: float
    ) -> float:
        """
        Calculate safe position size based on risk limits.

        All comparisons are in shares. The USDC-denominated limits
        (max_position_pct, max_total_exposure, max_exposure_per_market)
        are converted to share-equivalent via current_price before
        clipping.

        Args:
            signal: Trading signal
            balance: Current balance
            positions: Current positions
            current_price: Current market price

        Returns:
            Safe position size (shares), or 0 if risk limits prohibit entry.
        """
        # Session loss cap: stop sizing new entries once cumulative PnL
        # exceeds the absolute USDC loss limit.
        if self.daily_pnl <= -self.limits.max_session_loss_usdc:
            return 0.0

        # Percentage-based daily loss check
        if balance > 0 and self.daily_pnl < -abs(self.limits.max_daily_loss * balance):
            return 0.0

        # Start with signal size
        size = signal.size

        # Apply position size limit. Convert the notional cap to shares
        # so both terms have the same unit.
        notional_cap = balance * self.limits.max_position_pct
        shares_from_notional = notional_cap / current_price if current_price > 0 else 0
        max_size = min(
            self.limits.max_position_size,
            shares_from_notional
        )
        size = min(size, max_size)
        
        # Check total exposure
        total_exposure = sum(
            abs(pos.size * (pos.current_price or pos.average_price))
            for pos in positions
        )
        remaining_exposure = balance * self.limits.max_total_exposure - total_exposure
        max_by_exposure = remaining_exposure / current_price if current_price > 0 else 0
        size = min(size, max_by_exposure)
        
        # Check market exposure
        market_exposure = sum(
            abs(pos.size * (pos.current_price or pos.average_price))
            for pos in positions
            if pos.market_id == signal.market_id
        )
        remaining_market = balance * self.limits.max_exposure_per_market - market_exposure
        max_by_market = remaining_market / current_price if current_price > 0 else 0
        size = min(size, max_by_market)
        
        return max(0.0, size)
    
    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """
        Check if stop loss should be triggered
        
        Args:
            position: Position to check
            current_price: Current market price
            
        Returns:
            True if stop loss should trigger
        """
        if position.size == 0:
            return False
        
        # Calculate unrealized PnL
        if position.outcome == "YES":
            pnl_pct = (current_price - position.average_price) / position.average_price
        else:  # NO
            pnl_pct = ((1.0 - current_price) - (1.0 - position.average_price)) / (1.0 - position.average_price)
        
        # Check stop loss
        if pnl_pct < -self.limits.stop_loss_pct:
            return True
        
        return False
    
    def record_trade(self, pnl: float):
        """
        Record a completed trade for daily tracking
        
        Args:
            pnl: Profit/loss from the trade
        """
        self.daily_pnl += pnl
        self.trade_count += 1
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            "daily_pnl": self.daily_pnl,
            "trade_count": self.trade_count,
            "circuit_breaker_active": self.circuit_breaker_active,
            "max_drawdown": self.max_drawdown,
            "peak_balance": self.peak_balance
        }

