"""Arbitrage trading strategies"""

from typing import List, Dict, Any
from .base import Strategy, Signal


class ArbitrageStrategy(Strategy):
    """
    Arbitrage strategy that looks for price discrepancies
    
    Strategies:
    - Cross-market arbitrage: Same question, different prices
    - Three-way arbitrage: YES + NO prices don't sum to 1.0
    - Cross-platform arbitrage: Price differences across platforms
    """
    
    def __init__(self, name: str = "Arbitrage", config: Dict[str, Any] = None):
        if config is None:
            config = {}
        config.setdefault("min_profit_threshold", 0.01)  # 1% minimum profit
        config.setdefault("max_spread", 0.05)  # 5% max spread to consider
        super().__init__(name, config)
    
    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Analyze markets for arbitrage opportunities
        
        Args:
            market_data: Should contain 'markets' list and 'order_books' dict
            
        Returns:
            List of arbitrage signals
        """
        signals = []
        
        if not self.enabled:
            return signals
        
        markets = market_data.get("markets", [])
        order_books = market_data.get("order_books", {})
        
        # Strategy 1: Three-way arbitrage (YES + NO should sum to ~1.0)
        for market in markets:
            market_id = market.market_id
            yes_token = market.outcome_tokens.get("YES")
            no_token = market.outcome_tokens.get("NO")
            
            if not yes_token or not no_token:
                continue
            
            yes_book = order_books.get(yes_token)
            no_book = order_books.get(no_token)
            
            if not yes_book or not no_book:
                continue
            
            # Get best bid prices (what we can sell at)
            if yes_book.bids and no_book.bids:
                yes_bid = yes_book.bids[0].price
                no_bid = no_book.bids[0].price
                total_bid = yes_bid + no_bid
                
                # If we can sell both for more than 1.0, arbitrage opportunity
                if total_bid > 1.0 + self.config["min_profit_threshold"]:
                    profit = total_bid - 1.0
                    confidence = min(1.0, profit / 0.1)  # Higher profit = higher confidence
                    
                    # Signal to buy both outcomes (to then sell at higher price)
                    if yes_bid >= no_bid:
                        signals.append(Signal(
                            market_id=market_id,
                            outcome="YES",
                            action="BUY",
                            confidence=confidence,
                            price=yes_book.asks[0].price if yes_book.asks else 0.5,
                            size=self.max_position_size,
                            reason=f"Three-way arbitrage: total bid {total_bid:.4f} > 1.0"
                        ))
            
            # Get best ask prices (what we can buy at)
            if yes_book.asks and no_book.asks:
                yes_ask = yes_book.asks[0].price
                no_ask = no_book.asks[0].price
                total_ask = yes_ask + no_ask
                
                # If we can buy both for less than 1.0, arbitrage opportunity
                if total_ask < 1.0 - self.config["min_profit_threshold"]:
                    profit = 1.0 - total_ask
                    confidence = min(1.0, profit / 0.1)
                    
                    signals.append(Signal(
                        market_id=market_id,
                        outcome="YES",
                        action="BUY",
                        confidence=confidence,
                        price=yes_ask,
                        size=self.max_position_size,
                        reason=f"Three-way arbitrage: total ask {total_ask:.4f} < 1.0"
                    ))
        
        # Strategy 2: Spread arbitrage (large bid-ask spread)
        for market in markets:
            market_id = market.market_id
            for outcome, token_id in market.outcome_tokens.items():
                book = order_books.get(token_id)
                if not book or not book.bids or not book.asks:
                    continue
                
                spread = book.asks[0].price - book.bids[0].price
                spread_pct = spread / book.bids[0].price if book.bids[0].price > 0 else 0
                
                if spread_pct > self.config["max_spread"]:
                    # Large spread indicates potential arbitrage
                    mid_price = (book.bids[0].price + book.asks[0].price) / 2
                    confidence = min(0.7, spread_pct / 0.2)  # Cap confidence
                    
                    signals.append(Signal(
                        market_id=market_id,
                        outcome=outcome,
                        action="BUY",
                        confidence=confidence,
                        price=book.asks[0].price,
                        size=self.max_position_size * 0.5,  # Smaller size for spread trades
                        reason=f"Large spread: {spread_pct:.2%}"
                    ))
        
        return signals
    
    def should_enter(self, signal: Signal) -> bool:
        """Enter if confidence is high enough and profit threshold is met"""
        if not self.validate_signal(signal):
            return False
        
        # Additional validation for arbitrage
        if "arbitrage" in signal.reason.lower():
            return signal.confidence >= self.min_confidence
        
        return False

