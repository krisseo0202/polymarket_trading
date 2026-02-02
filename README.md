# Polymarket Trading System

A modular trading system for Polymarket with support for multiple strategies, backtesting, and risk management.

## Features

- **Multiple Trading Strategies**: Arbitrage, Momentum, Mean Reversion
- **Strategy Pattern**: Easy to add new strategies by extending the base `Strategy` class
- **Risk Management**: Position sizing, stop-loss, daily limits, circuit breakers
- **Backtesting Framework**: Test strategies on historical data
- **Paper Trading**: Simulate trades without real money
- **Live Trading**: Execute real trades on Polymarket
- **Configuration Management**: YAML-based configuration
- **Comprehensive Logging**: Track all trading activity

## Installation

### Using Conda (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd polymarket_trading
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate polymarket_trading
```

### Using pip

1. Clone the repository:
```bash
git clone <repository-url>
cd polymarket_trading
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
export PRIVATE_KEY="your_private_key"
export FUNDER_ADDRESS="your_funder_address"
export PAPER_TRADING="true"  # Set to "false" for live trading
```

See `SETUP.md` for detailed setup instructions.

## Configuration

Edit `config/config.yaml` to configure:
- API credentials
- Risk limits
- Strategy parameters
- Trading settings

## Usage

### Live Trading

Run the trading system with configured strategies:

```bash
python examples/live_trading.py
```

### Backtesting

Test a strategy on historical data:

```bash
python examples/backtest_example.py
```

### Custom Strategy

Create your own strategy by extending the `Strategy` base class:

```python
from src.strategies.base import Strategy, Signal

class MyStrategy(Strategy):
    def analyze(self, market_data):
        # Your analysis logic
        return signals
    
    def should_enter(self, signal):
        # Your entry logic
        return True
```

See `examples/simple_strategy.py` for a complete example.

## Project Structure

```
polymarket_trading/
├── src/
│   ├── api/              # Polymarket API client
│   ├── strategies/       # Trading strategies
│   ├── engine/           # Trading engine and risk management
│   ├── backtest/         # Backtesting framework
│   └── utils/            # Configuration and logging
├── config/               # Configuration files
├── examples/             # Example scripts
└── tests/                # Unit tests
```

## Strategies

### Arbitrage Strategy
- Three-way arbitrage (YES + NO prices)
- Spread arbitrage
- Cross-market opportunities

### Momentum Strategy
- Price momentum detection
- Moving average crossovers
- Volume breakouts

### Mean Reversion Strategy
- Oversold/overbought detection
- RSI-based signals
- Price deviation from mean

## Risk Management

The system includes comprehensive risk management:
- Maximum position size limits
- Daily loss limits
- Stop-loss mechanisms
- Circuit breakers
- Exposure limits per market

## Backtesting

The backtesting framework allows you to:
- Test strategies on historical data
- Calculate performance metrics (Sharpe ratio, max drawdown, win rate)
- Compare different strategies
- Optimize strategy parameters

## Paper Trading

By default, the system runs in paper trading mode, which:
- Simulates trades without executing real orders
- Tracks positions and PnL
- Allows safe testing of strategies

## Contributing

1. Create a new strategy by extending `Strategy` base class
2. Implement `analyze()` and `should_enter()` methods
3. Add strategy configuration to `config/config.yaml`
4. Test with backtesting framework
5. Deploy with paper trading first

## License

MIT License

## Disclaimer

This software is for educational purposes only. Trading involves risk of loss. Use at your own risk.

