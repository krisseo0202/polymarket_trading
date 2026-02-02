"""Configuration management"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Application configuration"""
    # API settings
    private_key: Optional[str] = None
    funder_address: Optional[str] = None
    api_host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    
    # Trading settings
    paper_trading: bool = True
    trading_interval: int = 60  # seconds
    
    # Risk management
    risk_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_position_size": 1000.0,
        "max_position_pct": 0.1,
        "max_total_exposure": 0.5,
        "max_daily_loss": 0.1,
        "max_loss_per_trade": 0.05,
        "max_exposure_per_market": 0.2,
        "stop_loss_pct": 0.1,
        "circuit_breaker_enabled": True,
        "circuit_breaker_threshold": 0.15
    })
    
    # Strategies
    strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Backtesting
    backtest_initial_balance: float = 10000.0


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file or environment variables
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Config object
    """
    config = Config()
    
    # Try to load from file
    if config_path:
        config_path_obj = Path(config_path)
        if config_path_obj.exists():
            with open(config_path_obj, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config = _dict_to_config(yaml_config)
    else:
        # Try default config path
        default_path = Path("config/config.yaml")
        if default_path.exists():
            with open(default_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config = _dict_to_config(yaml_config)
    
    # Override with environment variables
    config.private_key = os.getenv("PRIVATE_KEY", config.private_key)
    config.funder_address = os.getenv("FUNDER_ADDRESS", config.funder_address)
    config.paper_trading = os.getenv("PAPER_TRADING", str(config.paper_trading)).lower() == "true"
    config.log_level = os.getenv("LOG_LEVEL", config.log_level)
    
    return config


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Convert dictionary to Config object"""
    config = Config()
    
    if "api" in data:
        api = data["api"]
        config.private_key = api.get("private_key")
        config.funder_address = api.get("funder_address")
        config.api_host = api.get("host", config.api_host)
        config.chain_id = api.get("chain_id", config.chain_id)
    
    if "trading" in data:
        trading = data["trading"]
        config.paper_trading = trading.get("paper_trading", config.paper_trading)
        config.trading_interval = trading.get("interval", config.trading_interval)
    
    if "risk" in data:
        config.risk_limits.update(data["risk"])
    
    if "strategies" in data:
        config.strategies = data["strategies"]
    
    if "logging" in data:
        logging = data["logging"]
        config.log_level = logging.get("level", config.log_level)
        config.log_file = logging.get("file")
    
    if "backtest" in data:
        backtest = data["backtest"]
        config.backtest_initial_balance = backtest.get("initial_balance", config.backtest_initial_balance)
    
    return config

