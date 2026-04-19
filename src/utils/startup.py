"""Service initialization and pre-launch prompts for the trading bot."""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from ..api.client import PolymarketClient
from ..engine.cycle_snapshot import SnapshotStore
from ..engine.execution import ExecutionTracker
from ..engine.inventory import InventoryState
from ..engine.performance_store import PerformanceStore
from ..engine.risk_manager import RiskLimits, RiskManager
from ..engine.slot_state import SlotStateManager
from ..engine.state_store import BotState, StateStore
from ..models import BTCSigmoidModel
from ..models.logreg_model import LogRegModel
from ..models.logreg_v4_model import LogRegV4Model
from ..strategies.coin_toss import CoinTossStrategy
from ..strategies.logreg_edge import LogRegEdgeStrategy
from ..strategies.prob_edge import ProbEdgeStrategy
from ..utils.btc_feed import BtcPriceFeed
from ..utils.chainlink_feed import ChainlinkFeed
from ..utils.config import load_config
from ..utils.logger import setup_logger
from ..utils.market_utils import get_server_time

STRATEGIES = ["coin_toss", "logreg_edge", "logreg", "prob_edge"]

# The `logreg` strategy loads the v4 model (18 features + isotonic calibration)
# from models/logreg_v4 via LogRegV4Model. This is the single supported live
# logreg configuration — v2 runs through the separate `logreg_edge` strategy,
# and v5/v6 are trained but have no live feature builder.
LOGREG_MODEL_DIR = "models/logreg_v4"


@dataclass
class Services:
    """Container for all initialized bot services. Passed to CycleRunner."""
    client: PolymarketClient
    strategy: Any
    risk_manager: RiskManager
    state: BotState
    state_store: StateStore
    snapshot_store: SnapshotStore
    perf_store: PerformanceStore
    execution_tracker: ExecutionTracker
    inventories: Dict[str, InventoryState]
    btc_feed: Optional[BtcPriceFeed]
    chainlink_feed: Optional[ChainlinkFeed]
    slot_mgr: SlotStateManager
    paper_trading: bool
    interval: int
    session_loss_cap: float
    strategy_name: str
    market_keywords: List[str]
    min_volume: int
    logger: logging.Logger
    trade_log_path: Optional[str] = None
    price_tick_path: Optional[str] = None
    bot_start_ts: float = 0.0
    # Online retraining: daemon thread that periodically refits logreg_v4
    # on a rolling window of live data. None if disabled or not applicable
    # to the active strategy.
    retrainer: Optional[Any] = None
    # Multi-asset support: slug prefix for market discovery (e.g. "btc-updown-5m")
    slug_prefix: str = "btc-updown-5m"
    # Active asset ticker (e.g. "BTC", "ETH", "SOL", "DOGE", "XRP")
    asset: str = "BTC"


def init_services(args) -> Services:
    """Load config, initialize all services, return a Services bundle."""
    cfg = load_config("config/config.yaml")

    raw_yaml: dict = {}
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r") as f:
            raw_yaml = yaml.safe_load(f) or {}

    bot_cfg = raw_yaml.get("updown_bot", raw_yaml.get("btc_updown_bot", {}))
    state_file: str = bot_cfg.get("state_file", "bot_state.json")
    min_volume: int = int(bot_cfg.get("min_volume", 1_000_000))
    interval: int = int(raw_yaml.get("trading", {}).get("interval", 300))

    # Resolve active asset from CLI --asset flag or config default
    asset: str = (getattr(args, "asset", None) or bot_cfg.get("default_asset", "BTC")).upper()
    all_assets: dict = raw_yaml.get("assets", {})
    asset_cfg: dict = all_assets.get(asset, {})
    if not asset_cfg:
        # Fallback defaults for BTC when assets section is missing
        asset_cfg = {
            "slug_prefix": "btc-updown-5m",
            "market_keywords": ["Bitcoin", "Up or Down"],
            "chainlink_symbol": "btc/usd",
            "price_symbol": "BTC-USD",
            "price_exchange": "coinbase",
        }
    slug_prefix: str = asset_cfg.get("slug_prefix", "btc-updown-5m")
    market_keywords: List[str] = asset_cfg.get("market_keywords", bot_cfg.get("market_keywords", ["Bitcoin", "Up or Down"]))

    paper_trading: bool = cfg.paper_trading
    if getattr(args, "live", False):
        paper_trading = False
    elif getattr(args, "paper", False):
        paper_trading = True

    strategy_name: str
    if args.strategy is None and sys.stdin.isatty() and not getattr(args, "no_confirm", False):
        strategy_name, paper_trading = prompt_strategy_and_mode(
            default_strategy="logreg_edge",
            default_paper=paper_trading,
        )
    else:
        strategy_name = args.strategy or "logreg_edge"

    strategy_cfg: dict = cfg.strategies.get(strategy_name, {})

    # CLI --exit-rule overrides config
    exit_rule = getattr(args, "exit_rule", None)
    if exit_rule:
        strategy_cfg["exit_rule"] = exit_rule

    # CLI overrides for strategy config
    delta = getattr(args, "delta", None)
    if delta is not None:
        strategy_cfg["delta"] = delta
    position_size = getattr(args, "position_size", None)
    if position_size is not None:
        strategy_cfg["position_size_usdc"] = position_size
    kelly = getattr(args, "kelly", None)
    if kelly is not None:
        strategy_cfg["kelly_fraction"] = kelly

    # Inject paper_balance into strategy_cfg so it appears in the
    # pre-launch config panel and can be overridden interactively.
    cli_balance = getattr(args, "balance", None)
    strategy_cfg["paper_balance"] = cli_balance if cli_balance is not None else 10000.0

    if not getattr(args, "no_confirm", False) and sys.stdin.isatty():
        strategy_cfg = display_and_confirm_config(
            strategy_name=strategy_name,
            strategy_cfg=strategy_cfg,
            risk_cfg=raw_yaml.get("risk", {}),
            paper_trading=paper_trading,
            interval=interval,
        )

    # New-session prompt — runs after config confirmation, before state is loaded
    _clean = getattr(args, "clean", False)
    if not _clean and sys.stdin.isatty() and not getattr(args, "no_confirm", False):
        _clean = _prompt_new_session()
    if _clean:
        _clear_session_files(
            state_file=state_file,
            trade_log_path=raw_yaml.get("logging", {}).get("trade_log_file"),
            price_tick_path=raw_yaml.get("logging", {}).get("price_tick_file"),
        )

    # Logger
    log_file = raw_yaml.get("logging", {}).get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger(level=cfg.log_level, log_file=log_file)

    trade_log_path = raw_yaml.get("logging", {}).get("trade_log_file")
    if trade_log_path:
        os.makedirs(os.path.dirname(trade_log_path), exist_ok=True)
    price_tick_path = raw_yaml.get("logging", {}).get("price_tick_file")
    if price_tick_path:
        os.makedirs(os.path.dirname(price_tick_path), exist_ok=True)

    # Apply asset-level defaults for price symbol/exchange into strategy config
    # so strategies don't need to know about the asset layer — they just read btc_symbol / btc_exchange.
    if "btc_symbol" not in strategy_cfg:
        strategy_cfg["btc_symbol"] = asset_cfg.get("price_symbol", "BTC-USD")
    if "btc_exchange" not in strategy_cfg:
        strategy_cfg["btc_exchange"] = asset_cfg.get("price_exchange", "coinbase")

    logger.info(
        f"Starting bot | asset={asset} | strategy={strategy_name} | paper_trading={paper_trading} | interval={interval}s"
    )

    from ..utils.vpn import check_vpn
    if raw_yaml.get("vpn", {}).get("require_non_us", True):
        check_vpn(abort_if_us=True)

    perf_store = PerformanceStore(bot_cfg.get("perf_db_path", "perf.db"))

    paper_balance = float(strategy_cfg.pop("paper_balance", 10000.0))
    client = PolymarketClient(
        private_key=cfg.private_key,
        funder_address=cfg.funder_address,
        host=cfg.api_host,
        chain_id=cfg.chain_id,
        paper_trading=paper_trading,
        paper_balance=paper_balance,
    )

    session_loss_cap: float = raw_yaml.get("risk", {}).get("max_session_loss_usdc", float("inf"))
    risk_limits = RiskLimits(
        max_position_size=cfg.risk_limits.get("max_position_size", 200.0),
        max_position_pct=cfg.risk_limits.get("max_position_pct", 0.05),
        max_total_exposure=cfg.risk_limits.get("max_total_exposure", 0.15),
        max_daily_loss=cfg.risk_limits.get("max_daily_loss", 0.05),
        max_exposure_per_market=cfg.risk_limits.get("max_exposure_per_market", 0.10),
        circuit_breaker_enabled=cfg.risk_limits.get("circuit_breaker_enabled", True),
        circuit_breaker_threshold=cfg.risk_limits.get("circuit_breaker_threshold", 0.20),
        max_session_loss_usdc=session_loss_cap,
    )
    risk_manager = RiskManager(limits=risk_limits)

    execution_tracker = ExecutionTracker(
        orders_sync_interval_s=5.0,
        positions_sync_interval_s=10.0,
    )

    def _maybe_warmup(feed: BtcPriceFeed) -> None:
        """Seed the BTC feed buffer with 3 days of Binance 1s klines so
        multi-timeframe indicator features (rsi_*, ut_*_, td_*_) are live
        from the first cycle rather than zero-filled."""
        try:
            feed.warmup_from_binance(days=3)
        except Exception as e:
            # Warmup is opportunistic — a transient Binance outage shouldn't
            # block the bot from starting.
            logger.warning("Binance warmup skipped: %s", e)

    btc_feed: Optional[BtcPriceFeed] = None
    _retrainer: Optional[Any] = None  # set only by the `logreg` branch when enabled
    if strategy_name == "coin_toss":
        strategy = CoinTossStrategy(config=strategy_cfg)
    elif strategy_name == "logreg_edge":
        btc_feed = BtcPriceFeed(
            symbol=str(strategy_cfg.get("btc_symbol", "BTC-USD")),
            exchange=str(strategy_cfg.get("btc_exchange", "coinbase")),
            logger=logger,
        ).start()
        _maybe_warmup(btc_feed)
        model_dir = str(strategy_cfg.get("model_dir", "models/logreg"))
        logreg_model = LogRegModel.load(model_dir, logger=logger)
        if not logreg_model.ready:
            logger.warning("LogReg model not found at %s — strategy will skip all trades until a model is trained", model_dir)
        strategy = LogRegEdgeStrategy(
            config=strategy_cfg,
            btc_feed=btc_feed,
            model_service=logreg_model,
            logger=logger,
        )
    elif strategy_name == "logreg":
        btc_feed = BtcPriceFeed(
            symbol=str(strategy_cfg.get("btc_symbol", "BTC-USD")),
            exchange=str(strategy_cfg.get("btc_exchange", "coinbase")),
            logger=logger,
        ).start()
        _maybe_warmup(btc_feed)
        model_dir = str(strategy_cfg.get("model_dir", LOGREG_MODEL_DIR))
        v4_model = LogRegV4Model.load(model_dir, logger=logger)
        if not v4_model.ready:
            logger.warning("LogReg v4 model not found at %s — strategy will skip all trades until a model is trained", model_dir)
        else:
            logger.info("LogReg v4 loaded from %s", model_dir)
        strategy = LogRegEdgeStrategy(
            config=strategy_cfg,
            btc_feed=btc_feed,
            model_service=v4_model,
            logger=logger,
        )
        # Optional: online retrainer. Wired here (not in Services) because
        # it's strategy-specific — only the logreg branch has CSVs shaped
        # the way the retrainer expects.
        from ..engine.model_retrainer import Retrainer, RetrainConfig
        retrain_cfg = RetrainConfig.from_dict(strategy_cfg.get("retrain") or {})
        _retrainer = Retrainer(
            cfg=retrain_cfg,
            prod_model_dir=model_dir,
            logger=logger,
        ) if retrain_cfg.enabled else None
        if _retrainer is not None:
            _retrainer.start()
    elif strategy_name == "prob_edge":
        btc_feed = BtcPriceFeed(
            symbol=str(strategy_cfg.get("btc_symbol", "BTC-USD")),
            exchange=str(strategy_cfg.get("btc_exchange", "coinbase")),
            logger=logger,
        ).start()
        _maybe_warmup(btc_feed)
        strategy = ProbEdgeStrategy(
            config=strategy_cfg,
            btc_feed=btc_feed,
            model_service=BTCSigmoidModel(logger=logger),
            logger=logger,
        )
    else:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Valid: {', '.join(STRATEGIES)}"
        )

    chainlink_cfg = raw_yaml.get("chainlink_feed", {})
    chainlink_symbol = asset_cfg.get("chainlink_symbol", chainlink_cfg.get("symbol", "btc/usd"))
    chainlink_feed: Optional[ChainlinkFeed] = None
    if chainlink_cfg.get("enabled", True):
        chainlink_feed = ChainlinkFeed(
            symbol=chainlink_symbol,
            slot_interval_s=interval,
            logger=logger,
        ).start()
        logger.info(f"Chainlink reference price feed started (symbol={chainlink_symbol})")

    slot_mgr = SlotStateManager(clock_fn=get_server_time, logger=logger)

    state_store = StateStore(path=state_file)
    state = state_store.load()
    snapshot_store = SnapshotStore(path=state_file.replace(".json", "_snapshot.json"))

    inventories: Dict[str, InventoryState] = {}
    for token_id, inv_data in state.inventories.items():
        inv = InventoryState(token_id=token_id)
        inv.position = float(inv_data.get("position", 0.0))
        inv.avg_cost = float(inv_data.get("avg_cost", 0.0))
        inventories[token_id] = inv

    logger.info(
        f"State loaded | cycle_count={state.cycle_count} "
        f"| daily_pnl={state.daily_realized_pnl:.4f}"
    )
    risk_manager.daily_pnl = state.daily_realized_pnl

    import time
    return Services(
        client=client,
        strategy=strategy,
        risk_manager=risk_manager,
        state=state,
        state_store=state_store,
        snapshot_store=snapshot_store,
        perf_store=perf_store,
        execution_tracker=execution_tracker,
        inventories=inventories,
        btc_feed=btc_feed,
        chainlink_feed=chainlink_feed,
        slot_mgr=slot_mgr,
        paper_trading=paper_trading,
        interval=interval,
        session_loss_cap=session_loss_cap,
        strategy_name=strategy_name,
        market_keywords=market_keywords,
        min_volume=min_volume,
        logger=logger,
        trade_log_path=trade_log_path,
        price_tick_path=price_tick_path,
        bot_start_ts=time.time(),
        retrainer=_retrainer,
        slug_prefix=slug_prefix,
        asset=asset,
    )


def prompt_strategy_and_mode(
    default_strategy: str, default_paper: bool
) -> tuple:
    """Interactively ask user to pick strategy and trading mode. Returns (strategy, paper_trading)."""
    print("\nAvailable strategies:")
    for i, s in enumerate(STRATEGIES, 1):
        marker = " (default)" if s == default_strategy else ""
        print(f"  {i}. {s}{marker}")

    while True:
        try:
            raw = input(f"\nSelect strategy [1-{len(STRATEGIES)}] or name (Enter = {default_strategy}): ").strip()
        except EOFError:
            return default_strategy, default_paper
        if not raw:
            strategy = default_strategy
            break
        if raw.isdigit() and 1 <= int(raw) <= len(STRATEGIES):
            strategy = STRATEGIES[int(raw) - 1]
            break
        if raw in STRATEGIES:
            strategy = raw
            break
        print(f"  Invalid choice. Enter a number 1–{len(STRATEGIES)} or a strategy name.")

    default_mode = "paper" if default_paper else "live"
    while True:
        try:
            raw = input(f"Trading mode — paper or live? (Enter = {default_mode}): ").strip().lower()
        except EOFError:
            return strategy, default_paper
        if not raw:
            paper_trading = default_paper
            break
        if raw in ("paper", "p"):
            paper_trading = True
            break
        if raw in ("live", "l"):
            paper_trading = False
            break
        print("  Enter 'paper' or 'live'.")

    return strategy, paper_trading


def display_and_confirm_config(
    strategy_name: str,
    strategy_cfg: dict,
    risk_cfg: dict,
    paper_trading: bool,
    interval: int,
) -> dict:
    """Display active config and allow inline overrides before launch."""
    W = 48
    mode_str = "PAPER TRADING" if paper_trading else "LIVE TRADING"

    try:
        "╔═╗║╠╣╚╝".encode(sys.stdout.encoding or "utf-8")
        TL, H, TR, V, ML, MR, BL, BR = "╔", "═", "╗", "║", "╠", "╣", "╚", "╝"
    except (UnicodeEncodeError, LookupError):
        TL, H, TR, V, ML, MR, BL, BR = "+", "-", "+", "|", "+", "+", "+", "+"

    def _row(label: str, value: str) -> str:
        content = f"  {label:<15}{value}"
        return f"{V}{content:<{W}}{V}"

    def _header(text: str) -> str:
        return f"{V}  {text:<{W - 2}}{V}"

    border_top = f"{TL}{H * W}{TR}"
    border_mid = f"{ML}{H * W}{MR}"
    border_bot = f"{BL}{H * W}{BR}"

    lines = [border_top, _header("Polymarket Bot — Pre-launch Config"), border_mid,
             _row("Mode:", mode_str), _row("Strategy:", strategy_name),
             _row("Interval:", f"{interval}s ({interval // 60}-min cycles)"),
             border_mid, _header("Strategy Parameters")]
    for key in sorted(strategy_cfg.keys()):
        lines.append(_row(f"  {key}:", str(strategy_cfg[key])))
    lines += [border_mid, _header("Risk Limits")]
    for key in sorted(risk_cfg.keys()):
        lines.append(_row(f"  {key}:", str(risk_cfg[key])))
    lines.append(border_bot)
    print("\n".join(lines))

    while True:
        try:
            user_input = input("\nOverride? (e.g. 'stop_loss_pct=0.08', Enter to start)\n> ").strip()
        except EOFError:
            break
        if not user_input:
            break
        if "=" not in user_input:
            print("  Invalid format. Use key=value")
            continue
        key, _, raw_value = user_input.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in strategy_cfg:
            print(f"  Unknown key '{key}'. Valid keys: {', '.join(sorted(strategy_cfg.keys()))}")
            continue
        old_value = strategy_cfg[key]
        try:
            new_value = int(raw_value)
        except ValueError:
            try:
                new_value = float(raw_value)
            except ValueError:
                new_value = raw_value
        strategy_cfg[key] = new_value
        print(f"  Updated {key}: {old_value} → {new_value}")

    return strategy_cfg


def _prompt_new_session() -> bool:
    """Ask whether to resume or start clean. Returns True if clean session requested."""
    while True:
        try:
            raw = input(
                "\nResume previous session or start clean? (resume/clean, Enter=resume): "
            ).strip().lower()
        except EOFError:
            return False
        if not raw or raw in ("resume", "r"):
            return False
        if raw in ("clean", "c"):
            return True
        print("  Enter 'resume' or 'clean'.")


def _clear_session_files(
    state_file: str,
    trade_log_path: Optional[str],
    price_tick_path: Optional[str],
) -> None:
    """Delete accumulated session files so the next run starts clean.

    Clears: bot_state.json, bot_state_snapshot.json, trade log, price tick log.
    Never touches perf.db (historical session data kept for analysis).
    """
    targets = [
        state_file,
        state_file.replace(".json", "_snapshot.json"),
        trade_log_path,
        price_tick_path,
    ]
    cleared = []
    for path in targets:
        if not path:
            continue
        try:
            os.remove(path)
            cleared.append(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  Warning: could not clear {path}: {e}")
    if cleared:
        print(f"  Clean session — cleared: {', '.join(cleared)}")
    else:
        print("  Clean session — no previous files found.")
