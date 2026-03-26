"""CLI for evaluating a strategy and returning a PROMOTE / REJECT verdict.

Usage:
    python scripts/evaluate_strategy.py \\
        --strategy coin_toss \\
        --csv data/btc_updown_5m.csv \\
        [--hours 48] \\
        [--spread 0.02] \\
        [--min-trades 20] \\
        [--max-drawdown 0.25] \\
        [--debug] \\
        [--json-out results/eval.json]

Exit codes:
    0 — PROMOTE
    1 — REJECT
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.agent import EvaluationAgent, EvaluationConfig
from src.strategies.btc_updown import BTCUpDownStrategy
from src.strategies.btc_vol_reversion import BTCVolReversionStrategy
from src.strategies.coin_toss import CoinTossStrategy

STRATEGY_REGISTRY = {
    "coin_toss": lambda: CoinTossStrategy({"enabled": True, "position_size_usdc": 20.0}),
    "btc_updown": lambda: BTCUpDownStrategy({"enabled": True, "position_size_usdc": 20.0}),
    "btc_vol_reversion": lambda: BTCVolReversionStrategy({"enabled": True, "position_size_usdc": 20.0}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a strategy for promotion.")
    parser.add_argument("--strategy", required=True, choices=list(STRATEGY_REGISTRY), help="Strategy name")
    parser.add_argument("--csv", required=True, help="Path to resolved market CSV")
    parser.add_argument("--hours", type=float, default=None, help="Evaluation window in hours (default: all)")
    parser.add_argument("--spread", type=float, default=0.02, help="Synthetic full spread (default: 0.02)")
    parser.add_argument("--min-trades", type=int, default=20, help="Minimum trades required (default: 20)")
    parser.add_argument("--max-drawdown", type=float, default=0.25, help="Max drawdown threshold (default: 0.25)")
    parser.add_argument("--debug", action="store_true", help="Print trade-by-trade debug output")
    parser.add_argument("--json-out", default=None, help="Write result JSON to this path")
    args = parser.parse_args()

    strategy = STRATEGY_REGISTRY[args.strategy]()
    config = EvaluationConfig(
        spread=args.spread,
        min_trades=args.min_trades,
        max_drawdown_threshold=args.max_drawdown,
        evaluation_window_hours=args.hours,
        debug=args.debug,
    )

    agent = EvaluationAgent(config)
    result = agent.evaluate(strategy, args.csv)

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULT: {result.decision}")
    print(f"Reason: {result.reason}")
    print(f"Confidence: {result.confidence:.3f}")
    print("-" * 60)
    m = result.metrics
    print(f"Trades:       {m['num_trades']} / {m['total_resolved_rows']} resolved slots")
    print(f"Total PnL:    ${m['total_pnl']:+.4f}")
    print(f"PnL/trade:    ${m['pnl_per_trade']:+.4f}")
    print(f"Win rate:     {m['win_rate']:.1%}")
    print(f"Max drawdown: {m['max_drawdown']:.2%}")
    print(f"Sharpe:       {m['sharpe_ratio']:.2f}")
    print(f"Exposure:     {m['exposure_time']:.1%}")
    print(f"Concentration (top-3): {m['concentration_ratio']:.1%}")
    print(f"Instability:  {m['instability_ratio']:.2f}")
    print("=" * 60 + "\n")

    if args.json_out:
        out = dataclasses.asdict(result)
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Result written to {args.json_out}")

    sys.exit(0 if result.decision == "PROMOTE" else 1)


if __name__ == "__main__":
    main()
