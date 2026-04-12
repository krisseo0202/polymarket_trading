"""
LLM experiment suggester for logreg_v5.

Reads a completed Optuna round, asks Claude (sonnet-4-6) to propose the next
round's feature subset, tightened hyperparameter ranges, and a best-guess
seed point. Writes a structured JSON to `<round_dir>/suggestion.json`.

Usage:
  ANTHROPIC_API_KEY=... python scripts/llm_suggest_v5.py --round 0

Env:
  ANTHROPIC_API_KEY  required
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except Exception:
    pass

from _v5_feature_docs import FEATURE_DOCS, AVAILABLE_FEATURES  # noqa: E402


MODEL_ID = "claude-sonnet-4-6"
MAX_TOKENS = 2048
TEMPERATURE = 0.4

SYSTEM_PROMPT = """You are an ML experiment designer helping tune a logistic-regression model
that trades 5-minute BTC Up/Down binary markets on Polymarket.

Setup:
- Model: sklearn LogisticRegression with isotonic post-hoc calibration.
- Labels: y=1 if the up side resolves, y=0 otherwise, derived from the
  market's late-slot mid price.
- Split: walk-forward, last 20% of slots held out.
- Backtest: $100 start. For each held-out slot, take the first decision row
  whose edge (p - fillable_price) exceeds edge_threshold, size with
  kelly_mult * Kelly fraction capped at max_frac, resolve immediately.
- Objective: multi-objective — MINIMIZE validation Brier AND MAXIMIZE
  final backtest equity. Calibration (Brier) matters because Kelly sizing
  multiplies probability error into dollar loss.

Your job each round:
- Look at what worked in the last Optuna round.
- Propose a feature subset (drawn ONLY from the provided available list),
  tightened hyperparameter ranges around the Pareto front, and one seed
  point that represents your best guess.
- Be willing to DROP features that dominate coefficients without improving
  the Pareto front — collinear/redundant features hurt calibration.
- Keep ranges non-degenerate (low < high) and realistic (e.g. don't set
  edge_threshold above 0.2 — nothing would ever trade).

Output: STRICT JSON matching this schema, and NOTHING ELSE. No markdown
fences, no prose before or after.

{
  "rationale": "2-4 sentence explanation of why these changes",
  "feature_subset": ["col1", "col2", ...],
  "hparam_ranges": {
    "C":              {"type": "log_uniform", "low": <float>, "high": <float>},
    "row_interval":   {"type": "categorical", "choices": [<int>, ...]},
    "edge_threshold": {"type": "uniform",    "low": <float>, "high": <float>},
    "kelly_mult":     {"type": "uniform",    "low": <float>, "high": <float>},
    "max_frac":       {"type": "uniform",    "low": <float>, "high": <float>}
  },
  "seed_trial": {
    "C": <float>,
    "row_interval": <int>,
    "edge_threshold": <float>,
    "kelly_mult": <float>,
    "max_frac": <float>
  }
}
"""


def build_feature_docs_block() -> str:
    lines = ["# Available features\n"]
    lines += [f"- `{k}`: {v}" for k, v in FEATURE_DOCS.items()]
    return "\n".join(lines)


def build_user_prompt(round_dir: str, top_trials: dict) -> str:
    pareto = top_trials.get("pareto", [])
    all_trials = top_trials.get("all_trials", [])
    # worst 5 completed trials (by Brier desc) for "what failed"
    completed = [t for t in all_trials if t.get("brier") is not None]
    worst = sorted(completed, key=lambda t: -t["brier"])[:5]

    parts = []
    parts.append(f"Current round: {top_trials.get('round', '?')}")
    parts.append(f"Trials run: {len(all_trials)}")
    parts.append(f"\n## Current feature subset ({len(top_trials.get('feature_subset', []))})")
    parts.append(json.dumps(top_trials.get("feature_subset", []), indent=2))
    parts.append("\n## Current search space")
    parts.append(json.dumps(top_trials.get("search_space", {}), indent=2))
    parts.append(f"\n## Pareto front ({len(pareto)} trials)")
    parts.append(json.dumps(pareto[:15], indent=2, default=str))
    parts.append("\n## Worst 5 trials (high Brier)")
    parts.append(json.dumps(worst, indent=2, default=str))
    parts.append(
        "\nPropose the next round now. Return only the JSON object, no fences."
    )
    return "\n".join(parts)


def call_claude(system: str, feature_docs: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=[
            {"type": "text", "text": system,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": feature_docs,
             "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text if resp.content else ""


REQUIRED_HPARAMS = {"C", "row_interval", "edge_threshold", "kelly_mult", "max_frac"}


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # drop the first fence line
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def parse_and_validate(text: str) -> dict:
    data = json.loads(_strip_fences(text))
    for key in ("rationale", "feature_subset", "hparam_ranges", "seed_trial"):
        if key not in data:
            raise ValueError(f"missing key: {key}")

    fs = data["feature_subset"]
    if not isinstance(fs, list) or not fs:
        raise ValueError("feature_subset must be a non-empty list")
    bad = [f for f in fs if f not in AVAILABLE_FEATURES]
    if bad:
        raise ValueError(f"unknown features: {bad}")

    ranges = data["hparam_ranges"]
    missing = REQUIRED_HPARAMS - set(ranges.keys())
    if missing:
        raise ValueError(f"hparam_ranges missing keys: {missing}")
    for name, spec in ranges.items():
        t = spec.get("type")
        if t in ("log_uniform", "uniform"):
            lo, hi = float(spec["low"]), float(spec["high"])
            if not (lo < hi):
                raise ValueError(f"{name}: low must be < high")
        elif t == "categorical":
            ch = spec.get("choices")
            if not isinstance(ch, list) or not ch:
                raise ValueError(f"{name}: categorical needs non-empty choices")
        else:
            raise ValueError(f"{name}: unknown type {t}")

    seed = data["seed_trial"]
    missing_seed = REQUIRED_HPARAMS - set(seed.keys())
    if missing_seed:
        raise ValueError(f"seed_trial missing keys: {missing_seed}")

    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--round-dir", default=None,
                    help="Override round dir (default experiments/v5/round_<N>).")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment", file=sys.stderr)
        sys.exit(2)

    round_dir = args.round_dir or os.path.join(
        REPO_ROOT, f"experiments/v5/round_{args.round}"
    )
    top_path = os.path.join(round_dir, "top_trials.json")
    if not os.path.exists(top_path):
        print(f"ERROR: {top_path} not found — run tune_v5_optuna.py first",
              file=sys.stderr)
        sys.exit(1)

    top_trials = json.loads(open(top_path).read())
    feature_docs = build_feature_docs_block()
    user_prompt = build_user_prompt(round_dir, top_trials)

    print(f"Calling Claude for round {args.round}...")
    raw = call_claude(SYSTEM_PROMPT, feature_docs, user_prompt)

    # Parse with a single retry on failure
    try:
        data = parse_and_validate(raw)
    except Exception as e:
        print(f"Parse error: {e}; retrying with error feedback...")
        retry_user = user_prompt + (
            f"\n\nYOUR LAST RESPONSE FAILED VALIDATION: {e}\n"
            f"Return ONLY the JSON object matching the schema, no fences."
        )
        raw = call_claude(SYSTEM_PROMPT, feature_docs, retry_user)
        data = parse_and_validate(raw)

    out_path = os.path.join(round_dir, "suggestion.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {out_path}")
    print(f"\nRationale: {data['rationale']}")
    print(f"Feature subset ({len(data['feature_subset'])}): {data['feature_subset']}")


if __name__ == "__main__":
    main()
