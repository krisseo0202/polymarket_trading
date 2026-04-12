"""
Closed-loop tuner: Optuna → LLM suggester → Optuna → ...

Each round runs an Optuna multi-objective study (minimize Brier, maximize
final equity), asks Claude to propose the next round's feature subset and
search space based on the study, then kicks off the next round seeded with
the LLM's best-guess point.

Stops early if the Pareto hypervolume does not improve from round r-1 → r.

Usage:
  ANTHROPIC_API_KEY=... python scripts/tune_and_suggest_loop.py \\
      --rounds 3 --trials-per-round 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except Exception:
    pass


def run(cmd: list) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def load_top_trials(round_dir: str) -> dict:
    p = os.path.join(round_dir, "top_trials.json")
    return json.loads(open(p).read()) if os.path.exists(p) else {}


def hypervolume(pareto: list, ref_brier: float = 1.0,
                ref_equity: float = 100.0) -> float:
    """Simple 2D hypervolume vs (ref_brier, ref_equity).
    Larger = better. Assumes we want low brier + high equity, and that the
    Pareto set is small (≤ few dozen points)."""
    if not pareto:
        return 0.0
    pts = sorted(
        [(float(t["brier"]), float(t["final_equity"])) for t in pareto],
        key=lambda p: p[0],
    )
    hv = 0.0
    prev_brier = ref_brier
    for brier, equity in pts:
        if brier >= ref_brier or equity <= ref_equity:
            continue
        width = prev_brier - brier
        height = equity - ref_equity
        hv += width * height
        prev_brier = brier
    return hv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--trials-per-round", type=int, default=100)
    ap.add_argument("--stop-on-no-improve", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    py = sys.executable
    tune = os.path.join(SCRIPT_DIR, "tune_v5_optuna.py")
    suggest = os.path.join(SCRIPT_DIR, "llm_suggest_v5.py")

    history = []
    prev_hv = -1.0

    for r in range(args.rounds):
        round_dir = os.path.join(REPO_ROOT, f"experiments/v5/round_{r}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"\n{'=' * 60}\nROUND {r}\n{'=' * 60}")

        # ── Optuna ─────────────────────────────────────────────────────
        tune_cmd = [py, tune, "--trials", str(args.trials_per_round),
                    "--round", str(r)]
        if r > 0:
            prev_sugg = os.path.join(
                REPO_ROOT, f"experiments/v5/round_{r - 1}/suggestion.json"
            )
            tune_cmd += ["--feature-subset-file", prev_sugg,
                         "--search-space-file", prev_sugg,
                         "--seed-trial-file", prev_sugg]
        if run(tune_cmd) != 0:
            print(f"ERROR: round {r} Optuna failed", file=sys.stderr); sys.exit(1)

        top = load_top_trials(round_dir)
        hv = hypervolume(top.get("pareto", []))
        best_brier = min((t["brier"] for t in top.get("pareto", [])), default=None)
        best_eq = max((t["final_equity"] for t in top.get("pareto", [])), default=None)
        print(f"\n[round {r}] hypervolume={hv:.3f}  "
              f"best_brier={best_brier}  best_equity={best_eq}")
        history.append({
            "round": r, "hypervolume": hv,
            "best_brier": best_brier, "best_equity": best_eq,
            "n_pareto": len(top.get("pareto", [])),
        })

        if r > 0 and args.stop_on_no_improve and hv <= prev_hv:
            print(f"\nEarly stop: hypervolume {hv:.3f} ≤ prev {prev_hv:.3f}")
            break
        prev_hv = hv

        # ── LLM suggester (skip on final round) ─────────────────────────
        if r == args.rounds - 1:
            break
        sugg_cmd = [py, suggest, "--round", str(r)]
        if run(sugg_cmd) != 0:
            print(f"WARN: round {r} suggester failed, stopping loop")
            break

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\nLOOP SUMMARY\n{'=' * 60}")
    print(f"{'round':<8}{'hv':>10}{'brier':>12}{'equity':>12}{'n_pareto':>12}")
    for h in history:
        print(f"{h['round']:<8}{h['hypervolume']:>10.3f}"
              f"{(h['best_brier'] or 0):>12.4f}"
              f"{(h['best_equity'] or 0):>12.2f}"
              f"{h['n_pareto']:>12}")

    out = os.path.join(REPO_ROOT, "experiments/v5/loop_history.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved history → {out}")


if __name__ == "__main__":
    main()
