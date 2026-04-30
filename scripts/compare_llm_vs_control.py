"""
Compare the LLM-assisted tuning loop vs a pure-Optuna control.

Both tracks use 300 total trials. LLM track = 3 rounds × 100 trials with
Claude proposing feature subsets + search space between rounds. Control =
one 300-trial Optuna study with default search space.

Produces a side-by-side Pareto plot + a hypervolume summary table.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM_DIRS = [os.path.join(REPO, f"experiments/v5/round_{r}") for r in (0, 1, 2)]
CTRL_DIR = os.path.join(REPO, "experiments/v5/control_300")


def load_top(dir_):
    return json.loads(open(os.path.join(dir_, "top_trials.json")).read())


def pareto_filter(pts):
    """Return the non-dominated subset (min brier, max equity).
    pts: list of dicts with 'brier' and 'final_equity'."""
    out = []
    for i, p in enumerate(pts):
        b, e = float(p["brier"]), float(p["final_equity"])
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            bq, eq = float(q["brier"]), float(q["final_equity"])
            if bq <= b and eq >= e and (bq < b or eq > e):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return out


def hv(pareto, ref_b=1.0, ref_e=100.0):
    """2D hypervolume for (min brier, max equity) vs reference point.
    `pareto` must already be Pareto-filtered (use pareto_filter first).
    """
    pts = sorted(
        [(float(p["brier"]), float(p["final_equity"])) for p in pareto
         if float(p["brier"]) < ref_b and float(p["final_equity"]) > ref_e]
    )
    total = 0.0
    for i, (b, e) in enumerate(pts):
        next_b = pts[i + 1][0] if i + 1 < len(pts) else ref_b
        total += (next_b - b) * (e - ref_e)
    return total


def merged_pareto(round_pairs):
    """Merge multiple round Pareto sets → one global non-dominated set."""
    pts = []
    for d in round_pairs:
        for p in d.get("pareto", []):
            pts.append({
                "brier": float(p["brier"]),
                "final_equity": float(p["final_equity"]),
                "params": p.get("params", {}),
            })
    return sorted(pareto_filter(pts), key=lambda r: r["brier"])


llm_rounds = [load_top(d) for d in LLM_DIRS]
ctrl = load_top(CTRL_DIR)

llm_global = merged_pareto(llm_rounds)
ctrl_global = sorted(pareto_filter(ctrl.get("pareto", [])),
                     key=lambda r: float(r["brier"]))

print("=" * 70)
print(f"{'source':<22}{'n_trials':>10}{'pareto':>10}{'best_brier':>14}{'best_equity':>14}")
print("-" * 70)
for i, d in enumerate(llm_rounds):
    nt = len(d.get("all_trials", []))
    p = d.get("pareto", [])
    bb = min((t["brier"] for t in p), default=float("nan"))
    be = max((t["final_equity"] for t in p), default=float("nan"))
    print(f"{'LLM round '+str(i):<22}{nt:>10}{len(p):>10}{bb:>14.4f}{be:>14.2f}")
print(f"{'LLM merged':<22}{'300':>10}{len(llm_global):>10}"
      f"{min(p['brier'] for p in llm_global):>14.4f}"
      f"{max(p['final_equity'] for p in llm_global):>14.2f}")

nt = len(ctrl.get("all_trials", []))
p = ctrl.get("pareto", [])
bb = min((t["brier"] for t in p), default=float("nan"))
be = max((t["final_equity"] for t in p), default=float("nan"))
print(f"{'Control (pure Optuna)':<22}{nt:>10}{len(p):>10}{bb:>14.4f}{be:>14.2f}")
print("=" * 70)
print("\nHypervolume (ref: brier=1.0, equity=$100):")
print(f"  LLM merged : {hv(llm_global):.2f}")
print(f"  Control    : {hv(ctrl_global):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
colors = {"round_0": "#1f77b4", "round_1": "#2ca02c", "round_2": "#d62728"}
for i, d in enumerate(llm_rounds):
    trials = [t for t in d.get("all_trials", [])
              if t.get("brier") is not None]
    xs = [t["brier"] for t in trials]
    ys = [t["final_equity"] for t in trials]
    ax.scatter(xs, ys, s=10, alpha=0.35,
               color=colors[f"round_{i}"], label=f"round {i} trials")
pts = sorted(llm_global, key=lambda p: p["brier"])
ax.plot([p["brier"] for p in pts], [p["final_equity"] for p in pts],
        "ko-", lw=1.5, ms=7, label="LLM merged Pareto")
ax.axhline(100, color="k", lw=0.5, ls="--")
ax.set_xlabel("valid Brier (minimize)")
ax.set_ylabel("final equity $ (maximize)")
ax.set_title("LLM-assisted loop (3 rounds × 100 trials)")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
trials = [t for t in ctrl.get("all_trials", []) if t.get("brier") is not None]
ax.scatter([t["brier"] for t in trials], [t["final_equity"] for t in trials],
           s=10, alpha=0.35, color="#888", label="control trials")
pts = sorted(ctrl_global, key=lambda p: p["brier"])
ax.plot([p["brier"] for p in pts], [p["final_equity"] for p in pts],
        "ro-", lw=1.5, ms=7, label="control Pareto")
# Overlay LLM Pareto for direct comparison
pts_llm = sorted(llm_global, key=lambda p: p["brier"])
ax.plot([p["brier"] for p in pts_llm], [p["final_equity"] for p in pts_llm],
        "k^-", lw=1.5, ms=7, mfc="none", label="LLM Pareto (overlay)")
ax.axhline(100, color="k", lw=0.5, ls="--")
ax.set_xlabel("valid Brier (minimize)")
ax.set_ylabel("final equity $ (maximize)")
ax.set_title("Control: pure Optuna (300 trials, default space)")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
out = os.path.join(REPO, "experiments/v5/llm_vs_control.png")
plt.savefig(out, dpi=120)
print(f"\nSaved plot → {out}")
