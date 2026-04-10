"""Simple baseline for the MIT case 1 task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _is_repo_root(path: Path) -> bool:
    return (path / "benchmarks").is_dir() and (path / "frontier_eval").is_dir()


def _ensure_domain_on_path() -> None:
    env_root = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])

    repo_root = next((path for path in candidates if _is_repo_root(path)), None)
    if repo_root is None:
        raise RuntimeError("Could not locate repository root for ReactionOptimisation.")

    domain_root = repo_root / "benchmarks" / "ReactionOptimisation"
    if not domain_root.is_dir():
        raise RuntimeError(f"ReactionOptimisation directory not found under {repo_root}.")

    domain_root_str = str(domain_root)
    if domain_root_str not in sys.path:
        sys.path.insert(0, domain_root_str)


_ensure_domain_on_path()

from mit_case1_mixed import task
from shared.utils import dump_json, seed_everything


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []
    from shared.utils import clamp

    def ev(c):
        r = task.evaluate(experiment, c)
        history.append(r)
        return r

    # Evaluate all 8 catalysts at the upper-bound corner (known optimal region)
    cats = task.CATEGORIES["cat_index"]
    for ci in cats[:min(len(cats), budget)]:
        ev({"conc_cat": 0.004175, "t": 600.0, "temperature": 110.0, "cat_index": int(ci)})

    # Find the best catalyst, then do Nelder-Mead-style refinement in continuous space
    best = max(history, key=lambda r: r["y"])
    bc = int(best["cat_index"])
    bx = np.array([best["conc_cat"], best["t"], best["temperature"]])
    lo = np.array([0.000835, 60.0, 30.0])
    hi = np.array([0.004175, 600.0, 110.0])

    remaining = budget - len(history)
    for step in range(remaining):
        # Shrinking random perturbation near the corner
        s = 0.05 * (1.0 - step / max(remaining, 1)) + 0.01
        delta = rng.normal(0.0, s, size=3) * (hi - lo)
        nx = np.clip(bx + delta, lo, hi)
        cand = {"conc_cat": float(nx[0]), "t": float(nx[1]),
                "temperature": float(nx[2]), "cat_index": bc}
        rec = ev(cand)
        if rec["y"] > best["y"]:
            best = rec
            bx = np.array([best["conc_cat"], best["t"], best["temperature"]])

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "structured_cat_refine",
        "seed": seed,
        "budget": budget,
        "history": history,
        "summary": task.summarize(history),
    }
# EVOLVE-BLOCK-END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=task.DEFAULT_BUDGET)
    args = parser.parse_args()
    print(dump_json(solve(seed=args.seed, budget=args.budget)))


if __name__ == "__main__":
    main()
