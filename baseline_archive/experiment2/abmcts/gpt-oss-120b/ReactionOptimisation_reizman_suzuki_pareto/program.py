"""Simple baseline for the Reizman Suzuki emulator task."""

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

from reizman_suzuki_pareto import task
from shared.utils import dump_json, seed_everything


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    # --- Stage 1: quick screening of catalysts ---
    # Allocate a modest portion of the budget to sample each catalyst at least once.
    # We use 20% of the total budget (but at least 10 evaluations) for this stage.
    screening_budget = max(10, budget // 5)
    best_score_per_catalyst: dict = {}

    for _ in range(screening_budget):
        cand = task.sample_candidate(rng)
        rec = task.evaluate(experiment, cand)
        history.append(rec)

        # scalarize the two objectives (yield & ton) the same way as the summary does
        yld = rec.get("yld", 0.0)
        ton = rec.get("ton", 0.0)
        ny = max(0.0, min(1.0, yld / 100.0))
        nt = max(0.0, min(1.0, 1.0 - ton / 200.0))
        score = ny + nt

        cat = cand.get("catalyst")
        if cat is not None:
            if cat not in best_score_per_catalyst or score > best_score_per_catalyst[cat]:
                best_score_per_catalyst[cat] = score

    # Determine the most promising catalysts (up to three)
    top_catalysts = [
        cat for cat, _ in sorted(best_score_per_catalyst.items(), key=lambda kv: kv[1], reverse=True)[:3]
    ]

    # If for any reason we didn't discover any catalyst (unlikely), fall back to random
    if not top_catalysts:
        top_catalysts = [task.sample_candidate(rng).get("catalyst")]

    # --- Stage 2: focused search using the top catalysts ---
    remaining_budget = budget - screening_budget
    for _ in range(remaining_budget):
        cand = task.sample_candidate(rng)
        # force the catalyst to one of the top performers
        cand["catalyst"] = rng.choice(top_catalysts)
        rec = task.evaluate(experiment, cand)
        history.append(rec)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "screen_then_focus_random",
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
