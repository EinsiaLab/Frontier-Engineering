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
    best_candidate = None
    best_y = float('-inf')
    stagnation = 0

    for step in range(budget):
        # Adaptive exploration: high early, low later
        explore_rate = 0.45 * (1 - step / budget) + 0.1
        
        if best_candidate is None:
            candidate = task.sample_candidate(rng)
        elif stagnation >= 12:
            candidate = task.sample_candidate(rng)
            stagnation = 0
        elif rng.random() < explore_rate:
            candidate = task.sample_candidate(rng)
        else:
            candidate = task.mutate_candidate(best_candidate, rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)
        
        if record["y"] > best_y:
            best_y = record["y"]
            best_candidate = {name: record[name] for name in task.INPUT_NAMES}
            stagnation = 0
        else:
            stagnation += 1

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "adaptive_exploration_search",
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
