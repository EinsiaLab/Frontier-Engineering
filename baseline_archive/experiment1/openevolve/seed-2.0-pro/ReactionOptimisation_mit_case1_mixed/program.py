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
    initial_candidates = task.initial_candidates(rng)

    for step in range(budget):
        # First test all 8 catalysts once to identify promising categories early
        if step < len(initial_candidates):
            candidate = initial_candidates[step]
        else:
            # Anneal exploration rate from 0.4 to 0.1 after initial screening
            remaining_steps = budget - len(initial_candidates)
            progress = (step - len(initial_candidates)) / remaining_steps if remaining_steps > 0 else 0.0
            exploration_prob = 0.4 * (1 - progress) + 0.1
            if rng.random() < exploration_prob:
                candidate = task.sample_candidate(rng)
            else:
                candidate = task.mutate_candidate(best_candidate, rng)
                # 40% chance for second mutation to escape local minima
                if rng.random() < 0.4:
                    candidate = task.mutate_candidate(candidate, rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)
        incumbent = max(history, key=lambda row: row["y"])
        best_candidate = {name: incumbent[name] for name in task.INPUT_NAMES}

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "initial_catalyst_screen_annealed_search_v5",
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
