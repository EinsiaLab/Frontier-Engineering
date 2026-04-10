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

    # Stage 1: Screen all catalyst categories with fixed continuous parameters
    for cat in task.CATEGORIES["catalyst"]:
        if len(history) >= budget:
            break
        candidate = {
            "catalyst": cat,
            "t_res": 360.0,
            "temperature": 100.0,
            "catalyst_loading": 2.0,
        }
        history.append(task.evaluate(experiment, candidate))

    # Stage 2: Refine promising candidates using different scalarization weights
    weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    while len(history) < budget:
        w = float(rng.choice(weights))
        best_record = max(history, key=lambda r: task.scalarize(r, w))
        
        base_candidate = {k: best_record[k] for k in task.INPUT_NAMES}
        new_candidate = task.mutate_candidate(base_candidate, rng)
        
        history.append(task.evaluate(experiment, new_candidate))

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "scalarized_mutation_search",
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
