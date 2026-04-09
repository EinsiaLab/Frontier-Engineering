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
    
    # Phase 1: Try all catalysts with high temperature and time to quickly identify promising ones
    all_cats = list(task.CATEGORIES["cat_index"])
    # Shuffle to avoid bias
    rng.shuffle(all_cats)
    initial_evaluations = min(8, budget)  # We can try up to all 8 catalysts
    for i in range(initial_evaluations):
        cat = all_cats[i]
        candidate = {
            "cat_index": int(cat),
            "conc_cat": float(rng.uniform(*task.BOUNDS["conc_cat"])),
            "t": float(task.BOUNDS["t"][1]),  # max time
            "temperature": float(task.BOUNDS["temperature"][1])  # max temperature
        }
        record = task.evaluate(experiment, candidate)
        history.append(record)
    
    # Find global best
    best_candidate = None
    best_score = -float('inf')
    if history:
        best_record = max(history, key=lambda row: row["y"])
        best_score = best_record["y"]
        best_candidate = {name: best_record[name] for name in task.INPUT_NAMES}
    
    # Phase 2: Focus on refining the best catalyst found, with occasional exploration
    while len(history) < budget:
        # Determine if we should explore a new catalyst or refine
        # As we progress, exploration probability decreases
        used = len(history)
        explore_prob = max(0.05, 0.3 - 0.25 * (used / budget))
        
        if best_candidate is None or rng.random() < explore_prob:
            # Explore: try a random catalyst with high temperature/time
            cat = rng.choice(task.CATEGORIES["cat_index"])
            candidate = {
                "cat_index": int(cat),
                "conc_cat": float(rng.uniform(*task.BOUNDS["conc_cat"])),
                "t": float(task.BOUNDS["t"][1]),  # max time
                "temperature": float(task.BOUNDS["temperature"][1])  # max temperature
            }
        else:
            # Refine: mutate the best candidate, bias towards high temperature/time
            candidate = task.mutate_candidate(best_candidate, rng)
            # Bias mutation towards higher temperature and time
            if rng.random() < 0.4:
                candidate["temperature"] = float(task.BOUNDS["temperature"][1])
            if rng.random() < 0.4:
                candidate["t"] = float(task.BOUNDS["t"][1])
        
        record = task.evaluate(experiment, candidate)
        history.append(record)
        
        # Update global best
        new_score = record["y"]
        if new_score > best_score:
            best_score = new_score
            best_candidate = {name: candidate[name] for name in task.INPUT_NAMES}
        
        # Strategic local search when few evaluations remain
        remaining = budget - len(history)
        if remaining > 0 and remaining <= 3 and best_candidate is not None:
            # Focus on the best candidate with systematic perturbations
            for i in range(remaining):
                # Always bias towards high temperature and time
                local_candidate = dict(best_candidate)
                local_candidate["temperature"] = float(task.BOUNDS["temperature"][1])
                local_candidate["t"] = float(task.BOUNDS["t"][1])
                local_candidate["conc_cat"] = float(rng.uniform(*task.BOUNDS["conc_cat"]))
                local_record = task.evaluate(experiment, local_candidate)
                history.append(local_record)
                
                local_score = local_record["y"]
                if local_score > best_score:
                    best_score = local_score
                    best_candidate = {name: local_candidate[name] for name in task.INPUT_NAMES}
            break
    
    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "simplified_catalyst_search",
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
