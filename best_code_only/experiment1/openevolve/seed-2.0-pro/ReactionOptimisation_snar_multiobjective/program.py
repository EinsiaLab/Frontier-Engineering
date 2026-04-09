"""Simple baseline for the SnAr multi-objective task."""

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

from shared.utils import dump_json, seed_everything
from snar_multiobjective import task


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []
    best_by_weight = {}
    pareto_set = []
    # Balanced weights covering key tradeoffs, 3 weights = 8 runs each for budget 24
    weights = [0.2, 0.5, 0.8]

    for step in range(budget):
        weight = weights[step % len(weights)]
        current_best = best_by_weight.get(weight, None)
        # Anneal exploration: 55% random early to find good regions, 5% random late to exploit
        explore_prob = 0.55 - 0.5 * (step / budget)
        if not history or current_best is None or rng.random() < explore_prob:
            candidate = task.sample_candidate(rng)
        else:
            # Choose mutation source for better diversity: 35% per-weight best, 25% overall best, 40% random Pareto point
            choice = rng.random()
            if choice < 0.25:
                overall_best = max(history, key=lambda r: sum(task.scalarize(r, w) for w in weights))
                current_best = {n: overall_best[n] for n in task.INPUT_NAMES}
            elif choice < 0.65:
                # Mutate random Pareto point to explore better tradeoffs (higher weight for Pareto improves frontier)
                pareto_point = rng.choice(pareto_set)
                current_best = {n: pareto_point[n] for n in task.INPUT_NAMES}
            candidate = task.mutate_candidate(current_best, rng)
            # Anneal resample chance to reduce randomness late in run (from 7% to 0%)
            resample_prob = 0.07 * (1 - step / budget)
            if rng.random() < resample_prob:
                candidate = task.sample_candidate(rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)

        # Update Pareto set to use as diverse mutation candidates
        obj_array = np.array([[r["sty"], r["e_factor"]] for r in history], dtype=float)
        pareto_mask = task.pareto_mask(obj_array, maximize=[True, False])
        pareto_set = [r for r, is_pareto in zip(history, pareto_mask) if is_pareto]

        incumbent = max(history, key=lambda row: task.scalarize(row, weight))
        best_by_weight[weight] = {name: incumbent[name] for name in task.INPUT_NAMES}

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "adaptive_random_scalarization",
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
