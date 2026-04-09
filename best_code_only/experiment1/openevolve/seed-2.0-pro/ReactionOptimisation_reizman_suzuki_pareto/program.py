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

    # Track best candidates for multiple scalarization weights to cover full Pareto front
    weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # Cover extremes (pure yield / E-factor) + midpoints
    best_scores = [-float('inf') for _ in weights]
    best_candidates = [None for _ in weights]
    # Step 1: Screen all catalysts first at standardized fixed conditions for fair comparison
    initial_candidates = []
    for catalyst in task.CATEGORIES["catalyst"]:
        initial_candidates.append({
            "catalyst": catalyst,
            "t_res": 360.0,
            "temperature": 100.0,
            "catalyst_loading": 2.0,
        })
    for candidate in initial_candidates[:budget]:  # Handle edge case if budget < 8
        record = task.evaluate(experiment, candidate)
        history.append(record)
        # Update best candidates for all scalarization weights
        for i, weight in enumerate(weights):
            current_score = task.scalarize(record, weight)
            if current_score > best_scores[i]:
                best_scores[i] = current_score
                best_candidates[i] = candidate

    remaining_budget = budget - len(history)
    # Step 2: Optimize promising candidates for remaining budget with decaying exploration rate
    for step_idx in range(remaining_budget):
        # Decay exploration rate and mutation parameters for better refinement
        progress = step_idx / remaining_budget if remaining_budget > 0 else 1.0
        # Gradually reduce exploration from 10% to 0%: focus more on fine-tuning as we go
        exploration_rate = max(0.0, 0.1 * (1.0 - progress))
        # Gradually reduce mutation step size from 10% to 2% of parameter span
        step_scale = 0.1 - (0.08 * progress)
        # Gradually reduce random reset probability from 20% to 5%
        reset_prob = 0.2 - (0.15 * progress)
        
        # Epsilon-greedy selection
        if rng.random() < exploration_rate:
            candidate = task.sample_candidate(rng)
        else:
            # Pick random weight's best candidate to mutate for better Pareto coverage
            target_weight_idx = rng.integers(0, len(weights))
            base_candidate = best_candidates[target_weight_idx]
            # Custom mutation: keep catalyst fixed (we already found best catalysts for each weight from screening)
            # Only mutate continuous variables to refine performance without losing good catalyst selection
            candidate = dict(base_candidate)
            for name, (low, high) in task.BOUNDS.items():
                span = high - low
                step = float(rng.normal(0.0, step_scale * span))
                if rng.random() < reset_prob:
                    candidate[name] = float(rng.uniform(low, high))
                else:
                    candidate[name] = np.clip(float(base_candidate[name]) + step, low, high)
        
        record = task.evaluate(experiment, candidate)
        history.append(record)
        
        # Update best candidates for all scalarization weights
        for i, weight in enumerate(weights):
            current_score = task.scalarize(record, weight)
            if current_score > best_scores[i]:
                best_scores[i] = current_score
                best_candidates[i] = candidate

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "screened_multi_weight_hill_climbing",
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
