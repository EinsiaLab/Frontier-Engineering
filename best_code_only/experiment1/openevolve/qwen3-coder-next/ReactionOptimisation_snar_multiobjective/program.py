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
    best_candidate = None
    
    # Track Pareto points for diversity
    pareto_points: list[dict] = []
    
    # Adaptive exploration schedule with smoother decay
    def get_exploration_rate(step):
        return max(0.15, 0.4 * (1 - (step / budget) ** 1.5))
    
    # Dynamic weight selection based on Pareto spread
    def get_dynamic_weight():
        if len(pareto_points) < 3:
            return 0.5  # Neutral weight when we don't have enough info
        
        sty_values = [p["sty"] for p in pareto_points]
        eco_values = [p["e_factor"] for p in pareto_points]
        
        sty_range = max(sty_values) - min(sty_values)
        eco_range = max(eco_values) - min(eco_values)
        
        # Adjust weight based on which objective has more spread
        if sty_range + eco_range > 1e-10:
            weight = 0.5 + 0.1 * (sty_range - eco_range) / (sty_range + eco_range)
            return max(0.2, min(0.8, weight))
        return 0.5

    for step in range(budget):
        exploration_rate = get_exploration_rate(step)
        
        if not history or best_candidate is None or rng.random() < exploration_rate:
            candidate = task.sample_candidate(rng)
        else:
            # Select a random Pareto point for mutation to maintain diversity
            if pareto_points and rng.random() < 0.7:
                base_record = rng.choice(pareto_points)
                candidate = task.mutate_candidate({name: base_record[name] for name in task.INPUT_NAMES}, rng)
            else:
                candidate = task.mutate_candidate(best_candidate, rng)
            
            # Occasional random jump for exploration
            if rng.random() < 0.08:
                candidate = task.sample_candidate(rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)
        
        # Update best candidate using scalarization with dynamic weight
        weight = get_dynamic_weight() if len(pareto_points) >= 3 else 0.5 + 0.6 * (step % 10) / 9.0
        incumbent = max(history, key=lambda row: task.scalarize(row, weight))
        best_candidate = {name: incumbent[name] for name in task.INPUT_NAMES}
        
        # Maintain Pareto set for diversity using non-dominated sorting
        pareto_points = []
        for row in history:
            is_dominated = False
            for other in history:
                if other is row:
                    continue
                # Check if other dominates row
                if other["sty"] >= row["sty"] and other["e_factor"] <= row["e_factor"]:
                    if other["sty"] > row["sty"] or other["e_factor"] < row["e_factor"]:
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_points.append(row)
        
        # Limit Pareto set size while maintaining diversity - increase limit slightly
        if len(pareto_points) > 7:
            # Keep diverse points by sorting by both objectives
            sorted_by_sty = sorted(pareto_points, key=lambda x: x["sty"], reverse=True)
            sorted_by_eco = sorted(pareto_points, key=lambda x: x["e_factor"])
            selected = []
            # Select top points from both extremes - increase base selection
            for i in range(min(4, len(sorted_by_sty))):
                if sorted_by_sty[i] not in selected:
                    selected.append(sorted_by_sty[i])
            for i in range(min(4, len(sorted_by_eco))):
                if sorted_by_eco[i] not in selected:
                    selected.append(sorted_by_eco[i])
            pareto_points = selected[:7]

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
