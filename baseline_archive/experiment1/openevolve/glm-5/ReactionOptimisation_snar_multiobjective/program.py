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
    pareto_archive: list[dict] = []
    
    def dominates(a: dict, b: dict) -> bool:
        better_any = False
        for name in task.OBJECTIVE_NAMES:
            maximize = task.OBJECTIVE_DIRECTIONS.get(name, True)
            a_val, b_val = a[name], b[name]
            if maximize:
                if a_val < b_val: return False
                if a_val > b_val: better_any = True
            else:
                if a_val > b_val: return False
                if a_val < b_val: better_any = True
        return better_any
    
    def update_pareto(archive: list, rec: dict) -> list:
        archive = [r for r in archive if not dominates(rec, r)]
        if not any(dominates(r, rec) for r in archive):
            archive.append(rec)
        return archive

    for step in range(budget):
        if not pareto_archive or rng.random() < 0.25:
            candidate = task.sample_candidate(rng)
        else:
            parent = {n: rng.choice(pareto_archive)[n] for n in task.INPUT_NAMES}
            candidate = task.mutate_candidate(parent, rng)
        
        record = task.evaluate(experiment, candidate)
        history.append(record)
        pareto_archive = update_pareto(pareto_archive, record)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "pareto_archive_search",
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
