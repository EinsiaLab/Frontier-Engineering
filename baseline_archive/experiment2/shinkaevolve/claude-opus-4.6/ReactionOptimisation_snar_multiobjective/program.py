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

    bounds = {
        "tau": (0.5, 2.0),
        "equiv_pldn": (1.0, 5.0),
        "conc_dfnb": (0.1, 0.5),
        "temperature": (30.0, 120.0),
    }
    input_names = list(bounds.keys())
    weights = [0.2, 0.5, 0.8]
    best_per_weight: dict[float, tuple[dict, float]] = {}

    for step in range(budget):
        w = weights[step % len(weights)]

        if step < 3 or rng.random() < 0.3:
            candidate = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in bounds.items()}
        else:
            info = best_per_weight.get(w)
            if info is None:
                candidate = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in bounds.items()}
            else:
                base = info[0]
                candidate = {}
                for name, (lo, hi) in bounds.items():
                    scale = (hi - lo) * 0.15
                    val = base[name] + rng.normal(0, scale)
                    candidate[name] = float(np.clip(val, lo, hi))

        record = task.evaluate(experiment, candidate)
        history.append(record)

        for ww in weights:
            score = task.scalarize(record, ww)
            if ww not in best_per_weight or score > best_per_weight[ww][1]:
                cand = {n: record[n] for n in input_names}
                best_per_weight[ww] = (cand, score)

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
