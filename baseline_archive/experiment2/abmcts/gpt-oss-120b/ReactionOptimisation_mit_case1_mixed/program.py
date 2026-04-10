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

    # Assume three catalyst categories (0,1,2). Adjust if task defines otherwise.
    NUM_CATS = 3

    # Helper to perturb continuous variables of a candidate
    def perturb(candidate: dict, rng: np.random.Generator) -> dict:
        new_candidate = candidate.copy()
        # Use task.INPUT_BOUNDS if available, otherwise fallback to small relative noise
        bounds = getattr(task, "INPUT_BOUNDS", {})
        for name, value in candidate.items():
            if name == "cat_index":
                continue  # keep catalyst fixed
            if name in bounds:
                low, high = bounds[name]
                std = (high - low) * 0.05  # 5% of range
                new_val = value + rng.normal(0.0, std)
                # clip to bounds
                new_candidate[name] = float(np.clip(new_val, low, high))
            else:
                # fallback: relative Gaussian noise
                new_candidate[name] = float(value + rng.normal(0.0, 0.1 * abs(value) if value != 0 else 0.1))
        return new_candidate

    for step in range(budget):
        if step < NUM_CATS:
            # Early exploration: force each catalyst once
            candidate = task.sample_candidate(rng)
            if "cat_index" in candidate:
                candidate["cat_index"] = step % NUM_CATS
        else:
            if best_candidate is None:
                candidate = task.sample_candidate(rng)
            else:
                candidate = perturb(best_candidate, rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)

        # Update incumbent
        incumbent = max(history, key=lambda row: row["y"])
        best_candidate = {name: incumbent[name] for name in task.INPUT_NAMES}

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "mixed_explore_then_local",
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
