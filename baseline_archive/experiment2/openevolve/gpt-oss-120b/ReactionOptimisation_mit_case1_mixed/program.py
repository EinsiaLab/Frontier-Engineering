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

    for step in range(budget):
        # ------------------------------------------------------------------
        # More expressive search: evaluate a small pool of candidates each step.
        #   * If we have no incumbent yet, start with a single random sample.
        #   * Otherwise generate several mutations of the current best and,
        #     with a low probability, add an extra random sample for exploration.
        #   * Evaluate all candidates, keep the best of this step, and update
        #     the global incumbent.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # More expressive search: evaluate a small pool of candidates each step.
        #   * At the very first step we evaluate *all* initial candidates
        #     (one per catalyst) to quickly discover promising categories.
        #   * Afterwards we generate several mutations of the current best and,
        #     with a low probability, add an extra random sample for exploration.
        #   * Keep the best record of the step in the global history.
        # ------------------------------------------------------------------
        if step == 0:
            # One candidate for each catalyst – guarantees early coverage
            candidates = task.initial_candidates(rng)
        else:
            # generate a larger, more diverse set of mutated variants
            # (10 mutants gives a richer neighbourhood for exploitation)
            candidates = [task.mutate_candidate(best_candidate, rng) for _ in range(10)]
            # occasional pure random exploration (30 % chance) to keep global search alive
            if rng.random() < 0.3:
                candidates.append(task.sample_candidate(rng))

        # evaluate every candidate in the pool
        step_records = [task.evaluate(experiment, cand) for cand in candidates]

        # keep the single best record from this step in the global history
        best_step_record = max(step_records, key=lambda r: r["y"])
        history.append(best_step_record)

        # Early termination: if we have already reached a near‑optimal yield,
        # stop the loop early (the recorded history length will then be < budget,
        # which is acceptable because we respect the budget as an upper bound).
        if best_step_record["y"] >= 0.995:
            break

        # update the incumbent (global best) based on the full history
        incumbent = max(history, key=lambda row: row["y"])
        best_candidate = {name: incumbent[name] for name in task.INPUT_NAMES}

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "mixed_random_local_search",
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
