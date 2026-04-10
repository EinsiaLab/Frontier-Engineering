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

    # ----------------------------------------------------------------------
    # Pareto archive of non‑dominated solutions.
    # ----------------------------------------------------------------------
    archive: list[dict] = []

    # Helper to test Pareto domination (larger sty is better, smaller e_factor is better)
    def dominates(a: dict, b: dict) -> bool:
        better_or_equal = (
            a["sty"] >= b["sty"] and a["e_factor"] <= b["e_factor"]
        )
        strictly_better = a["sty"] > b["sty"] or a["e_factor"] < b["e_factor"]
        return better_or_equal and strictly_better

    for step in range(budget):
        weight = rng.random()

        # --------------------------------------------------------------
        # Candidate generation – mix of exploration, archive exploitation,
        # and mutation of the current best scalarised incumbent.
        # --------------------------------------------------------------
        if not history or best_candidate is None:
            # First iteration – pure random sample
            candidate = task.sample_candidate(rng)

        elif archive and rng.random() < 0.4:
            # Exploit a random member of the Pareto archive (inputs only)
            elite = rng.choice(archive)
            # Mutate the elite *and* keep only the input variables.
            # The elite contains objective values; keeping them would
            # create duplicate column names in the experiment dataset.
            mutated = task.mutate_candidate(elite, rng)
            candidate = {name: mutated[name] for name in task.INPUT_NAMES}

        elif rng.random() < 0.3:
            # Pure exploration
            candidate = task.sample_candidate(rng)

        else:
            # Exploit the current best scalarised solution
            candidate = task.mutate_candidate(best_candidate, rng)
            # Occasionally force a fresh random sample to keep diversity
            if rng.random() < 0.15:
                candidate = task.sample_candidate(rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)

        # --------------------------------------------------------------
        # Update Pareto archive with the new record
        # --------------------------------------------------------------
        new_archive: list[dict] = []
        dominated = False
        for existing in archive:
            if dominates(record, existing):
                # record dominates existing → drop existing
                continue
            if dominates(existing, record):
                # existing dominates record → keep existing, discard record
                dominated = True
            new_archive.append(existing)
        if not dominated:
            new_archive.append(record)
        archive = new_archive

        # --------------------------------------------------------------
        # Update the best scalarised incumbent for the next mutation step
        # --------------------------------------------------------------
        incumbent = max(history, key=lambda row: task.scalarize(row, weight))
        best_candidate = {name: incumbent[name] for name in task.INPUT_NAMES}

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
