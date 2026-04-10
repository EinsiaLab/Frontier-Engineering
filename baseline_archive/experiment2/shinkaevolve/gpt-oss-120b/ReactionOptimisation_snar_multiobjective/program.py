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
    """
    Adaptive random search that maintains a Pareto front.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []
    pareto_front: list[dict] = []  # non‑dominated records
    weights = [0.2, 0.5, 0.8]
    # Track the best record seen for each scalarisation weight
    best_per_weight: dict[float, dict] = {}

    def dominates(a: dict, b: dict) -> bool:
        """Return True if `a` dominates `b` (higher sty and lower e_factor)."""
        return (
            a["sty"] >= b["sty"]
            and a["e_factor"] <= b["e_factor"]
            and (a["sty"] > b["sty"] or a["e_factor"] < b["e_factor"])
        )

    def update_pareto(pareto: list[dict], record: dict) -> list[dict]:
        """Insert `record` into the Pareto front, removing any dominated entries."""
        # If the new record is dominated by any existing point, ignore it
        if any(dominates(p, record) for p in pareto):
            return pareto
        # Remove points that are dominated by the new record
        new_front = [p for p in pareto if not dominates(record, p)]
        new_front.append(record)
        return new_front

    # Initial diverse sampling to seed the Pareto front
    init_samples = min(5, max(1, budget // 4))
    for _ in range(init_samples):
        cand = task.sample_candidate(rng)
        rec = task.evaluate(experiment, cand)
        history.append(rec)
        pareto_front = update_pareto(pareto_front, rec)
        # Update best per weight with the initial sample
        for w in weights:
            cur = best_per_weight.get(w)
            if cur is None or task.scalarize(rec, w) > task.scalarize(cur, w):
                best_per_weight[w] = rec

    # Exploration probability decays over time (starts at 0.2, minimum 0.05)
    base_explore = 0.2

    for step in range(budget):
        # Linear decay of exploration probability
        explore_prob = max(0.05, base_explore * (1 - step / budget))
        weight = weights[step % len(weights)]

        # Decide whether to explore randomly or exploit via mutation of a Pareto point
        if not history or not pareto_front or rng.random() < explore_prob:
            candidate = task.sample_candidate(rng)
        else:
            parent_record = best_per_weight.get(weight, rng.choice(pareto_front))
            candidate = task.mutate_candidate(
                {name: parent_record[name] for name in task.INPUT_NAMES}, rng
            )
            if rng.random() < 0.05:
                candidate = task.sample_candidate(rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)

        # Update Pareto front with the new observation
        pareto_front = update_pareto(pareto_front, record)

        # Update the best record for each weight based on the new record
        for w in weights:
            cur = best_per_weight.get(w)
            if cur is None or task.scalarize(record, w) > task.scalarize(cur, w):
                best_per_weight[w] = record

        # Choose incumbent for the current scalarization weight from the Pareto front
        incumbent = max(pareto_front, key=lambda row: task.scalarize(row, weight))

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "adaptive_random_with_pareto",
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