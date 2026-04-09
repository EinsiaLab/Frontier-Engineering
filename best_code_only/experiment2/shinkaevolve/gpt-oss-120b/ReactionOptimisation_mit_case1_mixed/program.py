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
    """
    Category‑focused mixed random / local search.

    * First ``num_categories`` steps sample each catalyst category once.
    * Remaining steps exploit the category that has produced the highest yield so far,
      using the existing mutation operator for local refinement.
    """
    from math import inf

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []
    best_candidate = None

    # Track the best observed yield per catalyst category
    num_categories = len(task.CAT_INDEXES) if hasattr(task, "CAT_INDEXES") else 8
    cat_best_y: dict[int, float] = {i: -inf for i in range(num_categories)}
    # Store the best candidate (record) for each category
    cat_best_candidate: dict[int, dict] = {}
    # Per‑category mutation scale (sigma). Starts at 0.1 and will be decayed on improvement.
    cat_sigma: dict[int, float] = {i: 0.1 for i in range(num_categories)}
    # Per‑category counter of steps without improvement (used to adapt sigma)
    cat_no_improve_counter: dict[int, int] = {i: 0 for i in range(num_categories)}
    # Global sigma for cross‑category transfer of the overall best candidate
    global_sigma: float = 0.05

    for step in range(budget):
        if step < num_categories:
            # Exhaustive sweep: force a specific category
            candidate = task.sample_candidate(rng)
            candidate["cat_index"] = float(step)  # ensure correct type
        else:
            # Exploit categories based on observed yields
            # Rank categories descending by best observed yield
            sorted_cats = sorted(cat_best_y.items(), key=lambda kv: kv[1], reverse=True)
            best_cat = sorted_cats[0][0]
            # With 20 % probability consider the second‑best category (if available)
            if len(sorted_cats) > 1 and rng.random() < 0.2:
                best_cat = sorted_cats[1][0]

            # Decaying random injection probability: start ~30 % and decay to 2 %
            exploit_step = max(step - num_categories, 0)
            exploit_budget = max(budget - num_categories, 1)
            random_prob = max(0.02, 0.3 * (1.0 - exploit_step / exploit_budget))

            if best_candidate is None or rng.random() < random_prob:
                # Fresh random sample in the chosen category
                candidate = task.sample_candidate(rng)
                candidate["cat_index"] = float(best_cat)
            else:
                # Use the best candidate of the selected category (fallback to overall best)
                base = cat_best_candidate.get(best_cat, best_candidate)
                if base is None:
                    # No incumbent yet – fall back to a random sample
                    candidate = task.sample_candidate(rng)
                    candidate["cat_index"] = float(best_cat)
                else:
                    # 10 % probability: transfer the overall best candidate across categories
                    if rng.random() < 0.15:
                        candidate = best_candidate.copy()
                        candidate["conc_cat"] = max(0.0, candidate["conc_cat"] + rng.normal(0, global_sigma))
                        candidate["t"] = max(0.0, candidate["t"] + rng.normal(0, global_sigma))
                        candidate["temperature"] = max(0.0, candidate["temperature"] + rng.normal(0, global_sigma))
                        # keep original cat_index (no change)
                        global_sigma = max(0.005, global_sigma * 0.9)
                    elif rng.random() < 0.70:
                        candidate = base.copy()
                        # Adaptive sigma: increase with stagnation, cap at 0.5
                        sigma = min(0.5, cat_sigma[best_cat] * (1 + 0.2 * cat_no_improve_counter.get(best_cat, 0)))
                        candidate["conc_cat"] = max(0.0, candidate["conc_cat"] + rng.normal(0, sigma))
                        candidate["t"] = max(0.0, candidate["t"] + rng.normal(0, sigma))
                        candidate["temperature"] = max(0.0, candidate["temperature"] + rng.normal(0, sigma))
                        candidate["cat_index"] = float(best_cat)
                    else:
                        candidate = task.mutate_candidate(base, rng)
                        candidate["cat_index"] = float(best_cat)

        record = task.evaluate(experiment, candidate)
        history.append(record)

        # Update per‑category best yields and store best candidate per category
        cat_idx = int(record["cat_index"])
        if record["y"] > cat_best_y[cat_idx]:
            cat_best_y[cat_idx] = record["y"]
            # Save a copy of the record as the best candidate for this category
            cat_best_candidate[cat_idx] = {name: record[name] for name in task.INPUT_NAMES}
            # Decay the per‑category sigma after an improvement
            cat_sigma[cat_idx] = max(0.01, cat_sigma[cat_idx] * 0.9)
            # Reset stagnation counter for this category
            cat_no_improve_counter[cat_idx] = 0
        else:
            # Increment stagnation counter (capped to 10)
            cat_no_improve_counter[cat_idx] = min(cat_no_improve_counter[cat_idx] + 1, 10)

        # Keep track of the overall incumbent (best overall candidate)
        incumbent = max(history, key=lambda row: row["y"])
        best_candidate = {name: incumbent[name] for name in task.INPUT_NAMES}

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "mixed_category_focused_search",
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