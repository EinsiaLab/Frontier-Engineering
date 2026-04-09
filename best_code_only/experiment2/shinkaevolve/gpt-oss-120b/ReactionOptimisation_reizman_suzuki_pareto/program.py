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
    """
    Two‑stage random search:
    1. Screen each catalyst (up to one random evaluation per catalyst or 1/3 of the budget).
    2. Spend the remaining budget on the two best catalysts discovered.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    # --- Phase 1: screening -------------------------------------------------
    # Allocate at most one evaluation per catalyst (or 1/3 of the budget, whichever is smaller)
    screening_steps = min(budget // 3, 8)  # 8 is the known number of catalysts
    best_score_per_catalyst: dict[str, float] = {}
    # store the full record of the best candidate per catalyst (for local perturbation)
    best_record_per_catalyst: dict[str, dict] = {}

    for step in range(budget):
        if step < screening_steps:
            # Purely random candidate for screening
            candidate = task.sample_candidate(rng)
        else:
            # --- Phase 2: focused search ------------------------------------
            if best_score_per_catalyst:
                # pick the top‑3 catalysts (or fewer if not enough data) and sample
                # proportionally to their best scores to bias towards the most promising.
                top_k = 3
                # sort catalysts by their best score (descending)
                sorted_cats = sorted(
                    best_score_per_catalyst.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                top_catalysts = [c for c, _ in sorted_cats[:top_k]]
                # compute selection probabilities (higher score → higher probability)
                scores = np.array([best_score_per_catalyst[c] for c in top_catalysts])
                # shift scores to be non‑negative
                probs = scores - scores.min() + 1e-6
                probs = probs / probs.sum()
                # choose a catalyst from the weighted top list
                chosen_cat = rng.choice(top_catalysts, p=probs)

                # start from a fresh random candidate and then replace its catalyst
                candidate = task.sample_candidate(rng)
                candidate["catalyst"] = chosen_cat

                # If we have a good record for this catalyst, perturb its continuous variables
                best_rec = best_record_per_catalyst.get(chosen_cat)
                if best_rec is not None:
                    for var in ("t_res", "temperature", "catalyst_loading"):
                        base_val = best_rec[var]
                        # apply a finer Gaussian perturbation (≈5 % of the value)
                        perturb = rng.normal(0.0, 0.05) * base_val
                        candidate[var] = max(0.0, base_val + perturb)  # keep values non‑negative
            else:
                # fallback to pure random if no data yet
                candidate = task.sample_candidate(rng)

        record = task.evaluate(experiment, candidate)
        history.append(record)

        # Update best scalarised score for the catalyst seen in this record.
        # Simple linear combination: higher yield and lower ton are better.
        try:
            score = float(record.get("yld", 0)) - float(record.get("ton", 0))
        except Exception:
            score = 0.0
        cat = record.get("catalyst")
        if isinstance(cat, str):
            # Update best scalarised score used for catalyst ranking
            if cat not in best_score_per_catalyst or score > best_score_per_catalyst[cat]:
                best_score_per_catalyst[cat] = score
            # Update best full record for perturbation in phase 2
            prev_best = best_record_per_catalyst.get(cat)
            if prev_best is None or score > (
                float(prev_best.get("yld", 0)) - float(prev_best.get("ton", 0))
            ):
                best_record_per_catalyst[cat] = record

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "two_stage_random_search",
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