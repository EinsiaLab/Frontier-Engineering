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
def _local_mutate(base: dict, rng: np.random.Generator, scale: float = 0.15) -> dict:
    """Mutate continuous variables with controllable scale, keep cat_index."""
    from shared.utils import clamp
    proposal = dict(base)
    for name, (low, high) in task.BOUNDS.items():
        span = high - low
        step = float(rng.normal(0.0, scale * span))
        proposal[name] = clamp(float(base[name]) + step, low, high)
    return proposal


def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    def eval_and_record(cand):
        rec = task.evaluate(experiment, cand)
        history.append(rec)
        return rec

    # Phase 1: Try ALL 8 catalysts at the optimal corner (max conc, max t, max temp)
    # This is the most efficient use of budget since the optimum is at the boundary
    n_cats = len(task.CATEGORIES["cat_index"])
    cat_best: dict[int, tuple[float, dict]] = {}
    n_corner = min(n_cats, budget)

    for i in range(n_corner):
        cat_idx = task.CATEGORIES["cat_index"][i]
        candidate = {
            "conc_cat": 0.004175,
            "t": 600.0,
            "temperature": 110.0,
            "cat_index": int(cat_idx),
        }
        rec = eval_and_record(candidate)
        y = rec["y"]
        cand_dict = {name: rec[name] for name in task.INPUT_NAMES}
        cat_best[cat_idx] = (y, cand_dict)

    # Phase 3: Intensify around the best candidates
    remaining = budget - len(history)
    all_results = sorted(
        [(rec["y"], {name: rec[name] for name in task.INPUT_NAMES}) for rec in history],
        key=lambda x: x[0], reverse=True
    )
    top_k = all_results[:min(3, len(all_results))]

    for step in range(remaining):
        progress = step / max(remaining, 1)
        scale = 0.10 * (1.0 - progress) + 0.02 * progress  # tighter perturbation

        r = rng.random()
        if r < 0.05:
            # Rare random exploration (Phase 1 already covered catalysts)
            candidate = task.sample_candidate(rng)
        elif r < 0.18 and len(top_k) > 1:
            # Crossover: take cat_index from best, continuous from another top
            best_c = top_k[0][1]
            other_idx = rng.integers(1, len(top_k))
            other_c = top_k[other_idx][1]
            candidate = {
                "cat_index": int(best_c["cat_index"]),
                "conc_cat": float(other_c["conc_cat"]),
                "t": float(other_c["t"]),
                "temperature": float(other_c["temperature"]),
            }
            candidate = _local_mutate(candidate, rng, scale=scale)
        else:
            # Mutate from top candidates (weighted toward best)
            weights = np.array([1.0 / (i + 1) ** 1.5 for i in range(len(top_k))])
            weights /= weights.sum()
            idx = rng.choice(len(top_k), p=weights)
            base = top_k[idx][1]
            candidate = _local_mutate(base, rng, scale=scale)
            # Strongly bias toward optimal corner (max temp, max t, max conc)
            from shared.utils import clamp
            if rng.random() < 0.55:
                candidate["temperature"] = clamp(
                    float(candidate["temperature"]) + abs(float(rng.normal(0, 18))),
                    task.BOUNDS["temperature"][0], task.BOUNDS["temperature"][1]
                )
            if rng.random() < 0.35:
                candidate["t"] = clamp(
                    float(candidate["t"]) + abs(float(rng.normal(0, 45))),
                    task.BOUNDS["t"][0], task.BOUNDS["t"][1]
                )
            if rng.random() < 0.30:
                candidate["conc_cat"] = clamp(
                    float(candidate["conc_cat"]) + abs(float(rng.normal(0, 0.0007))),
                    task.BOUNDS["conc_cat"][0], task.BOUNDS["conc_cat"][1]
                )
            # Very small chance to switch catalyst
            if rng.random() < 0.02:
                candidate["cat_index"] = int(rng.choice(task.CATEGORIES["cat_index"]))

        rec = eval_and_record(candidate)
        y = rec["y"]
        cand_dict = {name: rec[name] for name in task.INPUT_NAMES}
        # Update top_k
        if len(top_k) < 3:
            top_k.append((y, cand_dict))
            top_k.sort(key=lambda x: x[0], reverse=True)
        elif y > top_k[-1][0]:
            top_k[-1] = (y, cand_dict)
            top_k.sort(key=lambda x: x[0], reverse=True)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "structured_cat_refine",
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
