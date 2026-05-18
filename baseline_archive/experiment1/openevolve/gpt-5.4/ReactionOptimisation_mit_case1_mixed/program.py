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
    experiment = task.create_benchmark()
    history: list[dict] = []
    best_by_cat: dict[int, dict] = {}
    seen: set[tuple[int, float, float, float]] = set()
    conc_bounds = task.BOUNDS["conc_cat"]
    t_bounds = task.BOUNDS["t"]
    temp_bounds = task.BOUNDS["temperature"]
    cat_order = [2, 4, 0, 1, 5, 3, 6, 7]

    def clip(x: float, bounds: tuple[float, float]) -> float:
        return float(min(max(x, bounds[0]), bounds[1]))

    def make_candidate(cat: int, conc: float, t: float, temp: float) -> dict:
        return {
            "cat_index": int(cat),
            "conc_cat": clip(conc, conc_bounds),
            "t": clip(t, t_bounds),
            "temperature": clip(temp, temp_bounds),
        }

    def key(candidate: dict) -> tuple[int, float, float, float]:
        return (
            int(candidate["cat_index"]),
            round(float(candidate["conc_cat"]), 12),
            round(float(candidate["t"]), 6),
            round(float(candidate["temperature"]), 6),
        )

    def run(candidate: dict) -> bool:
        k = key(candidate)
        if k in seen or len(history) >= budget:
            return False
        seen.add(k)
        record = task.evaluate(experiment, candidate)
        history.append(record)
        cat = int(record["cat_index"])
        if cat not in best_by_cat or record["y"] > best_by_cat[cat]["y"]:
            best_by_cat[cat] = record
        return True

    span = conc_bounds[1] - conc_bounds[0]
    conc_low = conc_bounds[0] + 0.18 * span
    conc_high = conc_bounds[0] + 0.87 * span

    plan: list[dict] = []
    for cat in cat_order:
        plan.extend(
            [
                make_candidate(cat, conc_low, 565.0, 105.0),
                make_candidate(cat, conc_high, 600.0, 110.0),
            ]
        )

    for candidate in plan:
        if len(history) >= budget:
            break
        run(candidate)

    if len(history) < budget and best_by_cat:
        ranked = sorted(best_by_cat.values(), key=lambda row: row["y"], reverse=True)
        for row in ranked[:2]:
            cat = int(row["cat_index"])
            conc = float(row["conc_cat"])
            for delta, t, temp in [(-0.00045, 585.0, 108.0), (0.00045, 600.0, 110.0)]:
                if len(history) >= budget:
                    break
                run(make_candidate(cat, conc + delta, t, temp))

    i = 1
    while len(history) < budget and history:
        best = max(history, key=lambda row: row["y"])
        step = 0.0002 * i
        run(
            make_candidate(
                int(best["cat_index"]),
                float(best["conc_cat"]) + (step if i % 2 else -step),
                600.0 - 10.0 * (i % 3),
                110.0 - 1.0 * ((i // 2) % 3),
            )
        )
        i += 1

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "category_screen_trust_region",
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
