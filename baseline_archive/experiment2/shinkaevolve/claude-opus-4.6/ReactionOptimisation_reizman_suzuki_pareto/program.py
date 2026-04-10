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
def _scalarized_score(record: dict, w_yld: float) -> float:
    """Weighted scalarization: w_yld on yield, (1-w_yld) on TON minimization."""
    yld = record.get("yld", 0.0)
    ton = record.get("ton", 200.0)
    obj_yld = np.clip(yld / 100.0, 0.0, 1.0)
    obj_ton = np.clip(1.0 - ton / 200.0, 0.0, 1.0)
    return w_yld * obj_yld + (1.0 - w_yld) * obj_ton


def _clamp(t_res, temperature, catalyst_loading):
    return (
        float(np.clip(t_res, 60.0, 600.0)),
        float(np.clip(temperature, 30.0, 110.0)),
        float(np.clip(catalyst_loading, 0.5, 2.5)),
    )


def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    catalysts = ["P1-L1", "P2-L1", "P1-L2", "P1-L3", "P1-L4", "P1-L5", "P1-L6", "P1-L7"]

    # ── Phase 1: Screen all catalysts (8 evals) ──
    screening_budget = min(len(catalysts), budget)
    cat_records = {}
    for i in range(screening_budget):
        candidate = {
            "catalyst": catalysts[i],
            "t_res": 300.0,
            "temperature": 100.0,
            "catalyst_loading": 1.5,
        }
        record = task.evaluate(experiment, candidate)
        history.append(record)
        cat_records[catalysts[i]] = record

    remaining = budget - screening_budget
    if remaining <= 0:
        return {
            "task_name": task.TASK_NAME,
            "algorithm_name": "pareto_local_search",
            "seed": seed,
            "budget": budget,
            "history": history,
            "summary": task.summarize(history),
        }

    # ── Select promising catalysts per weight ──
    weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # For each weight, rank catalysts and pick top ones
    promising_tasks = []  # list of (catalyst, weight)
    seen = set()
    for w in weights:
        ranked = sorted(
            cat_records.items(),
            key=lambda x: _scalarized_score(x[1], w),
            reverse=True,
        )
        # Top 2 catalysts for each weight
        for cat_name, _ in ranked[:3]:
            key = (cat_name, w)
            if key not in seen:
                seen.add(key)
                promising_tasks.append(key)

    # ── Phase 2: Targeted optimization ──
    # For each task, maintain best known point
    best_points = {}  # (cat, w) -> (score, t_res, temp, loading)

    # Initialize from screening
    for cat, rec in cat_records.items():
        for w in weights:
            sc = _scalarized_score(rec, w)
            key = (cat, w)
            if key not in best_points or sc > best_points[key][0]:
                best_points[key] = (sc, 300.0, 100.0, 1.5)

    # Also evaluate promising catalysts at extreme conditions for diversity
    # High yield conditions: high temp, long residence, high loading
    # Low TON conditions: low loading, short residence
    extreme_conditions = [
        (600.0, 110.0, 2.5),   # max yield push
        (60.0, 30.0, 0.5),     # min TON push
        (600.0, 110.0, 0.5),   # high yield, low loading (good TON)
        (120.0, 100.0, 2.5),   # short time, high temp, high loading
    ]

    # Pick top 2 catalysts overall (by combined score)
    combined_ranked = sorted(
        cat_records.items(),
        key=lambda x: _scalarized_score(x[1], 0.5),
        reverse=True,
    )
    top_cats_for_extremes = [c[0] for c in combined_ranked[:2]]

    # Also get best catalyst for pure yield and pure TON
    best_yld_cat = max(cat_records.items(), key=lambda x: x[1].get("yld", 0))[0]
    best_ton_cat = min(cat_records.items(), key=lambda x: x[1].get("ton", 200))[0]
    extreme_cats = list(set(top_cats_for_extremes + [best_yld_cat, best_ton_cat]))

    used = 0
    # Evaluate extreme conditions for diverse Pareto coverage
    for cat in extreme_cats:
        if used >= remaining:
            break
        for cond in extreme_conditions:
            if used >= remaining:
                break
            t_r, temp, load = cond
            candidate = {
                "catalyst": cat,
                "t_res": t_r,
                "temperature": temp,
                "catalyst_loading": load,
            }
            record = task.evaluate(experiment, candidate)
            history.append(record)
            used += 1

            # Update best points for all weights
            for w in weights:
                sc = _scalarized_score(record, w)
                key = (cat, w)
                if key not in best_points or sc > best_points[key][0]:
                    best_points[key] = (sc, t_r, temp, load)

    # ── Phase 3: Local search around best points for each weight ──
    # Prioritize tasks by spreading across weights
    # Create a round-robin schedule across weights
    task_schedule = []
    for w in weights:
        # Find best (cat, w) pair
        best_key = None
        best_sc = -1
        for key, val in best_points.items():
            if key[1] == w and val[0] > best_sc:
                best_sc = val[0]
                best_key = key
        if best_key is not None:
            task_schedule.append(best_key)

    if not task_schedule:
        task_schedule = list(promising_tasks[:6])

    iteration = 0
    while used < remaining:
        # Round robin through weight-targeted tasks
        cat, w = task_schedule[iteration % len(task_schedule)]

        best_sc, best_tres, best_temp, best_load = best_points.get(
            (cat, w), (0.0, 300.0, 100.0, 1.5)
        )

        # Adaptive perturbation: decrease over iterations
        decay = max(0.15, 1.0 - iteration * 0.05)

        # Perturbation with bias towards promising directions
        new_tres = best_tres + rng.normal(0, 120.0 * decay)
        new_temp = best_temp + rng.normal(0, 20.0 * decay)
        new_load = best_load + rng.normal(0, 0.5 * decay)

        new_tres, new_temp, new_load = _clamp(new_tres, new_temp, new_load)

        candidate = {
            "catalyst": cat,
            "t_res": new_tres,
            "temperature": new_temp,
            "catalyst_loading": new_load,
        }
        record = task.evaluate(experiment, candidate)
        history.append(record)
        used += 1
        iteration += 1

        # Update best points for ALL weights (cross-pollination)
        for w2 in weights:
            sc2 = _scalarized_score(record, w2)
            key2 = (cat, w2)
            if key2 not in best_points or sc2 > best_points[key2][0]:
                best_points[key2] = (sc2, new_tres, new_temp, new_load)

        # Also periodically refresh the schedule to track improvements
        if iteration % len(task_schedule) == 0:
            new_schedule = []
            for w_s in weights:
                best_key = None
                best_sc_s = -1
                for key, val in best_points.items():
                    if key[1] == w_s and val[0] > best_sc_s:
                        best_sc_s = val[0]
                        best_key = key
                if best_key is not None:
                    new_schedule.append(best_key)
            if new_schedule:
                task_schedule = new_schedule

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "pareto_local_search",
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