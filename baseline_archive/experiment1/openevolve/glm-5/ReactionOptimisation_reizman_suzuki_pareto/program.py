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
    seed_everything(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []
    
    # Phase 1: Dual-condition screening for Pareto diversity (8x2=16 experiments)
    for cat in task.CATEGORIES["catalyst"]:
        history.append(task.evaluate(experiment, {"catalyst": cat, "t_res": 360.0, "temperature": 110.0, "catalyst_loading": 2.5}))
    for cat in task.CATEGORIES["catalyst"]:
        history.append(task.evaluate(experiment, {"catalyst": cat, "t_res": 200.0, "temperature": 70.0, "catalyst_loading": 1.0}))
    
    # Phase 2: Multi-weight catalyst selection
    weights = [0.2, 0.5, 0.8]
    top_cats = set()
    for w in weights:
        top_cats.add(max(history, key=lambda r: task.scalarize(r, w))["catalyst"])
    top_cats.add(max(history, key=lambda r: r["yld"])["catalyst"])
    top_cats.add(min(history, key=lambda r: r["ton"])["catalyst"])
    
    # Phase 3: Refine top catalysts
    for cat in top_cats:
        for t_res in [200.0, 330.0, 500.0]:
            if len(history) >= budget: break
            history.append(task.evaluate(experiment, {"catalyst": cat, "t_res": t_res, "temperature": 110.0, "catalyst_loading": 2.5}))
        for temp in [70.0, 90.0]:
            if len(history) >= budget: break
            history.append(task.evaluate(experiment, {"catalyst": cat, "t_res": 200.0, "temperature": temp, "catalyst_loading": 1.0}))
        if len(history) >= budget: break
    
    # Phase 4: Fill remaining budget
    cat_best = max(history, key=lambda r: r["yld"])["catalyst"]
    while len(history) < budget:
        history.append(task.evaluate(experiment, {"catalyst": cat_best, "t_res": 268.0, "temperature": 110.0, "catalyst_loading": 2.5}))
    
    return {"task_name": task.TASK_NAME, "algorithm_name": "dual_screen_multiweight", "seed": seed, "budget": budget, "history": history[:budget], "summary": task.summarize(history[:budget])}
# EVOLVE-BLOCK-END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=task.DEFAULT_BUDGET)
    args = parser.parse_args()
    print(dump_json(solve(seed=args.seed, budget=args.budget)))


if __name__ == "__main__":
    main()
