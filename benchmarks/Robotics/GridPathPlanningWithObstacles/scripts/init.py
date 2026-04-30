#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def _is_repo_root(path: Path) -> bool:
    return (path / "benchmarks").is_dir() and (path / "frontier_eval").is_dir()


def _ensure_import_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if _is_repo_root(parent):
            ps = str(parent)
            if ps not in sys.path:
                sys.path.insert(0, ps)
            return
    benchmark_root = here.parents[1]
    ps = str(benchmark_root)
    if ps not in sys.path:
        sys.path.insert(0, ps)


_ensure_import_path()

try:
    from benchmarks.Robotics.GridPathPlanningWithObstacles.baseline.solution import plan_path as _baseline_plan_path
    from benchmarks.Robotics.GridPathPlanningWithObstacles.runtime.problem import load_instance, path_cost
except ModuleNotFoundError:
    from baseline.solution import plan_path as _baseline_plan_path
    from runtime.problem import load_instance, path_cost


# EVOLVE-BLOCK-START
def plan_path(grid, start, goal):
    return _baseline_plan_path(grid, start, goal)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    instance = load_instance()
    print(path_cost(instance, plan_path(instance["grid"], instance["start"], instance["goal"])))
