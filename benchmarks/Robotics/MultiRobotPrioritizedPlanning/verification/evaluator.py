from __future__ import annotations

import argparse
import json
import math
import runpy
import traceback
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "benchmarks").is_dir() and (parent / "frontier_eval").is_dir():
            return parent
    return Path.cwd().resolve()


def _benchmark_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_path() -> None:
    import sys

    for p in (_repo_root(), _benchmark_root()):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


_ensure_import_path()

try:
    from benchmarks.Robotics.MultiRobotPrioritizedPlanning.baseline.solution import plan_paths as baseline_plan_paths_fn
    from benchmarks.Robotics.MultiRobotPrioritizedPlanning.runtime.problem import HIDDEN_CASES, PUBLIC_CASES, _parse_grid, makespan, total_cost
except ModuleNotFoundError:
    from baseline.solution import plan_paths as baseline_plan_paths_fn
    from runtime.problem import HIDDEN_CASES, PUBLIC_CASES, _parse_grid, makespan, total_cost


def _instance(case):
    grid, robot_ids, starts, goals = _parse_grid(case["grid"])
    return {"grid": grid, "robot_ids": robot_ids, "starts": starts, "goals": goals}


def _run_case(plan_paths_fn, case):
    instance = _instance(case)
    solution = plan_paths_fn(instance["grid"], instance["starts"], instance["goals"])
    return float(total_cost(instance, solution)), float(makespan(instance, solution))


def evaluate(program_path: str) -> tuple[dict[str, float], dict[str, str]]:
    metrics = {
        "combined_score": -1e18,
        "valid": 0.0,
        "public_avg_total_cost": 0.0,
        "hidden_avg_total_cost": 0.0,
        "baseline_hidden_avg_total_cost": 0.0,
        "hidden_avg_makespan": 0.0,
        "num_public_cases": 0.0,
        "num_hidden_cases": 0.0,
    }
    artifacts: dict[str, str] = {}
    namespace = runpy.run_path(str(Path(program_path).expanduser().resolve()), run_name="candidate_program")
    plan_paths_fn = namespace.get("plan_paths")
    if not callable(plan_paths_fn):
        artifacts["error_message"] = "candidate must define plan_paths(grid, starts, goals)"
        return metrics, artifacts
    try:
        public_pairs = [_run_case(plan_paths_fn, case) for case in PUBLIC_CASES]
        hidden_pairs = [_run_case(plan_paths_fn, case) for case in HIDDEN_CASES]
        baseline_hidden_pairs = [_run_case(baseline_plan_paths_fn, case) for case in HIDDEN_CASES]
    except Exception:
        artifacts["error_message"] = traceback.format_exc()
        return metrics, artifacts
    hidden_avg = sum(pair[0] for pair in hidden_pairs) / len(hidden_pairs)
    if not math.isfinite(hidden_avg) or hidden_avg <= 0:
        artifacts["error_message"] = "candidate total cost is invalid"
        return metrics, artifacts
    metrics["valid"] = 1.0
    metrics["public_avg_total_cost"] = sum(pair[0] for pair in public_pairs) / len(public_pairs)
    metrics["hidden_avg_total_cost"] = hidden_avg
    metrics["baseline_hidden_avg_total_cost"] = sum(pair[0] for pair in baseline_hidden_pairs) / len(baseline_hidden_pairs)
    metrics["hidden_avg_makespan"] = sum(pair[1] for pair in hidden_pairs) / len(hidden_pairs)
    metrics["num_public_cases"] = float(len(PUBLIC_CASES))
    metrics["num_hidden_cases"] = float(len(HIDDEN_CASES))
    metrics["combined_score"] = -hidden_avg
    return metrics, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("program")
    parser.add_argument("--metrics-out", default="metrics.json")
    args = parser.parse_args()
    metrics, artifacts = evaluate(args.program)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if artifacts:
        Path("artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
