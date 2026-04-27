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
    from benchmarks.Robotics.NarrowPassagePlanning.baseline.solution import plan_path as baseline_plan_path
    from benchmarks.Robotics.NarrowPassagePlanning.runtime.problem import HIDDEN_CASES, PUBLIC_CASES, _parse_grid, path_cost
except ModuleNotFoundError:
    from baseline.solution import plan_path as baseline_plan_path
    from runtime.problem import HIDDEN_CASES, PUBLIC_CASES, _parse_grid, path_cost


def _instance(case):
    grid, start, goal = _parse_grid(case["grid"])
    return {"grid": grid, "start": start, "goal": goal}


def _run_case(plan_path_fn, case):
    instance = _instance(case)
    return float(path_cost(instance, plan_path_fn(instance["grid"], instance["start"], instance["goal"])))


def evaluate(program_path: str):
    metrics = {"combined_score": -1e18, "valid": 0.0, "public_avg_cost": 0.0, "hidden_avg_cost": 0.0, "baseline_hidden_avg_cost": 0.0, "num_public_cases": 0.0, "num_hidden_cases": 0.0}
    artifacts = {}
    namespace = runpy.run_path(str(Path(program_path).expanduser().resolve()), run_name="candidate_program")
    plan_path_fn = namespace.get("plan_path")
    if not callable(plan_path_fn):
        artifacts["error_message"] = "candidate must define plan_path(grid, start, goal)"
        return metrics, artifacts
    try:
        public_costs = [_run_case(plan_path_fn, case) for case in PUBLIC_CASES]
        hidden_costs = [_run_case(plan_path_fn, case) for case in HIDDEN_CASES]
        baseline_hidden_costs = [_run_case(baseline_plan_path, case) for case in HIDDEN_CASES]
    except Exception:
        artifacts["error_message"] = traceback.format_exc()
        return metrics, artifacts
    hidden_avg = sum(hidden_costs) / len(hidden_costs)
    if not math.isfinite(hidden_avg) or hidden_avg <= 0:
        artifacts["error_message"] = "candidate cost is invalid"
        return metrics, artifacts
    metrics["valid"] = 1.0
    metrics["public_avg_cost"] = sum(public_costs) / len(public_costs)
    metrics["hidden_avg_cost"] = hidden_avg
    metrics["baseline_hidden_avg_cost"] = sum(baseline_hidden_costs) / len(baseline_hidden_costs)
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
