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
    from benchmarks.ComputerSystems.DuckDBPreAggregationSelection.baseline.solution import (
        select_preaggregations as baseline_select_preaggregations,
    )
    from benchmarks.ComputerSystems.DuckDBPreAggregationSelection.runtime.problem import (
        HIDDEN_CASES,
        PUBLIC_CASES,
        evaluate_selection,
    )
except ModuleNotFoundError:
    from baseline.solution import select_preaggregations as baseline_select_preaggregations
    from runtime.problem import HIDDEN_CASES, PUBLIC_CASES, evaluate_selection


def _run_case(select_preaggregations, case):
    selection = select_preaggregations(dict(case))
    return evaluate_selection(selection, case)


def evaluate(program_path: str):
    metrics = {
        "combined_score": -1e18,
        "valid": 0.0,
        "public_avg_runtime_s": 0.0,
        "hidden_avg_runtime_s": 0.0,
        "baseline_hidden_avg_runtime_s": 0.0,
        "num_public_cases": 0.0,
        "num_hidden_cases": 0.0,
    }
    artifacts = {}
    namespace = runpy.run_path(str(Path(program_path).expanduser().resolve()), run_name="candidate_program")
    select_preaggregations = namespace.get("select_preaggregations")
    if not callable(select_preaggregations):
        artifacts["error_message"] = "candidate must define select_preaggregations(workload_manifest)"
        return metrics, artifacts
    try:
        public_candidate = [_run_case(select_preaggregations, case) for case in PUBLIC_CASES]
        hidden_candidate = [_run_case(select_preaggregations, case) for case in HIDDEN_CASES]
        hidden_baseline = [_run_case(baseline_select_preaggregations, case) for case in HIDDEN_CASES]
    except Exception:
        artifacts["error_message"] = traceback.format_exc()
        return metrics, artifacts

    hidden_avg = sum(float(item["candidate_total_runtime_s"]) for item in hidden_candidate) / len(hidden_candidate)
    baseline_hidden_avg = sum(float(item["candidate_total_runtime_s"]) for item in hidden_baseline) / len(hidden_baseline)
    public_avg = sum(float(item["candidate_total_runtime_s"]) for item in public_candidate) / len(public_candidate)
    if not math.isfinite(hidden_avg) or hidden_avg <= 0:
        artifacts["error_message"] = "candidate runtime is invalid"
        return metrics, artifacts

    metrics["valid"] = 1.0
    metrics["public_avg_runtime_s"] = public_avg
    metrics["hidden_avg_runtime_s"] = hidden_avg
    metrics["baseline_hidden_avg_runtime_s"] = baseline_hidden_avg
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
