from __future__ import annotations

import importlib.util
import time
from pathlib import Path


def evaluate(program_path: str, *, repo_root: Path | None = None):
    """Evaluate ACOPF candidate by calling benchmark verification script."""
    start = time.time()
    repo_root = (repo_root or Path.cwd()).expanduser().resolve()
    program_path_p = Path(program_path).expanduser().resolve()
    benchmark_dir = (
        repo_root / "benchmarks" / "PowerSystems" / "ACOPF"
    ).resolve()

    metrics: dict = {"combined_score": 0.0, "valid": 0.0, "runtime_s": 0.0}
    artifacts: dict = {}

    if not benchmark_dir.is_dir():
        artifacts["error_message"] = f"benchmark dir not found: {benchmark_dir}"
        metrics["runtime_s"] = time.time() - start
        return _wrap(metrics, artifacts)

    if not program_path_p.is_file():
        artifacts["error_message"] = f"program not found: {program_path_p}"
        metrics["runtime_s"] = time.time() - start
        return _wrap(metrics, artifacts)

    eval_path = benchmark_dir / "verification" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("fe_acopf_eval", str(eval_path))
    if spec is None or spec.loader is None:
        artifacts["error_message"] = f"failed to load evaluator: {eval_path}"
        metrics["runtime_s"] = time.time() - start
        return _wrap(metrics, artifacts)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate(
        str(program_path_p),
        output_dir=str(benchmark_dir / "output"),
    )

    metrics["combined_score"] = float(result.get("combined_score", 0.0))
    metrics["valid"] = float(result.get("valid", 0.0))
    metrics["runtime_s"] = float(time.time() - start)
    if "total_cost" in result:
        metrics["total_cost"] = float(result["total_cost"])
    if "error" in result:
        artifacts["error_message"] = str(result["error"])

    return _wrap(metrics, artifacts)


def _wrap(metrics: dict, artifacts: dict):
    try:
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception:
        return metrics
