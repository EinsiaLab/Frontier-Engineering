from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# A-n32-k5 instance — must stay in sync with verification/evaluate.py
# ---------------------------------------------------------------------------
INSTANCE: dict[str, Any] = {
    "coords": [
        (82, 76),
        (96, 44), (50,  5), (49,  8), (13,  7), (29, 89),
        (58, 30), (84, 39), (14, 24), ( 2, 39), ( 3, 82),
        ( 5, 10), (98, 52), (84, 25), (61, 59), ( 1, 65),
        (88, 51), (91,  2), (19, 32), (93,  3), (50, 93),
        (98, 14), ( 5, 42), (42,  9), (61, 62), ( 9, 97),
        (80, 55), (57, 69), (23, 15), (20, 70), (85, 60),
        (98,  5),
    ],
    "demands": [
         0,
        19, 21,  6, 19,  7, 12, 16,  6, 16,  8,
        14, 21, 16,  3, 22, 18, 19,  1, 24,  8,
        12,  4,  8, 24, 24,  2, 20, 15,  2, 14,  9,
    ],
    "capacity": 100,
}

HUMAN_BEST: float = 784.0


def euc2d(p1: tuple, p2: tuple) -> int:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def _validate(routes: list, instance: dict) -> dict[str, Any]:
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    n = len(coords) - 1

    all_customers = set(range(1, n + 1))
    visited: set[int] = set()
    errors: list[str] = []
    total_dist = 0.0
    n_routes = 0

    for i, route in enumerate(routes):
        if not route:
            continue
        n_routes += 1
        load = 0
        for c in route:
            if not isinstance(c, int) or c < 1 or c > n:
                errors.append(f"Route {i}: invalid customer {c!r}")
                continue
            if c in visited:
                errors.append(f"Route {i}: customer {c} visited twice")
            visited.add(c)
            load += demands[c]
        if load > capacity:
            errors.append(f"Route {i}: capacity exceeded ({load} > {capacity})")
        path = [0] + list(route) + [0]
        for j in range(len(path) - 1):
            total_dist += euc2d(coords[path[j]], coords[path[j + 1]])

    missing = all_customers - visited
    if missing:
        errors.append(f"Missing customers: {sorted(missing)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_distance": total_dist,
        "n_routes": n_routes,
    }


def evaluate(program_path: str, *, repo_root: Path | None = None):
    start = time.time()
    repo_root = (repo_root or Path.cwd()).expanduser().resolve()
    program_path_p = Path(program_path).expanduser().resolve()

    benchmark_dir = (
        repo_root / "benchmarks" / "Transportation" / "CapacitatedVehicleRouting"
    ).resolve()
    if not benchmark_dir.is_dir():
        benchmark_dir = (
            repo_root / "Transportation" / "CapacitatedVehicleRouting"
        ).resolve()

    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "runtime_s": 0.0,
    }
    artifacts: dict[str, Any] = {}

    if not benchmark_dir.is_dir():
        artifacts["error_message"] = f"benchmark dir not found: {benchmark_dir}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    if not program_path_p.is_file():
        artifacts["error_message"] = f"program not found: {program_path_p}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    work_dir = Path(tempfile.mkdtemp(prefix="fe_cvrp_")).resolve()
    try:
        sandbox = (work_dir / "CapacitatedVehicleRouting").resolve()
        shutil.copytree(benchmark_dir, sandbox)

        # Replace baseline/init.py with the candidate
        candidate_dst = sandbox / "baseline" / "init.py"
        candidate_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(program_path_p, candidate_dst)

        # Dynamically load and run the candidate
        spec = importlib.util.spec_from_file_location(
            "fe_cvrp_candidate", candidate_dst
        )
        if spec is None or spec.loader is None:
            artifacts["error_message"] = "failed to load candidate module"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            artifacts["error_message"] = f"candidate import error: {exc}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        solve_fn = getattr(mod, "solve", None)
        if solve_fn is None:
            artifacts["error_message"] = "candidate missing 'solve' function"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        try:
            routes = solve_fn(INSTANCE)
        except Exception as exc:
            artifacts["error_message"] = f"solve() raised: {exc}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        result = _validate(routes, INSTANCE)
        dist = result["total_distance"]

        if result["valid"] and dist > 0:
            score = min(1.0, HUMAN_BEST / dist)
            metrics["valid"] = 1.0
            metrics["combined_score"] = score
            metrics["total_distance"] = dist
            metrics["n_routes"] = float(result["n_routes"])
            metrics["human_best_distance"] = HUMAN_BEST
        else:
            metrics["valid"] = 0.0
            metrics["combined_score"] = 0.0
            if result["errors"]:
                artifacts["error_message"] = "; ".join(result["errors"][:5])

        artifacts["evaluation_result"] = json.dumps(result, ensure_ascii=False)
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _wrap(metrics: dict[str, float], artifacts: dict[str, Any]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
