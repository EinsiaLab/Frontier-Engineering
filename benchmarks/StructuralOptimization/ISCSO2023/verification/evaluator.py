"""
Evaluator for ISCSO 2023 — 284-Member 3D Truss Sizing Optimization

This script:
1. Runs a candidate Python program that outputs submission.json
2. Loads the problem data from references/problem_data.json
3. Generates tower topology and performs 3D FEM analysis
4. Checks all constraints (stress, displacement, variable bounds)
5. Returns a score (weight if feasible, +inf otherwise)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root directory."""
    if "FRONTIER_ENGINEERING_ROOT" in os.environ:
        return Path(os.environ["FRONTIER_ENGINEERING_ROOT"]).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "frontier_eval").is_dir() and (parent / "benchmarks").is_dir():
            return parent
    return Path.cwd().resolve()


def _tail(text: str, limit: int = 8000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _truncate_middle(text: str, limit: int = 200_000) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, (limit - 128) // 2)
    omitted = len(text) - 2 * keep
    return text[:keep] + f"\n\n[... truncated {omitted} chars ...]\n\n" + text[-keep:]


def load_problem_data(repo_root: Path) -> dict:
    """Load the problem definition JSON."""
    candidates = [
        repo_root / "benchmarks" / "StructuralOptimization" / "ISCSO2023"
        / "references" / "problem_data.json",
        repo_root / "StructuralOptimization" / "ISCSO2023"
        / "references" / "problem_data.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"problem_data.json not found. Searched: {[str(p) for p in candidates]}"
    )


def build_fem_and_evaluate(
    solution_vector: list[float], problem: dict
) -> dict[str, Any]:
    """
    Run FEM analysis and check all constraints.

    Parameters
    ----------
    solution_vector : list of float
        Length-284 vector: [A_0..A_283].
    problem : dict
        Problem data loaded from JSON.

    Returns
    -------
    result : dict
        Evaluation results including objective, feasibility, violations.
    """
    # Late import
    fem_dir = Path(__file__).resolve().parent
    if str(fem_dir) not in sys.path:
        sys.path.insert(0, str(fem_dir))
    from fem_truss3d import TrussFEM3D, generate_tower_topology

    x = np.array(solution_vector, dtype=float)

    # --- Input validation ---
    expected_dim = problem["dimension"]
    if len(x) != expected_dim:
        return {
            "objective": float("inf"),
            "feasible": False,
            "error": f"Expected {expected_dim} variables, got {len(x)}",
            "score": float("inf"),
        }

    if not np.all(np.isfinite(x)):
        return {
            "objective": float("inf"),
            "feasible": False,
            "error": "Solution contains NaN or Inf values",
            "score": float("inf"),
        }

    areas = x
    bounds = problem["variable_bounds"]
    a_min, a_max = bounds["area_min"], bounds["area_max"]

    if np.any(areas < a_min - 1e-9) or np.any(areas > a_max + 1e-9):
        return {
            "objective": float("inf"),
            "feasible": False,
            "error": f"Area variables out of bounds [{a_min}, {a_max}]",
            "score": float("inf"),
        }

    areas = np.clip(areas, a_min, a_max)

    # --- Generate tower topology ---
    tp = problem["tower_parameters"]
    nodes, elements = generate_tower_topology(
        num_levels=tp["num_levels"],
        total_height=tp["total_height_mm"],
        bottom_half_width=tp["bottom_half_width_mm"],
        top_half_width=tp["top_half_width_mm"],
        cross_bracing_levels=tp["cross_bracing_levels"],
    )

    n_elements = len(elements)
    if n_elements != problem["num_bars"]:
        return {
            "objective": float("inf"),
            "feasible": False,
            "error": (
                f"Topology mismatch: generated {n_elements} bars, "
                f"expected {problem['num_bars']}"
            ),
            "score": float("inf"),
        }

    # --- Create FEM solver ---
    E = problem["material"]["E"]
    rho = problem["material"]["rho"]
    supports = problem["supports"]

    try:
        fem = TrussFEM3D(nodes, elements, E, supports)
    except Exception as exc:
        return {
            "objective": float("inf"),
            "feasible": False,
            "error": f"FEM setup failed: {exc}",
            "score": float("inf"),
        }

    # --- Evaluate all load cases ---
    constraints = problem["constraints"]
    sigma_limit = constraints["stress_limit"]
    disp_limit = constraints["displacement_limit"]
    tol = constraints.get("tolerance", 1e-6)

    max_stress_vio = 0.0
    max_disp_vio = 0.0

    for lc in problem["load_cases"]:
        force_vec = np.zeros(3 * problem["num_nodes"])
        for load in lc["loads"]:
            nid = load["node"]
            force_vec[3 * nid] += load["fx"]
            force_vec[3 * nid + 1] += load["fy"]
            force_vec[3 * nid + 2] += load["fz"]

        try:
            displacements, stresses = fem.solve(areas, force_vec)
        except Exception as exc:
            return {
                "objective": float("inf"),
                "feasible": False,
                "error": f"FEM solve failed for LC {lc['id']}: {exc}",
                "score": float("inf"),
            }

        # Check stress constraints
        abs_stresses = np.abs(stresses)
        stress_violations = abs_stresses - sigma_limit
        lc_max_stress_vio = float(np.max(stress_violations))
        max_stress_vio = max(max_stress_vio, lc_max_stress_vio)

        # Check displacement constraints
        abs_disp = np.abs(displacements)
        disp_violations = abs_disp - disp_limit
        lc_max_disp_vio = float(np.max(disp_violations))
        max_disp_vio = max(max_disp_vio, lc_max_disp_vio)

    # --- Compute objective ---
    weight = fem.compute_weight(areas, rho)

    # --- Feasibility ---
    feasible = (max_stress_vio <= tol) and (max_disp_vio <= tol)

    return {
        "objective": float(weight),
        "feasible": bool(feasible),
        "max_stress_violation": float(max(max_stress_vio, 0.0)),
        "max_displacement_violation": float(max(max_disp_vio, 0.0)),
        "score": float(weight) if feasible else float("inf"),
        "num_load_cases": len(problem["load_cases"]),
    }


def evaluate(program_path: str, *, repo_root: Path | None = None) -> Any:
    """
    Full evaluation pipeline:
    1. Run candidate program to produce submission.json
    2. Parse and validate submission
    3. Run FEM + constraint check
    4. Return metrics
    """
    start = time.time()
    repo_root = (
        _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    )
    program_path_resolved = str(Path(program_path).expanduser().resolve())

    work_dir = Path(tempfile.mkdtemp(prefix="fe_iscso2023_")).resolve()
    artifacts: dict[str, str] = {}

    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "weight_kg": 0.0,
        "valid": 0.0,
        "feasible": 0.0,
        "timeout": 0.0,
        "runtime_s": 0.0,
    }

    try:
        # Copy problem data for the solver
        problem = load_problem_data(repo_root)
        refs_dir = work_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        with open(refs_dir / "problem_data.json", "w", encoding="utf-8") as f:
            json.dump(problem, f)

        # Run candidate program
        try:
            proc = subprocess.run(
                [sys.executable, program_path_resolved],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=1200,
            )
        except subprocess.TimeoutExpired as e:
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"program timeout: {e}"
            return _wrap(metrics, artifacts)

        artifacts["program_stdout"] = _tail(proc.stdout)
        artifacts["program_stderr"] = _tail(proc.stderr)
        artifacts["program_stdout_full"] = _truncate_middle(proc.stdout)
        artifacts["program_stderr_full"] = _truncate_middle(proc.stderr)
        metrics["program_returncode"] = float(proc.returncode)

        # Read submission
        submission_path = work_dir / "submission.json"
        if not submission_path.exists():
            artifacts["error_message"] = "submission.json not generated"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        try:
            with open(submission_path, "r", encoding="utf-8") as f:
                submission = json.load(f)
            artifacts["submission.json"] = json.dumps(submission, indent=2)
        except Exception as exc:
            artifacts["error_message"] = f"Failed to parse submission.json: {exc}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        if "solution_vector" not in submission:
            artifacts["error_message"] = "submission.json missing 'solution_vector'"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        # Evaluate
        result = build_fem_and_evaluate(submission["solution_vector"], problem)
        artifacts["evaluation_result"] = json.dumps(result, indent=2)

        runtime_s = time.time() - start
        metrics["weight_kg"] = result.get("objective", 0.0)
        metrics["runtime_s"] = float(runtime_s)
        metrics["feasible"] = 1.0 if result.get("feasible", False) else 0.0
        metrics["max_stress_violation"] = result.get("max_stress_violation", 0.0)
        metrics["max_displacement_violation"] = result.get(
            "max_displacement_violation", 0.0
        )

        if result.get("feasible", False):
            # Minimization: negate weight so higher combined_score = better
            metrics["combined_score"] = -float(result["objective"])
            metrics["valid"] = 1.0
        else:
            # Invalid: large negative so it's always worse than any feasible solution
            metrics["combined_score"] = -1e18
            metrics["valid"] = 0.0

        return _wrap(metrics, artifacts)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]) -> Any:
    try:
        from openevolve.evaluation_result import EvaluationResult

        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception:
        return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>")
        print("  or:  python evaluator.py --test <submission.json>")
        sys.exit(1)

    if sys.argv[1] == "--test" and len(sys.argv) >= 3:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            sub = json.load(f)
        repo = _find_repo_root()
        prob = load_problem_data(repo)
        result = build_fem_and_evaluate(sub["solution_vector"], prob)
        print(json.dumps(result, indent=2))
    else:
        result = evaluate(sys.argv[1])
        print(json.dumps(result, indent=2))

