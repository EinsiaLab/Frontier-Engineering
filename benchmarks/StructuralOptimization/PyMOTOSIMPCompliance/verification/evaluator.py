"""Evaluator for PyMOTO SIMP compliance benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pymoto as pym

VOL_TOL = 1e-3


def _find_repo_root(start: Path | None = None) -> Path:
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_problem_config(repo_root: Path) -> dict[str, Any]:
    candidates = [
        repo_root / "benchmarks" / "StructuralOptimization" / "PyMOTOSIMPCompliance" / "references" / "problem_config.json",
        repo_root / "StructuralOptimization" / "PyMOTOSIMPCompliance" / "references" / "problem_config.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"problem_config.json not found. Searched: {[str(p) for p in candidates]}")


def _build_domain_boundary_force(problem: dict[str, Any]) -> tuple[pym.VoxelDomain, np.ndarray, np.ndarray]:
    domain = pym.VoxelDomain(int(problem["nelx"]), int(problem["nely"]))

    left_nodes = domain.nodes[0, :].flatten()
    boundary_dofs = np.concatenate([left_nodes * 2, left_nodes * 2 + 1]).astype(int)

    force = np.zeros(domain.nnodes * 2, dtype=float)
    load_node = int(domain.nodes[int(problem["nelx"]), int(problem["nely"] // 2), 0])
    load_dof = 2 * load_node + int(problem.get("force_direction", 1))
    force[load_dof] = float(problem.get("force_magnitude", 1.0))

    return domain, boundary_dofs, force


def _compute_compliance(density: np.ndarray, problem: dict[str, Any]) -> float:
    domain, boundary_dofs, force = _build_domain_boundary_force(problem)

    sx = pym.Signal("x_eval", state=density.flatten())
    with pym.Network() as func:
        simp = pym.MathExpression(
            expression=f"{problem['Emin']} + {problem['E0'] - problem['Emin']}*inp0^{problem['penal']}"
        )(sx)
        stiffness = pym.AssembleStiffness(
            domain=domain,
            bc=boundary_dofs,
            e_modulus=1.0,
            poisson_ratio=float(problem["nu"]),
        )(simp)
        displacement = pym.LinSolve(symmetric=True, positive_definite=True)(stiffness, force)
        compliance = pym.EinSum(expression="i,i->")(displacement, force)

    func.response()
    return float(compliance.state)


def _evaluate_density(density_vector: list[float], problem: dict[str, Any]) -> dict[str, Any]:
    nelx = int(problem["nelx"])
    nely = int(problem["nely"])
    rho_min = float(problem.get("rho_min", 1e-3))

    arr = np.asarray(density_vector, dtype=float)
    expected_len = nelx * nely
    if arr.size != expected_len:
        return {
            "compliance": float("inf"),
            "volume_fraction": 0.0,
            "feasible": False,
            "error": f"Expected {expected_len} densities, got {arr.size}",
        }

    if not np.all(np.isfinite(arr)):
        return {
            "compliance": float("inf"),
            "volume_fraction": 0.0,
            "feasible": False,
            "error": "Density vector contains non-finite values",
        }

    density = np.clip(arr.reshape((nely, nelx)), rho_min, 1.0)
    volume_fraction = float(np.mean(density))
    feasible = bool(volume_fraction <= float(problem["volfrac"]) + float(VOL_TOL))

    try:
        compliance = _compute_compliance(density, problem)
    except Exception as exc:
        return {
            "compliance": float("inf"),
            "volume_fraction": volume_fraction,
            "feasible": False,
            "error": f"Compliance evaluation failed: {exc}",
        }

    return {
        "compliance": float(compliance),
        "volume_fraction": volume_fraction,
        "feasible": feasible,
    }


def evaluate(program_path: str, *, repo_root: Path | None = None, timeout_s: float = 300.0) -> Any:
    start_s = time.time()
    repo = _find_repo_root() if repo_root is None else repo_root.resolve()
    benchmark_dir = (repo / "benchmarks" / "StructuralOptimization" / "PyMOTOSIMPCompliance").resolve()

    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "runtime_s": 0.0,
        "timeout": 0.0,
    }
    artifacts: dict[str, Any] = {
        "benchmark_dir": str(benchmark_dir),
        "candidate_program": str(Path(program_path).expanduser().resolve()),
    }

    problem = _load_problem_config(repo)

    uniform_density = np.full(int(problem["nelx"] * problem["nely"]), float(problem["volfrac"]), dtype=float)
    baseline_eval = _evaluate_density(uniform_density.tolist(), problem)
    baseline_compliance = _safe_float(baseline_eval.get("compliance"), default=float("inf"))

    candidate_program = Path(program_path).expanduser().resolve()
    if not candidate_program.is_file():
        artifacts["error_message"] = f"Candidate program not found: {candidate_program}"
        metrics["runtime_s"] = float(time.time() - start_s)
        return _wrap(metrics, artifacts)

    submission_path = benchmark_dir / "temp" / "submission.json"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    if submission_path.is_file():
        submission_path.unlink()

    cmd = [sys.executable, str(candidate_program)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(benchmark_dir),
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_s)),
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        metrics["timeout"] = 1.0
        artifacts["error_message"] = f"Candidate timed out: {exc}"
        metrics["runtime_s"] = float(time.time() - start_s)
        return _wrap(metrics, artifacts)
    except Exception as exc:
        artifacts["error_message"] = f"Failed to run candidate: {exc}"
        metrics["runtime_s"] = float(time.time() - start_s)
        return _wrap(metrics, artifacts)

    artifacts["candidate_stdout"] = _tail(proc.stdout)
    artifacts["candidate_stderr"] = _tail(proc.stderr)
    artifacts["candidate_stdout_full"] = _truncate_middle(proc.stdout)
    artifacts["candidate_stderr_full"] = _truncate_middle(proc.stderr)
    metrics["candidate_returncode"] = float(proc.returncode)

    if proc.returncode != 0:
        artifacts["error_message"] = f"Candidate exited with code {proc.returncode}"
        metrics["runtime_s"] = float(time.time() - start_s)
        return _wrap(metrics, artifacts)

    if not submission_path.is_file():
        artifacts["error_message"] = f"Missing submission file: {submission_path}"
        metrics["runtime_s"] = float(time.time() - start_s)
        return _wrap(metrics, artifacts)

    try:
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
    except Exception as exc:
        artifacts["error_message"] = f"Invalid submission JSON: {exc}"
        metrics["runtime_s"] = float(time.time() - start_s)
        return _wrap(metrics, artifacts)

    density_vector = submission.get("density_vector")
    if not isinstance(density_vector, list):
        artifacts["error_message"] = "submission.json must contain list field `density_vector`"
        metrics["runtime_s"] = float(time.time() - start_s)
        return _wrap(metrics, artifacts)

    eval_result = _evaluate_density(density_vector, problem)
    compliance = _safe_float(eval_result.get("compliance"), default=float("inf"))
    volume_fraction = _safe_float(eval_result.get("volume_fraction"), default=0.0)
    feasible = bool(eval_result.get("feasible", False))

    metrics["compliance"] = compliance
    metrics["volume_fraction"] = volume_fraction
    metrics["feasible"] = 1.0 if feasible else 0.0
    metrics["baseline_uniform_compliance"] = baseline_compliance

    if feasible and np.isfinite(compliance) and compliance > 0 and np.isfinite(baseline_compliance) and baseline_compliance > 0:
        ratio = baseline_compliance / compliance
        metrics["score_ratio"] = float(ratio)
        metrics["combined_score"] = float(ratio)
        metrics["valid"] = 1.0
    else:
        metrics["score_ratio"] = 0.0
        metrics["combined_score"] = 0.0
        metrics["valid"] = 0.0

    if "error" in eval_result:
        artifacts["error_message"] = str(eval_result["error"])

    artifacts["submission_path"] = str(submission_path)
    artifacts["benchmark_id"] = str(submission.get("benchmark_id", ""))
    artifacts["reported_compliance"] = submission.get("compliance")
    artifacts["reported_volume_fraction"] = submission.get("volume_fraction")

    metrics["runtime_s"] = float(time.time() - start_s)
    return _wrap(metrics, artifacts)


def _wrap(metrics: dict[str, float], artifacts: dict[str, Any]) -> Any:
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return {"metrics": metrics, "artifacts": artifacts}
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PyMOTO SIMP candidate program")
    parser.add_argument("program_path", help="Path to candidate program")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Subprocess timeout")
    parser.add_argument("--metrics-out", type=str, default="", help="Optional metrics JSON output path")
    parser.add_argument("--artifacts-out", type=str, default="", help="Optional artifacts JSON output path")
    parser.add_argument("--stdout-out", type=str, default="", help="Optional candidate stdout path")
    parser.add_argument("--stderr-out", type=str, default="", help="Optional candidate stderr path")
    parser.add_argument("--run-meta-out", type=str, default="", help="Optional run metadata output path")
    args = parser.parse_args()

    result = evaluate(args.program_path, timeout_s=float(args.timeout_s))
    if isinstance(result, dict):
        metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
        artifacts = result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {}
    else:
        metrics = getattr(result, "metrics", {}) if isinstance(getattr(result, "metrics", {}), dict) else {}
        artifacts = getattr(result, "artifacts", {}) if isinstance(getattr(result, "artifacts", {}), dict) else {}

    payload = {
        "combined_score": _safe_float(metrics.get("combined_score"), 0.0),
        "valid": _safe_float(metrics.get("valid"), 0.0),
        "runtime_s": _safe_float(metrics.get("runtime_s"), 0.0),
    }
    payload.update({k: v for k, v in metrics.items() if k not in payload})

    if args.stdout_out:
        stdout_text = str(artifacts.get("candidate_stdout_full", artifacts.get("candidate_stdout", "")))
        Path(args.stdout_out).write_text(stdout_text, encoding="utf-8", errors="replace")

    if args.stderr_out:
        stderr_text = str(artifacts.get("candidate_stderr_full", artifacts.get("candidate_stderr", "")))
        Path(args.stderr_out).write_text(stderr_text, encoding="utf-8", errors="replace")

    if args.metrics_out:
        _write_json(Path(args.metrics_out), payload)

    if args.artifacts_out:
        _write_json(Path(args.artifacts_out), artifacts)

    if args.run_meta_out:
        lines = [
            f"candidate={Path(args.program_path).expanduser().resolve()}",
            f"combined_score={payload.get('combined_score', 0.0)}",
            f"valid={payload.get('valid', 0.0)}",
            f"runtime_s={payload.get('runtime_s', 0.0)}",
        ]
        Path(args.run_meta_out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
