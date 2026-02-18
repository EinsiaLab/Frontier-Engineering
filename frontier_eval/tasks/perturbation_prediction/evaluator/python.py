from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DATASET_ID = "neurips-2023-data"


def _is_repo_root(path: Path) -> bool:
    if not (path / "frontier_eval").is_dir():
        return False
    if (path / "benchmarks").is_dir():
        return True
    return (path / "Astrodynamics").is_dir() and (path / "ElectronicDesignAutomation").is_dir()


def _find_repo_root() -> Path:
    if "FRONTIER_ENGINEERING_ROOT" in os.environ:
        return Path(os.environ["FRONTIER_ENGINEERING_ROOT"]).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if _is_repo_root(parent):
            return parent
    return Path.cwd().resolve()


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _truncate_middle(text: str, limit: int = 200_000) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, (limit - 128) // 2)
    omitted = len(text) - (2 * keep)
    return text[:keep] + f"\n\n[... truncated {omitted} chars ...]\n\n" + text[-keep:]


def _load_json_from_stdout(stdout: str) -> Any | None:
    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    # Try last non-empty line first (baseline prints a single JSON line)
    candidates = [lines[-1], stdout]
    for s in candidates:
        try:
            return json.loads(s)
        except Exception:
            continue
    return None


def _remaining_timeout(deadline_s: float) -> float:
    return max(1.0, float(deadline_s - time.time()))


def evaluate(program_path: str, *, repo_root: Path | None = None):
    """
    OpenEvolve evaluator for benchmarks/SingleCellAnalysis/perturbation_prediction.

    Contract for candidate program:
    - `python <program.py>` must write a `prediction.h5ad` file in the current working directory
      (or print JSON to stdout containing {"output": "..."} pointing to the generated file).
    """
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    program_path = str(Path(program_path).expanduser().resolve())

    benchmark_dir = (
        repo_root / "benchmarks" / "SingleCellAnalysis" / "perturbation_prediction"
    ).resolve()
    eval_script = (benchmark_dir / "verification" / "evaluate_perturbation_prediction.py").resolve()
    dataset_dir = (benchmark_dir / "resources_cache" / DATASET_ID).resolve()

    artifacts: dict[str, str] = {}

    if not eval_script.is_file():
        metrics = {
            "combined_score": 0.0,
            "valid": 0.0,
            "timeout": 0.0,
            "runtime_s": float(time.time() - start),
        }
        artifacts["error_message"] = f"evaluation script not found: {eval_script}"
        return _wrap(metrics, artifacts)

    work_dir = Path(tempfile.mkdtemp(prefix="fe_pp_")).resolve()

    # Align subprocess timeouts with OpenEvolve's evaluator timeout to avoid
    # background processes continuing after OpenEvolve times out the evaluation.
    evaluator_timeout_s = float(os.environ.get("FRONTIER_EVAL_EVALUATOR_TIMEOUT_S", "300") or "300")
    deadline_s = start + max(1.0, evaluator_timeout_s - 5.0)

    try:
        metrics: dict[str, float] = {
            "combined_score": 0.0,
            "valid": 0.0,
            "timeout": 0.0,
            "runtime_s": 0.0,
        }

        # 1) Run candidate program to generate prediction.h5ad
        program_start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, program_path],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
            )
        except subprocess.TimeoutExpired as e:
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"program timeout: {e}"
            return _wrap(metrics, artifacts)

        metrics["program_runtime_s"] = float(time.time() - program_start)
        metrics["program_returncode"] = float(proc.returncode)
        artifacts["program_stdout"] = _tail(proc.stdout)
        artifacts["program_stderr"] = _tail(proc.stderr)
        artifacts["program_stdout_full"] = _truncate_middle(proc.stdout)
        artifacts["program_stderr_full"] = _truncate_middle(proc.stderr)

        prediction_path = work_dir / "prediction.h5ad"
        if not prediction_path.exists():
            parsed = _load_json_from_stdout(proc.stdout)
            if isinstance(parsed, dict) and parsed.get("output"):
                out = Path(str(parsed["output"]))
                if not out.is_absolute():
                    out = (work_dir / out).resolve()
                if out.suffix == ".h5ad" and out.exists():
                    prediction_path = out

        if not prediction_path.exists():
            candidates = sorted(work_dir.glob("*.h5ad"))
            if len(candidates) == 1:
                prediction_path = candidates[0]

        if not prediction_path.exists():
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = "prediction.h5ad not generated"
            return _wrap(metrics, artifacts)

        artifacts["prediction_path"] = str(prediction_path)

        # 2) Score prediction using the benchmark verification script
        score_start = time.time()
        try:
            proc2 = subprocess.run(
                [
                    sys.executable,
                    str(eval_script),
                    "--prediction",
                    str(prediction_path),
                    "--dataset-dir",
                    str(dataset_dir),
                ],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
            )
        except subprocess.TimeoutExpired as e:
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"scorer timeout: {e}"
            return _wrap(metrics, artifacts)

        metrics["score_runtime_s"] = float(time.time() - score_start)
        metrics["score_returncode"] = float(proc2.returncode)
        artifacts["score_stdout"] = _tail(proc2.stdout)
        artifacts["score_stderr"] = _tail(proc2.stderr)
        artifacts["score_stdout_full"] = _truncate_middle(proc2.stdout)
        artifacts["score_stderr_full"] = _truncate_middle(proc2.stderr)

        if proc2.returncode != 0:
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = "scorer failed (non-zero return code)"
            return _wrap(metrics, artifacts)

        try:
            score_metrics_raw = json.loads(proc2.stdout)
        except Exception as e:
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"failed to parse scorer output as JSON: {e}"
            return _wrap(metrics, artifacts)

        # Only keep numeric metrics for OpenEvolve.
        if isinstance(score_metrics_raw, dict):
            for key, value in score_metrics_raw.items():
                if isinstance(value, bool):
                    metrics[key] = float(value)
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)
                else:
                    artifacts[f"score_{key}"] = str(value)

        metrics["runtime_s"] = float(time.time() - start)
        if metrics.get("valid", 0.0) > 0:
            metrics["combined_score"] = float(metrics.get("combined_score", 0.0))
        else:
            metrics["combined_score"] = 0.0

        return _wrap(metrics, artifacts)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)

