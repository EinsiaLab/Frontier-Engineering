from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _is_repo_root(path: Path) -> bool:
    return (path / "frontier_eval").is_dir() and (path / "benchmarks").is_dir()


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


def evaluate(program_path: str, *, repo_root: Path | None = None):
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    task_dir = Path(__file__).resolve().parents[1]
    work_dir = Path(tempfile.mkdtemp(prefix="fe_perturb_")).resolve()
    program_path = Path(program_path).expanduser().resolve()
    dataset_dir = (
        repo_root
        / "benchmarks"
        / "SingleCellAnalysis"
        / "perturbation_prediction"
        / "resources_cache"
        / "neurips-2023-data"
    ).resolve()
    output_path = work_dir / "prediction.h5ad"
    env = os.environ.copy()
    env.setdefault("FRONTIER_ENGINEERING_ROOT", str(repo_root))
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(program_path),
                "--output",
                str(output_path),
                "--dataset-dir",
                str(dataset_dir),
            ],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        metrics = {
            "combined_score": -10000.0,
            "valid": 0.0,
            "timeout": 0.0,
            "runtime_s": 0.0,
            "program_returncode": float(proc.returncode),
        }
        artifacts = {
            "program_stdout": _tail(proc.stdout),
            "program_stderr": _tail(proc.stderr),
        }
        if proc.returncode != 0:
            artifacts["error_message"] = "candidate program exited non-zero"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        if not output_path.exists():
            artifacts["error_message"] = "prediction.h5ad not generated"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        try:
            artifacts["prediction_bytes"] = str(output_path.stat().st_size)
        except Exception:
            pass

        proc2 = subprocess.run(
            [
                sys.executable,
                str(task_dir / "verification" / "evaluate_perturbation_prediction.py"),
                "--prediction",
                str(output_path),
                "--dataset-dir",
                str(dataset_dir),
            ],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        artifacts["scoring_stdout"] = _tail(proc2.stdout)
        artifacts["scoring_stderr"] = _tail(proc2.stderr)
        if proc2.returncode != 0:
            artifacts["error_message"] = "scorer exited non-zero"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        try:
            score_metrics = json.loads(proc2.stdout)
        except Exception as exc:
            artifacts["error_message"] = f"failed to parse scorer JSON: {exc}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        if isinstance(score_metrics, dict):
            metrics.update(score_metrics)
            metrics["valid"] = float(score_metrics.get("valid", 1.0) or 0.0)
            metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return {"metrics": metrics, "artifacts": artifacts}
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
