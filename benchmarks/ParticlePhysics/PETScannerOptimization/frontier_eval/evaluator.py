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
    _ = repo_root
    program_path = Path(program_path).expanduser().resolve()
    task_dir = Path(__file__).resolve().parents[1]
    work_dir = Path(tempfile.mkdtemp(prefix="fe_pet_")).resolve()
    output_path = work_dir / "solution.json"

    try:
        proc = subprocess.run(
            [sys.executable, str(program_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        metrics = {
            "combined_score": -10000.0,
            "valid": 0.0,
            "timeout": 0.0,
            "runtime_s": float(time.time() - start),
            "program_returncode": float(proc.returncode),
        }
        artifacts = {
            "program_stdout": _tail(proc.stdout),
            "program_stderr": _tail(proc.stderr),
        }
        if not output_path.exists():
            artifacts["error_message"] = "solution.json not generated"
            return _wrap(metrics, artifacts)

        artifacts["solution.json"] = output_path.read_text(encoding="utf-8", errors="replace")
        proc2 = subprocess.run(
            [sys.executable, str(task_dir / "verification" / "evaluator.py"), str(output_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        artifacts["evaluator_stdout"] = _tail(proc2.stdout)
        artifacts["evaluator_stderr"] = _tail(proc2.stderr)

        try:
            result = json.loads(proc2.stdout.strip().splitlines()[-1])
            if result.get("status") == "success":
                metrics["combined_score"] = float(result.get("score", -10000.0))
                metrics["valid"] = 1.0
            else:
                artifacts["error_message"] = result.get("message", "Evaluation failed")
        except Exception as exc:
            artifacts["error_message"] = f"Failed to parse evaluator JSON output: {exc}"

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
