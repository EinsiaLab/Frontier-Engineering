from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _parse_result(stdout: str) -> dict:
    marker_pos = stdout.find("EVALUATION RESULT")
    search_start = marker_pos if marker_pos >= 0 else 0
    json_start = stdout.find("{", search_start)
    json_end = stdout.rfind("}")
    if json_start < 0 or json_end < json_start:
        raise ValueError("Failed to locate JSON result block in evaluator stdout")
    return json.loads(stdout[json_start : json_end + 1])


def evaluate(program_path: str, *, repo_root: Path | None = None):
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    _ = repo_root
    program_path = Path(program_path).expanduser().resolve()
    task_dir = Path(__file__).resolve().parents[1]

    eval_script = (task_dir / "verification" / "evaluator.py").resolve()
    proc = subprocess.run(
        [sys.executable, str(eval_script), str(program_path)],
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )

    metrics = {
        "combined_score": 0.0,
        "valid": 0.0,
        "timeout": 0.0,
        "runtime_s": float(time.time() - start),
        "program_returncode": float(proc.returncode),
    }
    artifacts = {
        "evaluator_stdout": _tail(proc.stdout),
        "evaluator_stderr": _tail(proc.stderr),
    }
    for candidate in [task_dir / "temp" / "submission.json", task_dir / "submission.json"]:
        if candidate.exists():
            artifacts[candidate.relative_to(task_dir).as_posix()] = candidate.read_text(
                encoding="utf-8", errors="replace"
            )

    try:
        result = _parse_result(proc.stdout)
        metrics["combined_score"] = float(result.get("combined_score", 0.0))
        metrics["valid"] = 1.0 if float(result.get("valid", 0.0)) > 0 else 0.0
    except Exception as exc:
        artifacts["error_message"] = f"Failed to parse evaluator result: {exc}"

    return _wrap(metrics, artifacts)


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return {"metrics": metrics, "artifacts": artifacts}
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
