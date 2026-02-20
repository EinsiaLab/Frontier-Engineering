from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task


def _tail(text: str, limit: int = 8000) -> str:
    return text if len(text) <= limit else text[-limit:]


class SmokeTask(Task):
    """
    Lightweight task to sanity-check algorithm adapters (OpenEvolve/ShinkaEvolve).

    - Runs the candidate Python program with a short timeout.
    - Reports `combined_score=1.0` when the program exits 0.
    """

    NAME = "smoke"

    def initial_program_path(self) -> Path:
        return (self.repo_root / "frontier_eval" / "tasks" / "smoke" / "init.py").resolve()

    def evaluate_program(self, program_path: Path) -> Any:
        start = time.time()
        timeout_s = int(os.environ.get("FRONTIER_EVAL_EVALUATOR_TIMEOUT_S", "10") or "10")

        program_path = program_path.expanduser().resolve()
        if not program_path.is_file():
            return _wrap(
                {
                    "combined_score": 0.0,
                    "valid": 0.0,
                    "runtime_s": float(time.time() - start),
                    "missing_program": 1.0,
                },
                {"error_message": f"program not found: {program_path}"},
            )

        with tempfile.TemporaryDirectory(prefix="fe_smoke_") as work_dir:
            proc = subprocess.run(
                [sys.executable, str(program_path)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=max(1, timeout_s),
                env=os.environ.copy(),
            )

        ok = proc.returncode == 0
        metrics: dict[str, float] = {
            "combined_score": 1.0 if ok else 0.0,
            "valid": 1.0 if ok else 0.0,
            "runtime_s": float(time.time() - start),
            "program_returncode": float(proc.returncode),
        }
        artifacts: dict[str, str] = {
            "program_stdout": _tail(proc.stdout),
            "program_stderr": _tail(proc.stderr),
        }
        if not ok and proc.stderr.strip():
            artifacts["error_message"] = "candidate program exited non-zero"

        return _wrap(metrics, artifacts)


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]) -> Any:
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)

