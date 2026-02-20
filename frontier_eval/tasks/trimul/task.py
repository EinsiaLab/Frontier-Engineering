from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task


class TriMulTask(Task):
    NAME = "trimul"

    def _kernel_python(self) -> str:
        task_cfg = getattr(self.cfg, "task", None)
        kernel_python = ""
        if task_cfg is not None:
            try:
                kernel_python = str(getattr(task_cfg, "kernel_python", "") or "")
            except Exception:
                kernel_python = ""
        if not kernel_python:
            kernel_python = str(os.environ.get("FRONTIER_EVAL_TRIMUL_PYTHON", "") or "")
        return kernel_python or "python"

    def initial_program_path(self) -> Path:
        candidates = [
            self.repo_root
            / "benchmarks"
            / "KernelEngineering"
            / "TriMul"
            / "baseline"
            / "submission.py",
            self.repo_root / "KernelEngineering" / "TriMul" / "baseline" / "submission.py",
        ]
        for path in candidates:
            if path.is_file():
                return path.resolve()
        return candidates[0].resolve()

    def evaluate_program(self, program_path: Path) -> Any:
        from .evaluator.python import evaluate

        return evaluate(
            str(program_path),
            repo_root=self.repo_root,
            kernel_python=self._kernel_python(),
        )
