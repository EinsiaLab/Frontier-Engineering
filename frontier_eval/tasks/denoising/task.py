from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task


class DenoisingTask(Task):
    NAME = "denoising"

    def _denoising_python(self) -> str:
        task_cfg = getattr(self.cfg, "task", None)
        denoising_python = ""
        if task_cfg is not None:
            try:
                denoising_python = str(getattr(task_cfg, "denoising_python", "") or "")
            except Exception:
                denoising_python = ""
        if not denoising_python:
            denoising_python = str(os.environ.get("FRONTIER_EVAL_DENOISING_PYTHON", "") or "")
        return denoising_python or "python"

    def initial_program_path(self) -> Path:
        candidates = [
            self.repo_root
            / "benchmarks"
            / "SingleCellAnalysis"
            / "denoising"
            / "task_denoising"
            / "src"
            / "methods"
            / "submission"
            / "script.py",
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
            denoising_python=self._denoising_python(),
        )
