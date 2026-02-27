from __future__ import annotations

from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task

from .spec import load_unified_task_spec


class UnifiedTask(Task):
    """
    Generic benchmark task driven by benchmark-local metadata files.

    The benchmark folder provides the paths/prompts/commands; Frontier Eval keeps
    a single reusable task implementation.
    """

    NAME = "unified"

    def initial_program_path(self) -> Path:
        spec = load_unified_task_spec(task_cfg=getattr(self.cfg, "task", None), repo_root=self.repo_root)
        return spec.initial_program_path

    def evaluate_program(self, program_path: Path) -> Any:
        from .evaluator.python import evaluate

        spec = load_unified_task_spec(task_cfg=getattr(self.cfg, "task", None), repo_root=self.repo_root)
        return evaluate(str(program_path), spec=spec)
