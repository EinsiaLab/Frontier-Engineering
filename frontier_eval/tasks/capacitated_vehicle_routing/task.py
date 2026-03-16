from __future__ import annotations

from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task


class CapacitatedVehicleRoutingTask(Task):
    NAME = "capacitated_vehicle_routing"

    def initial_program_path(self) -> Path:
        candidates = [
            self.repo_root
            / "benchmarks"
            / "Transportation"
            / "CapacitatedVehicleRouting"
            / "baseline"
            / "init.py",
            self.repo_root
            / "Transportation"
            / "CapacitatedVehicleRouting"
            / "baseline"
            / "init.py",
        ]
        for path in candidates:
            if path.is_file():
                return path.resolve()
        return candidates[0].resolve()

    def evaluate_program(self, program_path: Path) -> Any:
        from .evaluator.python import evaluate

        return evaluate(str(program_path), repo_root=self.repo_root)
