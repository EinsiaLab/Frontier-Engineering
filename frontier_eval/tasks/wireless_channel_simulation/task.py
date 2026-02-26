from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task


class HighReliableSimulationTask(Task):
    NAME = "high_reliable_simulation"

    def initial_program_path(self) -> Path:
        candidates = [
            self.repo_root
            / "benchmarks"
            / "WirelessChannelSimulation"
            / "HighReliableSimulation"
            / "scripts"
            / "init.py",
            self.repo_root
            / "WirelessChannelSimulation"
            / "HighReliableSimulation"
            / "scripts"
            / "init.py",
        ]
        for path in candidates:
            if path.is_file():
                return path.resolve()
        return candidates[0].resolve()

    def evaluate_program(self, program_path: Path) -> Any:
        eval_candidates = [
            self.repo_root
            / "benchmarks"
            / "WirelessChannelSimulation"
            / "HighReliableSimulation"
            / "verification"
            / "evaluator.py",
            self.repo_root
            / "WirelessChannelSimulation"
            / "HighReliableSimulation"
            / "verification"
            / "evaluator.py",
        ]
        eval_path = eval_candidates[0]
        for candidate in eval_candidates:
            if candidate.is_file():
                eval_path = candidate
                break

        spec = importlib.util.spec_from_file_location("hrs_evaluator", str(eval_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load evaluator: {eval_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.evaluate(str(program_path), repo_root=self.repo_root)
