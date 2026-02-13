from __future__ import annotations

from pathlib import Path

from frontier_eval.tasks.base import Task


class MannedLunarLandingTask(Task):
    NAME = "manned_lunar_landing"

    def initial_program_path(self) -> Path:
        rel = Path("Astrodynamics") / "MannedLunarLanding" / "scripts" / "init.py"
        return (self.repo_root / rel).resolve()

    def openevolve_evaluator_path(self) -> Path:
        rel = Path("frontier_eval") / "evaluators" / "manned_lunar_landing_octave.py"
        return (self.repo_root / rel).resolve()

