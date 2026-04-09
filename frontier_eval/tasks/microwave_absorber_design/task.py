from __future__ import annotations

from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task


class MicrowaveAbsorberDesignTask(Task):
    NAME = "microwave_absorber_design"

    def initial_program_path(self) -> Path:
        candidates = [
            self.repo_root
            / "benchmarks"
            / "MaterialEngineering"
            / "MicrowaveAbsorberDesign"
            / "scripts"
            / "init.py",
        ]
        for path in candidates:
            if path.is_file():
                return path.resolve()
        return candidates[0].resolve()

    def evaluate_program(self, program_path: Path) -> Any:
        import json
        import subprocess
        import sys

        task_dir = (
            self.repo_root
            / "benchmarks"
            / "MaterialEngineering"
            / "MicrowaveAbsorberDesign"
        )
        evaluator_path = task_dir / "verification" / "evaluator.py"

        result = subprocess.run(
            [sys.executable, str(evaluator_path), str(program_path)],
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Parse the JSON result from evaluator output
        stdout = result.stdout
        try:
            # Find the JSON block between EVALUATION RESULT markers
            lines = stdout.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip() == "{":
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if in_json and line.strip() == "}":
                    break
            if json_lines:
                return json.loads("\n".join(json_lines))
        except (json.JSONDecodeError, ValueError):
            pass

        return {"valid": 0, "feasible": 0, "combined_score": 0.0,
                "message": f"Failed to parse evaluator output. returncode={result.returncode}"}
