"""
Evaluation bridge for ISCSO 2015.

Delegates to the verification evaluator in the benchmarks directory,
providing the repo_root for path resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def evaluate(program_path: str, *, repo_root: Path | None = None) -> Any:
    """Run the ISCSO 2015 evaluator."""
    # Add the verification directory to path so we can import the evaluator
    eval_dir = None
    if repo_root is not None:
        eval_dir = (
            repo_root / "benchmarks" / "StructuralOptimization" / "ISCSO2015" / "verification"
        )
    else:
        # Try to find it relative to this file
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            candidate = parent / "benchmarks" / "StructuralOptimization" / "ISCSO2015" / "verification"
            if candidate.is_dir():
                eval_dir = candidate
                break

    if eval_dir is not None and str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    from evaluator import evaluate as _evaluate  # type: ignore[import-untyped]

    return _evaluate(program_path, repo_root=repo_root)

