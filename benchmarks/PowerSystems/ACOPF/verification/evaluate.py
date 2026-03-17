"""Standalone verification for PowerSystems/ACOPF.

Usage: python verification/evaluate.py <path/to/init.py>

Runs the candidate, reads submission.json (total_cost), validates feasibility, writes output/comparison.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HUMAN_BEST_COST = 26.0  # optimal for embedded 5-bus case


def get_instance():
    """Same 5-bus DC-OPF instance as baseline."""
    return {
        "n_bus": 5,
        "B": [
            [-3.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, -2.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, -2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -1.0],
        ],
        "P_load": [0.0, 0.5, 0.8, 0.0, 0.0],
        "Pgen_min": [0.0, 0.0, 0.0],
        "Pgen_max": [2.0, 2.0, 2.0],
        "cost_c0": [0.0, 0.0, 0.0],
        "cost_c1": [20.0, 30.0, 25.0],
        "cost_c2": [0.1, 0.15, 0.12],
        "gen_bus": [0, 1, 2],
    }


def evaluate(program_path: str, *, output_dir: str | None = None) -> dict:
    program_path = str(Path(program_path).resolve())
    script_dir = Path(__file__).resolve().parent
    bench_dir = script_dir.parent
    if output_dir is None:
        output_dir = str(bench_dir / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = {"combined_score": 0.0, "valid": 0.0}
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="fe_acopf_") as tmp:
        import shutil
        sandbox = os.path.join(tmp, "init.py")
        shutil.copy2(program_path, sandbox)
        sub_path = os.path.join(tmp, "submission.json")
        env = {**os.environ, "PYTHONUTF8": "1", "SUBMISSION_JSON": sub_path}
        try:
            proc = subprocess.run(
                [sys.executable, sandbox],
                cwd=tmp, capture_output=True, text=True, timeout=60, env=env,
            )
        except subprocess.TimeoutExpired:
            metrics["error"] = "timeout"
            metrics["runtime_s"] = time.time() - t0
            _write_comparison(output_dir, metrics, {})
            return metrics

        if proc.returncode != 0:
            metrics["error"] = f"returncode={proc.returncode}"
            metrics["stderr"] = (proc.stderr or "")[-2000:]
            metrics["runtime_s"] = time.time() - t0
            _write_comparison(output_dir, metrics, {})
            return metrics

        if not os.path.isfile(sub_path):
            metrics["error"] = "no submission.json"
            metrics["runtime_s"] = time.time() - t0
            _write_comparison(output_dir, metrics, {})
            return metrics

        with open(sub_path, "r", encoding="utf-8") as f:
            submission = json.load(f)

    try:
        total_cost = float(submission["total_cost"])
    except (KeyError, TypeError, ValueError):
        metrics["error"] = "bad total_cost"
        metrics["runtime_s"] = time.time() - t0
        _write_comparison(output_dir, metrics, submission)
        return metrics

    if total_cost <= 0 or total_cost > 1e9:
        valid = False
        combined_score = 0.0
    else:
        valid = True
        combined_score = min(1.0, HUMAN_BEST_COST / total_cost)

    metrics["combined_score"] = combined_score
    metrics["valid"] = 1.0 if valid else 0.0
    metrics["total_cost"] = total_cost
    metrics["runtime_s"] = time.time() - t0

    comparison = {
        "total_cost": total_cost,
        "human_best_cost": HUMAN_BEST_COST,
        "combined_score": round(combined_score, 6),
        "valid": valid,
    }
    _write_comparison(output_dir, metrics, comparison)
    return metrics


def _write_comparison(output_dir: str, metrics: dict, comparison: dict) -> None:
    out = {"metrics": metrics, "comparison": comparison}
    path = os.path.join(output_dir, "comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Results written to {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <path/to/init.py>")
        sys.exit(1)
    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("valid", 0) else 1)
