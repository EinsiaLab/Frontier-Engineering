from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# IEEE 14-bus instance (must stay in sync with verification/evaluate.py)
# ---------------------------------------------------------------------------
BASE_MVA: float = 100.0

BUSES: list[dict[str, Any]] = [
    {"id": 1,  "pd":  0.0},
    {"id": 2,  "pd": 21.7},
    {"id": 3,  "pd": 94.2},
    {"id": 4,  "pd": 47.8},
    {"id": 5,  "pd":  7.6},
    {"id": 6,  "pd": 11.2},
    {"id": 7,  "pd":  0.0},
    {"id": 8,  "pd":  0.0},
    {"id": 9,  "pd": 29.5},
    {"id": 10, "pd":  9.0},
    {"id": 11, "pd":  3.5},
    {"id": 12, "pd":  6.1},
    {"id": 13, "pd": 13.5},
    {"id": 14, "pd": 14.9},
]

GENERATORS: list[dict[str, Any]] = [
    {"bus": 1, "pmin":  0.0, "pmax": 332.4, "cost_a": 0.0430292519, "cost_b": 20.0},
    {"bus": 2, "pmin":  0.0, "pmax": 140.0, "cost_a": 0.25,         "cost_b": 20.0},
    {"bus": 3, "pmin":  0.0, "pmax": 100.0, "cost_a": 0.01,         "cost_b": 40.0},
    {"bus": 6, "pmin":  0.0, "pmax": 100.0, "cost_a": 0.01,         "cost_b": 40.0},
    {"bus": 8, "pmin":  0.0, "pmax": 100.0, "cost_a": 0.01,         "cost_b": 40.0},
]

BRANCHES: list[dict[str, Any]] = [
    {"from_bus":  1, "to_bus":  2, "x": 0.05917, "rate_a": 120.0},
    {"from_bus":  1, "to_bus":  5, "x": 0.22304, "rate_a":  65.0},
    {"from_bus":  2, "to_bus":  3, "x": 0.19797, "rate_a":  36.0},
    {"from_bus":  2, "to_bus":  4, "x": 0.17632, "rate_a":  65.0},
    {"from_bus":  2, "to_bus":  5, "x": 0.17388, "rate_a":  53.0},
    {"from_bus":  3, "to_bus":  4, "x": 0.17103, "rate_a":  40.0},
    {"from_bus":  4, "to_bus":  5, "x": 0.04211, "rate_a":  87.0},
    {"from_bus":  4, "to_bus":  7, "x": 0.20912, "rate_a": 130.0},
    {"from_bus":  4, "to_bus":  9, "x": 0.55618, "rate_a":  33.0},
    {"from_bus":  5, "to_bus":  6, "x": 0.25202, "rate_a": 133.0},
    {"from_bus":  6, "to_bus": 11, "x": 0.19890, "rate_a":  32.0},
    {"from_bus":  6, "to_bus": 12, "x": 0.25581, "rate_a":  16.0},
    {"from_bus":  6, "to_bus": 13, "x": 0.13027, "rate_a":  65.0},
    {"from_bus":  7, "to_bus":  8, "x": 0.17615, "rate_a": 130.0},
    {"from_bus":  7, "to_bus":  9, "x": 0.11001, "rate_a": 130.0},
    {"from_bus":  9, "to_bus": 10, "x": 0.08450, "rate_a":  13.0},
    {"from_bus":  9, "to_bus": 14, "x": 0.27038, "rate_a":  13.0},
    {"from_bus": 10, "to_bus": 11, "x": 0.19207, "rate_a":  12.0},
    {"from_bus": 12, "to_bus": 13, "x": 0.19988, "rate_a":  16.0},
    {"from_bus": 13, "to_bus": 14, "x": 0.34802, "rate_a":  16.0},
]

INSTANCE: dict[str, Any] = {
    "base_mva": BASE_MVA,
    "generators": GENERATORS,
    "buses": BUSES,
    "branches": BRANCHES,
}

HUMAN_BEST_COST: float = 7892.76


# ---------------------------------------------------------------------------
# DC power flow helpers
# ---------------------------------------------------------------------------

def _build_b_matrix(n: int, branches: list) -> np.ndarray:
    B = np.zeros((n, n))
    for br in branches:
        fr = br["from_bus"] - 1
        to = br["to_bus"] - 1
        b_line = 1.0 / br["x"]
        B[fr, fr] += b_line
        B[to, to] += b_line
        B[fr, to] -= b_line
        B[to, fr] -= b_line
    return B


def _validate_and_score(Pg: list[float]) -> dict[str, Any]:
    generators = INSTANCE["generators"]
    buses = INSTANCE["buses"]
    branches = INSTANCE["branches"]
    base_mva = INSTANCE["base_mva"]

    errors: list[str] = []
    n = len(buses)

    if not isinstance(Pg, (list, tuple)) or len(Pg) != len(generators):
        return {
            "valid": 0.0,
            "combined_score": 0.0,
            "errors": [f"Expected list of {len(generators)} floats"],
        }

    Pg = [float(p) for p in Pg]
    total_load = sum(b["pd"] for b in buses)

    balance_err = abs(sum(Pg) - total_load)
    if balance_err > 0.5:
        errors.append(f"Power imbalance: {balance_err:.3f} MW")

    for k, g in enumerate(generators):
        if Pg[k] < g["pmin"] - 0.001:
            errors.append(f"Gen {k+1}: Pg={Pg[k]:.3f} < Pmin={g['pmin']}")
        if Pg[k] > g["pmax"] + 0.001:
            errors.append(f"Gen {k+1}: Pg={Pg[k]:.3f} > Pmax={g['pmax']}")

    B = _build_b_matrix(n, branches)
    P_net = np.zeros(n)
    for b in buses:
        P_net[b["id"] - 1] -= b["pd"] / base_mva
    for k, g in enumerate(generators):
        P_net[g["bus"] - 1] += Pg[k] / base_mva

    try:
        theta_red = np.linalg.solve(B[1:, 1:], P_net[1:])
        theta = np.concatenate([[0.0], theta_red])
        for br in branches:
            fr = br["from_bus"] - 1
            to = br["to_bus"] - 1
            flow = (theta[fr] - theta[to]) / br["x"] * base_mva
            if abs(flow) > br["rate_a"] + 0.01:
                errors.append(
                    f"Line {br['from_bus']}-{br['to_bus']}: "
                    f"|{flow:.1f}| > {br['rate_a']:.0f} MW"
                )
    except np.linalg.LinAlgError:
        errors.append("DC power flow failed (singular B matrix)")

    cost = sum(
        g["cost_a"] * Pg[k] ** 2 + g["cost_b"] * Pg[k]
        for k, g in enumerate(generators)
    )

    valid = 1.0 if not errors else 0.0
    combined_score = (
        min(1.0, HUMAN_BEST_COST / max(HUMAN_BEST_COST, cost)) if valid else 0.0
    )

    return {
        "valid": valid,
        "combined_score": combined_score,
        "total_cost": round(cost, 4),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate(program_path: str, repo_root: Path | None = None) -> Any:
    """Load the candidate `solve` function and evaluate it."""
    from frontier_eval.tasks.base import EvaluationResult

    program_path_obj = Path(program_path)

    # Copy candidate into a temp directory so imports are isolated
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        candidate_dest = tmp / "solution.py"
        shutil.copy2(program_path_obj, candidate_dest)

        spec = importlib.util.spec_from_file_location("_candidate", candidate_dest)
        if spec is None or spec.loader is None:
            return EvaluationResult(
                combined_score=0.0,
                valid=0.0,
                metadata={"error": "cannot load candidate file"},
            )
        module = importlib.util.module_from_spec(spec)
        t_start = time.perf_counter()
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            return EvaluationResult(
                combined_score=0.0,
                valid=0.0,
                metadata={"error": f"import error: {exc}"},
            )

        if not hasattr(module, "solve"):
            return EvaluationResult(
                combined_score=0.0,
                valid=0.0,
                metadata={"error": "solve() not found in candidate"},
            )

        try:
            Pg = module.solve(INSTANCE)
        except Exception as exc:
            return EvaluationResult(
                combined_score=0.0,
                valid=0.0,
                metadata={"error": f"solve() raised {type(exc).__name__}: {exc}"},
            )
        elapsed = time.perf_counter() - t_start

    result = _validate_and_score(Pg)

    return EvaluationResult(
        combined_score=result["combined_score"],
        valid=result["valid"],
        metadata={
            "total_cost": result.get("total_cost"),
            "human_best_cost": HUMAN_BEST_COST,
            "elapsed_s": round(elapsed, 4),
            "errors": result.get("errors", []),
        },
    )
