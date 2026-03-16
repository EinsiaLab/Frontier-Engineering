#!/usr/bin/env python3
"""Evaluate a DC-OPF solution on the IEEE 14-bus benchmark.

Invoke from the benchmark root directory:
    python verification/evaluate.py

Loads baseline/init.solve(instance) -> list[float] and:
  1. Checks power balance (sum(Pg) == total_load within 0.5 MW)
  2. Checks generation limits (pmin <= Pg <= pmax)
  3. Runs DC power flow, checks thermal line limits
  4. Computes total generation cost (quadratic)
  5. Score = min(1.0, HUMAN_BEST_COST / solution_cost)

Human best (DC-OPF optimal): 7892.76 $/h
Baseline (equal dispatch):    9154.77 $/h  -> score ~0.862
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

# ---------------------------------------------------------------------------
# IEEE 14-bus system data  (MATPOWER case14 / PGLib-OPF case14_ieee)
# ---------------------------------------------------------------------------
BASE_MVA: float = 100.0

# 14 buses; bus 1 (index 0) is the slack/reference
BUSES: list[dict[str, Any]] = [
    {"id": 1,  "pd":  0.0},  # slack
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

# 5 generators (MATPOWER case14 cost polynomial model)
GENERATORS: list[dict[str, Any]] = [
    {"bus": 1, "pmin":  0.0, "pmax": 332.4, "cost_a": 0.0430292519, "cost_b": 20.0},
    {"bus": 2, "pmin":  0.0, "pmax": 140.0, "cost_a": 0.25,         "cost_b": 20.0},
    {"bus": 3, "pmin":  0.0, "pmax": 100.0, "cost_a": 0.01,         "cost_b": 40.0},
    {"bus": 6, "pmin":  0.0, "pmax": 100.0, "cost_a": 0.01,         "cost_b": 40.0},
    {"bus": 8, "pmin":  0.0, "pmax": 100.0, "cost_a": 0.01,         "cost_b": 40.0},
]

# 20 branches (from/to in 1-indexed, x in pu at 100 MVA, rate_a in MW)
# Reactances from MATPOWER case14; thermal limits from PGLib-OPF case14_ieee
BRANCHES: list[dict[str, Any]] = [
    {"from_bus":  1, "to_bus":  2, "x": 0.05917, "rate_a": 120.0},
    {"from_bus":  1, "to_bus":  5, "x": 0.22304, "rate_a":  65.0},
    {"from_bus":  2, "to_bus":  3, "x": 0.19797, "rate_a":  36.0},  # binding!
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

HUMAN_BEST_COST: float = 7892.76  # DC-OPF optimal (scipy SLSQP)
TOTAL_LOAD: float = sum(b["pd"] for b in BUSES)  # 259.0 MW


# ---------------------------------------------------------------------------
# DC power flow helpers
# ---------------------------------------------------------------------------

def _build_b_matrix(buses: list, branches: list, base_mva: float) -> np.ndarray:
    """Build the nodal susceptance matrix B (N x N) for DC power flow."""
    n = len(buses)
    B = np.zeros((n, n))
    for br in branches:
        fr = br["from_bus"] - 1  # 0-indexed
        to = br["to_bus"] - 1
        b_line = 1.0 / br["x"]
        B[fr, fr] += b_line
        B[to, to] += b_line
        B[fr, to] -= b_line
        B[to, fr] -= b_line
    return B


def _dc_power_flow(
    Pg: list[float],
    generators: list,
    buses: list,
    branches: list,
    base_mva: float,
    B: np.ndarray,
) -> np.ndarray:
    """Compute DC voltage angles (radians) given generator dispatch.

    Slack bus is bus index 0 (bus id 1); its angle is fixed at 0.
    Returns angles in radians for all N buses.
    """
    n = len(buses)
    # Net injection at each bus (in pu)
    P_net = np.zeros(n)
    for b in buses:
        idx = b["id"] - 1
        P_net[idx] -= b["pd"] / base_mva  # load injection negative
    for k, g in enumerate(generators):
        idx = g["bus"] - 1
        P_net[idx] += Pg[k] / base_mva   # generation injection positive

    # Remove slack bus (index 0) row and column
    B_red = B[1:, 1:]
    P_red = P_net[1:]
    theta_red = np.linalg.solve(B_red, P_red)
    return np.concatenate([[0.0], theta_red])


def _line_flows(theta: np.ndarray, branches: list, base_mva: float) -> list[float]:
    """Compute active power flows on each branch (MW), positive = from->to."""
    flows = []
    for br in branches:
        fr = br["from_bus"] - 1
        to = br["to_bus"] - 1
        flow = (theta[fr] - theta[to]) / br["x"] * base_mva
        flows.append(flow)
    return flows


# ---------------------------------------------------------------------------
# Validation and scoring
# ---------------------------------------------------------------------------

def validate_and_score(
    Pg: list[float],
    instance: dict[str, Any],
) -> dict[str, Any]:
    generators = instance["generators"]
    buses = instance["buses"]
    branches = instance["branches"]
    base_mva = instance["base_mva"]

    errors: list[str] = []

    # ---- 1. Type / length check ----
    if not isinstance(Pg, (list, tuple)) or len(Pg) != len(generators):
        return {
            "valid": False,
            "errors": [f"Expected list of {len(generators)} floats, got {type(Pg)}"],
            "combined_score": 0.0,
        }

    Pg = [float(p) for p in Pg]
    total_load = sum(b["pd"] for b in buses)

    # ---- 2. Power balance ----
    balance_err = abs(sum(Pg) - total_load)
    if balance_err > 0.5:
        errors.append(
            f"Power imbalance: sum(Pg)={sum(Pg):.3f} MW, load={total_load:.1f} MW "
            f"(error={balance_err:.3f} > 0.5 MW)"
        )

    # ---- 3. Generation limits ----
    for k, g in enumerate(generators):
        if Pg[k] < g["pmin"] - 0.001:
            errors.append(
                f"Gen {k+1} (bus {g['bus']}): Pg={Pg[k]:.3f} < Pmin={g['pmin']}"
            )
        if Pg[k] > g["pmax"] + 0.001:
            errors.append(
                f"Gen {k+1} (bus {g['bus']}): Pg={Pg[k]:.3f} > Pmax={g['pmax']}"
            )

    # ---- 4. DC power flow and line limits ----
    B = _build_b_matrix(buses, branches, base_mva)
    try:
        theta = _dc_power_flow(Pg, generators, buses, branches, base_mva, B)
        flows = _line_flows(theta, branches, base_mva)
        line_violations: list[str] = []
        for i, (flow, br) in enumerate(zip(flows, branches)):
            if abs(flow) > br["rate_a"] + 0.01:
                line_violations.append(
                    f"Line {br['from_bus']}-{br['to_bus']}: "
                    f"|{flow:.2f}| MW > {br['rate_a']:.1f} MW"
                )
        if line_violations:
            errors.extend(line_violations)
    except np.linalg.LinAlgError as exc:
        errors.append(f"DC power flow failed: {exc}")
        flows = None

    # ---- 5. Cost ----
    cost = sum(
        g["cost_a"] * Pg[k] ** 2 + g["cost_b"] * Pg[k]
        for k, g in enumerate(generators)
    )

    # ---- 6. Score ----
    valid = len(errors) == 0
    if valid:
        combined_score = min(1.0, HUMAN_BEST_COST / max(HUMAN_BEST_COST, cost))
    else:
        combined_score = 0.0

    result: dict[str, Any] = {
        "valid": valid,
        "combined_score": combined_score,
        "total_cost_$/h": round(cost, 4),
        "human_best_cost_$/h": HUMAN_BEST_COST,
        "Pg_MW": [round(p, 4) for p in Pg],
        "total_load_MW": total_load,
    }
    if errors:
        result["errors"] = errors
    if flows is not None:
        result["line_flows_MW"] = [round(f, 3) for f in flows]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    from baseline.init import solve

    try:
        Pg = solve(INSTANCE)
    except Exception as exc:
        print(f"ERROR: solve() raised {type(exc).__name__}: {exc}", file=sys.stderr)
        result = {
            "valid": False,
            "combined_score": 0.0,
            "baseline_final_score": 0.0,
            "error": str(exc),
        }
        _write_result(result)
        return 1

    result = validate_and_score(Pg, INSTANCE)
    result["baseline_final_score"] = result["combined_score"]

    _write_result(result)

    print(f"Valid:          {result['valid']}")
    print(f"Total cost:     {result.get('total_cost_$/h', 'N/A')} $/h")
    print(f"Human best:     {HUMAN_BEST_COST} $/h")
    print(f"Combined score: {result['combined_score']:.6f}")
    if not result["valid"]:
        for err in result.get("errors", []):
            print(f"  ERROR: {err}")

    return 0 if result["valid"] else 1


def _write_result(result: dict[str, Any]) -> None:
    output_dir = TASK_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
