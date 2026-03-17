# EVOLVE-BLOCK-START
"""5-bus DC-OPF: minimize generation cost s.t. power flow and limits.

Output: submission.json with 'total_cost' (float).
DO NOT MODIFY: get_instance(), cost_from_Pg(), is_feasible()
ALLOWED TO MODIFY: solve()
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PROBLEM CONSTANTS
# ---------------------------------------------------------------------------
HUMAN_BEST_COST = 26.0  # optimal for embedded 5-bus case


def get_instance():
    """Return 5-bus DC-OPF instance (same as scripts/init.py)."""
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


def cost_from_Pg(Pg: list[float], inst: dict) -> float:
    """Total cost = sum_g c0 + c1*Pg + c2*Pg^2."""
    c0, c1, c2 = inst["cost_c0"], inst["cost_c1"], inst["cost_c2"]
    return sum(c0[i] + c1[i] * Pg[i] + c2[i] * Pg[i] ** 2 for i in range(len(Pg)))


def is_feasible(Pg: list[float], inst: dict) -> bool:
    """Check power balance (DC: Pgen - Pload = B@theta, slack absorbs) and limits."""
    import numpy as np
    n_bus = inst["n_bus"]
    B = np.array(inst["B"])
    P_load = np.array(inst["P_load"])
    gen_bus = inst["gen_bus"]
    Pgen_min = inst["Pgen_min"]
    Pgen_max = inst["Pgen_max"]
    P_inj = np.zeros(n_bus)
    for i, b in enumerate(gen_bus):
        P_inj[b] += Pg[i]
    for b in range(n_bus):
        P_inj[b] -= P_load[b]
    # B@theta = P_inj; B singular (slack). Use pseudo-inverse or drop slack row.
    B_red = B[1:, 1:]
    P_red = P_inj[1:]
    if abs(np.linalg.det(B_red)) < 1e-10:
        return False
    theta_red = np.linalg.solve(B_red, P_red)
    theta = np.concatenate([[0.0], theta_red])
    P_inj_recon = B @ theta
    if np.max(np.abs(P_inj - P_inj_recon)) > 1e-4:
        return False
    for i, (pmin, pmax) in enumerate(zip(Pgen_min, Pgen_max)):
        if not (pmin - 1e-6 <= Pg[i] <= pmax + 1e-6):
            return False
    return True


def solve(instance: dict) -> dict:
    """Minimize generation cost s.t. DC power flow and limits. Return {'total_cost': float}."""
    import numpy as np
    from scipy.optimize import minimize

    n_bus = instance["n_bus"]
    B = np.array(instance["B"])
    P_load = np.array(instance["P_load"])
    gen_bus = instance["gen_bus"]
    Pgen_min = instance["Pgen_min"]
    Pgen_max = instance["Pgen_max"]
    c0, c1, c2 = instance["cost_c0"], instance["cost_c1"], instance["cost_c2"]
    n_gen = len(gen_bus)

    def obj(Pg):
        return sum(c0[i] + c1[i] * Pg[i] + c2[i] * Pg[i] ** 2 for i in range(n_gen))

    def power_balance(Pg):
        P_inj = np.zeros(n_bus)
        for i, b in enumerate(gen_bus):
            P_inj[b] += Pg[i]
        P_inj -= P_load
        B_red = B[1:, 1:]
        P_red = P_inj[1:]
        theta_red = np.linalg.solve(B_red, P_red)
        theta = np.concatenate([[0.0], theta_red])
        return B @ theta - P_inj  # should be ~0

    bounds = [(Pgen_min[i], Pgen_max[i]) for i in range(n_gen)]
    x0 = [0.5, 0.4, 0.4]  # sum=1.3 = sum(P_load)
    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda Pg: np.sum(Pg) - np.sum(P_load)},
    )
    Pg = list(res.x)
    total_cost = float(res.fun)
    if not is_feasible(Pg, instance):
        total_cost = 1e10
    return {"total_cost": total_cost, "Pg": Pg}


# ---------------------------------------------------------------------------
# Main: run solve, write submission.json
# ---------------------------------------------------------------------------
def main():
    inst = get_instance()
    out = solve(inst)
    submission = {"total_cost": out["total_cost"]}
    submission_path = os.environ.get("SUBMISSION_JSON", "submission.json")
    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)
    print(f"total_cost = {out['total_cost']}")


if __name__ == "__main__":
    main()
    sys.exit(0)

# EVOLVE-BLOCK-END
