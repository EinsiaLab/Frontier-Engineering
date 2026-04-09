# EVOLVE-BLOCK-START
"""Baseline implementation for Task 05.

No stockpyl EOQD optimizer is used here.
"""

from __future__ import annotations

import math


def classic_eoq(fixed_cost: float, holding_cost: float, demand_rate: float) -> float:
    return math.sqrt(2.0 * fixed_cost * demand_rate / holding_cost)


def solve(cfg: dict):
    q_classic = classic_eoq(cfg["fixed_cost"], cfg["holding_cost"], cfg["demand_rate"])
    s = cfg["disruption_rate"] + cfg["recovery_rate"]
    rho = cfg["disruption_rate"] / s if s else 0.0
    return q_classic, q_classic * (1.0 + 1.55 * rho), 1.0 + rho
# EVOLVE-BLOCK-END
