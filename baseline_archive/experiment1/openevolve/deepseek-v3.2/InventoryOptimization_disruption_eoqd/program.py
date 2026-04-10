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
    disruption_ratio = cfg["disruption_rate"] / cfg["recovery_rate"]
    cost_ratio = cfg["stockout_cost"] / cfg["holding_cost"]
    base_mult = math.sqrt(disruption_ratio * cost_ratio)
    demand_factor = math.log1p(cfg["demand_rate"]) / math.log1p(100.0)
    cost_factor = min(1.0, cost_ratio / 500.0)
    adjusted_coeff = 0.18 + 0.04 * demand_factor + 0.02 * cost_factor
    adjusted_coeff = min(max(adjusted_coeff, 0.18), 0.24)
    safety_multiplier = 1.0 + adjusted_coeff * base_mult
    q_manual = q_classic * safety_multiplier
    return q_classic, q_manual, safety_multiplier
# EVOLVE-BLOCK-END
