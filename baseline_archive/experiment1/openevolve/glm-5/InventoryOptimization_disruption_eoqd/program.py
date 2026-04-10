# EVOLVE-BLOCK-START
"""Baseline implementation for Task 05.

No stockpyl EOQD optimizer is used here.
"""

from __future__ import annotations

import math


def classic_eoq(fixed_cost: float, holding_cost: float, demand_rate: float) -> float:
    return math.sqrt(2.0 * fixed_cost * demand_rate / holding_cost)


def solve(cfg: dict):
    K = cfg["fixed_cost"]
    h = cfg["holding_cost"]
    D = cfg["demand_rate"]
    lam = cfg["disruption_rate"]
    mu = cfg["recovery_rate"]
    p = cfg["stockout_cost"]
    
    q_classic = math.sqrt(2.0 * K * D / h)
    
    # EOQD: adjust for disruption risk and stockout cost
    disruption_ratio = lam / mu
    stockout_factor = math.sqrt(p / h)
    safety_multiplier = math.sqrt(1.0 + disruption_ratio * stockout_factor)
    q_manual = q_classic * safety_multiplier
    
    return q_classic, q_manual, safety_multiplier
# EVOLVE-BLOCK-END
