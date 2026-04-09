# EVOLVE-BLOCK-START
"""Baseline implementation for Task 05.

No stockpyl EOQD optimizer is used here.
"""

from __future__ import annotations

import math


def solve(cfg: dict):
    q_classic = math.sqrt(2.0 * cfg["fixed_cost"] * cfg["demand_rate"] / cfg["holding_cost"])
    safety_multiplier = 1.0 + 1.311 * (cfg["disruption_rate"] / cfg["recovery_rate"])
    return q_classic, q_classic * safety_multiplier, safety_multiplier
# EVOLVE-BLOCK-END
