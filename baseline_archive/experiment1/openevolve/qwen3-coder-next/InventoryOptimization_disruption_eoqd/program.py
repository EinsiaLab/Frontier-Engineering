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
    
    # Calculate safety multiplier based on disruption dynamics
    # Using a responsive formula that captures disruption risk
    disruption_ratio = cfg["disruption_rate"] / max(cfg["recovery_rate"], 1e-8)
    
    # Apply quadratic growth for high disruption ratios to significantly increase order quantities
    # when disruption risk is high relative to recovery rate
    base_multiplier = 1.0 + 0.4 * disruption_ratio
    quadratic_factor = 0.15 * disruption_ratio * disruption_ratio
    safety_multiplier = base_multiplier + quadratic_factor
    
    # Add cost-aware adjustment: higher stockout cost justifies more inventory
    cost_ratio = cfg["stockout_cost"] / max(cfg["holding_cost"], 1e-8)
    cost_adjustment = 1.0 + 0.03 * min(cost_ratio, 8.0)
    
    # Apply moderate bounds to balance inventory costs with service levels
    safety_multiplier = max(1.0, min(safety_multiplier * cost_adjustment, 3.2))
    
    # Final adjustment based on disruption intensity for fine-tuning
    intensity_factor = 1.0 + 0.02 * disruption_ratio / (1.0 + disruption_ratio)
    q_manual = q_classic * safety_multiplier * intensity_factor
    
    return q_classic, q_manual, safety_multiplier
# EVOLVE-BLOCK-END
