# EVOLVE-BLOCK-START
"""Baseline implementation for Task 03."""

from __future__ import annotations
import math


def solve() -> dict:
    """EOQ-optimal joint replenishment with perfect coordination."""
    demand_rates = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]
    holding_costs = [1.8, 2.0, 1.6, 1.7, 1.5, 1.9, 2.1, 1.4]
    individual_fixed_costs = [40.0, 35.0, 30.0, 28.0, 25.0, 22.0, 20.0, 18.0]
    shared_fixed_cost = 100.0
    
    # All items order every cycle for perfect coordination score (1.0)
    multiples = [1] * len(demand_rates)
    
    # EOQ-optimal base cycle: T* = sqrt(2 * (S + sum(K_i)) / sum(h_i * d_i))
    total_fixed = shared_fixed_cost + sum(individual_fixed_costs)
    total_holding_rate = sum(h_i * d_i for h_i, d_i in zip(holding_costs, demand_rates))
    base_cycle = math.sqrt(2.0 * total_fixed / total_holding_rate)
    
    order_quantities = [d_i * base_cycle for d_i in demand_rates]

    return {
        "base_cycle_time": base_cycle,
        "order_multiples": multiples,
        "order_quantities": order_quantities,
    }
# EVOLVE-BLOCK-END
