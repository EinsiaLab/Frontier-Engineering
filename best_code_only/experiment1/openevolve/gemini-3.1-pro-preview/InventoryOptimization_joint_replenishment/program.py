# EVOLVE-BLOCK-START
"""Baseline implementation for Task 03.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve() -> dict:
    """Fixed-cycle + demand-bucket multiples heuristic."""

    shared_fixed_cost = 100.0
    individual_fixed_costs = [40.0, 35.0, 30.0, 28.0, 25.0, 22.0, 20.0, 18.0]
    holding_costs = [1.8, 2.0, 1.6, 1.7, 1.5, 1.9, 2.1, 1.4]
    demand_rates = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]

    K_total = shared_fixed_cost + sum(individual_fixed_costs)
    H_D = sum(h * d for h, d in zip(holding_costs, demand_rates))
    
    c = (2.0 * K_total / H_D) ** 0.5
    m = [1] * len(demand_rates)

    return {
        "base_cycle_time": c,
        "order_multiples": m,
        "order_quantities": [d * m_i * c for d, m_i in zip(demand_rates, m)],
    }
# EVOLVE-BLOCK-END
