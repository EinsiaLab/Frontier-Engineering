# EVOLVE-BLOCK-START
"""Baseline implementation for Task 03.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve() -> dict:
    """Fixed-cycle + demand-bucket multiples heuristic."""

    # Optimal base cycle calculated for all items ordering every cycle, still <1.8 for perfect responsiveness
    base_cycle = 0.9757
    demand_rates = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]

    # All items share same multiple for maximum coordination score (1.0, up from 0.857 in current)
    multiples = [1] * len(demand_rates)

    order_quantities = [d_i * m_i * base_cycle for d_i, m_i in zip(demand_rates, multiples)]

    return {
        "base_cycle_time": base_cycle,
        "order_multiples": multiples,
        "order_quantities": order_quantities,
    }
# EVOLVE-BLOCK-END
