# EVOLVE-BLOCK-START
"""Improved baseline for Joint Replenishment.

Uses the shortest allowed cycle time and a single order multiple
to maximize responsiveness and coordination while keeping costs low.
"""

from __future__ import annotations


def solve() -> dict:
    """Heuristic: minimal cycle + single order multiple."""

    # Shortest allowed cycle improves the responsiveness score
    base_cycle = 1.0

    # Demand rates for the eight items (units per period)
    demand_rates = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]

    # Use a single multiple for all items to maximize coordination
    multiples = [1] * len(demand_rates)

    # Order quantity = demand × multiple × cycle time
    order_quantities = [
        d_i * m_i * base_cycle for d_i, m_i in zip(demand_rates, multiples)
    ]

    return {
        "base_cycle_time": base_cycle,
        "order_multiples": multiples,
        "order_quantities": order_quantities,
    }
# EVOLVE-BLOCK-END
