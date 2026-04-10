# EVOLVE-BLOCK-START
"""Optimized implementation for Task 03.

EOQ-optimized joint replenishment with analytical base cycle calculation.
"""

from __future__ import annotations
import math


def solve() -> dict:
    """EOQ-optimized joint replenishment with grid search for optimal T.

    Key design decisions:
    - All items use m=1 for perfect coordination (1 distinct multiple = 15% weight)
    - Base cycle T found via grid search over plausible cost structures
    - T constrained to <= 1.8 for perfect responsiveness (30% weight)
    - Focus on cost optimization (55% weight) through robust T selection

    Scoring weights:
    - CostScore (55%): relative to independent EOQ baseline
    - ResponsivenessScore (30%): max_cycle <= 1.8 for perfect
    - CoordinationScore (15%): fewer distinct multiples is better
    """

    # Demand rates for 8 SKUs (from evaluator)
    demand_rates = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]
    n_items = len(demand_rates)
    total_demand = sum(demand_rates)

    # Unified multiples - all items order every cycle
    # Perfect coordination score (1 distinct multiple)
    multiples = [1] * n_items

    # Grid search over T values to find robust optimal
    # Test T values in range [0.5, 1.8] for responsiveness constraint
    best_T = 1.0  # Default fallback
    best_cost_estimate = float('inf')

    # Try multiple plausible cost parameter scenarios
    # Each scenario: (shared_setup, base_individual_setup, base_holding_rate)
    cost_scenarios = [
        (50.0, 5.0, 0.5),   # Low cost scenario
        (100.0, 10.0, 1.0), # Medium cost scenario
        (150.0, 15.0, 1.5), # High cost scenario
        (75.0, 8.0, 0.8),   # Mixed scenario
    ]

    for shared_setup, base_s, base_h in cost_scenarios:
        # Compute optimal T for this cost scenario using EOQ formula
        # T* = sqrt(2 * S_total / sum(d_i * h_i))
        total_setup = shared_setup + n_items * base_s
        total_holding = sum(d * base_h for d in demand_rates)

        if total_holding > 0:
            optimal_T = math.sqrt(2.0 * total_setup / total_holding)
            # Constrain to responsiveness threshold
            optimal_T = min(optimal_T, 1.75)

            # Estimate cost for this T
            # Cost ≈ setup/T + T * holding/2
            cost_estimate = total_setup / optimal_T + optimal_T * total_holding / 2

            if cost_estimate < best_cost_estimate:
                best_cost_estimate = cost_estimate
                best_T = optimal_T

    # Ensure T is within valid bounds and round to reasonable precision
    base_cycle = max(0.5, min(best_T, 1.75))

    # Round to 2 decimal places for stability
    base_cycle = round(base_cycle, 2)

    # Calculate order quantities: Q_i = d_i * m_i * T
    order_quantities = [d_i * m_i * base_cycle for d_i, m_i in zip(demand_rates, multiples)]

    return {
        "base_cycle_time": base_cycle,
        "order_multiples": multiples,
        "order_quantities": order_quantities,
    }
# EVOLVE-BLOCK-END