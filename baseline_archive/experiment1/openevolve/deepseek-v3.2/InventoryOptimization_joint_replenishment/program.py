# EVOLVE-BLOCK-START
"""Baseline implementation for Task 03.

No stockpyl optimizer is used here.
"""

from __future__ import annotations
import math


def solve() -> dict:
    """Optimized search for base cycle and multiples to maximize score."""
    
    # Problem parameters (same as in evaluator)
    demand_rates = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]
    shared_fixed_cost = 100.0
    individual_fixed_costs = [40.0, 35.0, 30.0, 28.0, 25.0, 22.0, 20.0, 18.0]
    holding_costs = [1.8, 2.0, 1.6, 1.7, 1.5, 1.9, 2.1, 1.4]
    # We don't need baseline_cost for our calculation, but keep it for reference
    
    # Direct analytical solution: all multiples = 1 gives perfect coordination_score and responsiveness_score if T <= 1.8
    # Compute optimal T for all m=1
    A = shared_fixed_cost + sum(individual_fixed_costs)
    B = sum(h_i * d_i for h_i, d_i in zip(holding_costs, demand_rates))
    # Note: total_cost = A/T + (B/2)*T
    # Actually, in the evaluator, holding cost term is h_i*d_i*(m_i*T)/2, so sum is (B/2)*T where B = sum(h_i*d_i)
    # But the formula we derived earlier: total_cost = A/T + B*T/2
    # So optimal T = sqrt(2*A/B)
    T_opt = math.sqrt(2.0 * A / B)
    
    # Ensure responsiveness_score = 1.0 by capping T at 1.8
    if T_opt > 1.8:
        T_opt = 1.8
    
    # All multiples = 1
    best_multiples = [1] * len(demand_rates)
    best_base_cycle = T_opt
    best_order_quantities = [d_i * best_base_cycle for d_i in demand_rates]

    return {
        "base_cycle_time": best_base_cycle,
        "order_multiples": best_multiples,
        "order_quantities": best_order_quantities,
    }
# EVOLVE-BLOCK-END
