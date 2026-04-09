# EVOLVE-BLOCK-START
"""Baseline implementation for Task 03.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve() -> dict:
    """Fixed-cycle + demand-bucket multiples heuristic with optimized parameters."""

    # Base cycle time optimized for responsiveness (max cycle ≤ 1.8)
    # Using 1.0 as initial value, then adjusting after determining multiples
    base_cycle = 1.0
    
    demand_rates = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]

    # Initial multiples based on demand rate groups with more granular thresholds
    # Following Silver heuristic approach more closely
    # Try to balance between responsiveness and coordination
    multiples = []
    for d_i in demand_rates:
        if d_i >= 60:
            m_i = 1
        elif d_i >= 45:
            m_i = 2
        elif d_i >= 30:
            m_i = 3
        elif d_i >= 20:
            m_i = 4
        else:
            m_i = 5
        multiples.append(m_i)
    
    # After determining initial multiples, adjust to ensure responsiveness while 
    # potentially reducing the number of unique multiples for better coordination
    max_allowed_multiple = int(1.8 / base_cycle)
    multiples = [min(m, max_allowed_multiple) for m in multiples]
    
    # Optional: further reduce unique multiples if possible without violating responsiveness
    # This could improve coordination score
    unique_multiples = sorted(set(multiples))
    if len(unique_multiples) > 1:
        # Try to consolidate to fewer multiples if possible
        # For example, if we have multiples [1,2,3,4,5], try to reduce to [1,2,3,4]
        # by adjusting the boundary thresholds
        pass  # Keep original logic for now, collapse might reduce cost performance

    # Enforce responsiveness constraint: max cycle ≤ 1.8
    max_allowed_multiple = int(1.8 / base_cycle)
    multiples = [min(m, max_allowed_multiple) for m in multiples]
    
    # Fine-tune base_cycle for optimal cost-performance balance
    # Based on high-performing versions, values around 0.97 work well
    # Try 0.97 to potentially improve cost score while maintaining responsiveness
    base_cycle = 0.97

    order_quantities = [d_i * m_i * base_cycle for d_i, m_i in zip(demand_rates, multiples)]

    return {
        "base_cycle_time": base_cycle,
        "order_multiples": multiples,
        "order_quantities": order_quantities,
    }
# EVOLVE-BLOCK-END
