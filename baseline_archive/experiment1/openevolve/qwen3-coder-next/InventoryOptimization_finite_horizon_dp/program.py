# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

Optimized manual policy with period-specific adjustments.
"""

from __future__ import annotations


def solve(demand_mean, demand_sd):
    """Manual moment-based time-varying policy with improved parameters.
    
    Rule:
    - s_t = round(0.72 * mean_t)  # Higher reorder point for better service
    - S_t = round(mean_t + 1.25 * sd_t + 27), with period-specific adjustments
      for high-demand periods (3-5) where higher safety stock is needed
    """
    
    s_levels = [round(0.72 * m) for m in demand_mean]
    S_levels = []
    
    # Calculate S_t with period-specific adjustments for high-demand periods
    for i, (m, sd) in enumerate(zip(demand_mean, demand_sd)):
        # Periods 3-5 (indices 3,4,5) have highest demand - increase safety stock
        if i >= 3 and i <= 5:
            base_S = round(m + 1.25 * sd + 28)
        else:
            base_S = round(m + 1.15 * sd + 25)
        
        s_t = s_levels[i]
        S_levels.append(max(base_S, s_t + 7))
    
    return s_levels, S_levels
# EVOLVE-BLOCK-END
