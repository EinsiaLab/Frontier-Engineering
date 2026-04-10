# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

No stockpyl DP solver is used here.
"""

from __future__ import annotations


def solve(demand_mean, demand_sd):
    """Manual moment-based time-varying policy with improved fill rate.

    Rule:
    - s_t = round(mean_t * (0.72 + 0.15 * cv + 0.08 * (t/7)))
    - S_t = round(mean_t + (1.65 + 0.28 * cv) * sd_t + 45)
    - Ensure S_t >= s_t + max(18, round(0.30 * mean_t))
    where cv = sd_t / mean_t if mean_t > 0 else 0.
    """
    s_levels = []
    S_levels = []
    for t, (m, sd) in enumerate(zip(demand_mean, demand_sd)):
        cv = sd / m if m > 0 else 0
        # Reorder point: increase with time to account for terminal penalties
        s_mult = 0.72 + 0.15 * cv + 0.08 * (t / 7.0)
        s_t = round(m * s_mult)
        s_levels.append(s_t)
        
        # Order-up-to level: higher safety stock to improve fill rate
        S_mult = 1.65 + 0.28 * cv
        S_t = round(m + S_mult * sd + 45)
        # Ensure sufficient order quantity
        min_gap = max(18, round(0.30 * m))
        S_levels.append(max(S_t, s_t + min_gap))

    return s_levels, S_levels
# EVOLVE-BLOCK-END
