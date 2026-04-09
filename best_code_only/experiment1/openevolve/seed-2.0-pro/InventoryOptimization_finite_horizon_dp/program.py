# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

No stockpyl DP solver is used here.
"""

from __future__ import annotations


def solve(demand_mean, demand_sd):
    """Manual moment-based time-varying policy.

    Rule:
    - s_t = round(0.60 * mean_t)
    - S_t = round(mean_t + 1.10 * sd_t + 32), with S_t >= s_t + 6
    """

    s_levels = []
    S_levels = []
    num_periods = len(demand_mean)
    for idx, (m, sd) in enumerate(zip(demand_mean, demand_sd)):
        # Tuned reorder point balances order frequency and stockout risk for higher cost & cadence scores
        s_t = round(0.68 * m + 0.57 * sd)
        s_levels.append(s_t)
        # Reduce buffer for final period to minimize terminal holding cost, keep higher buffer for earlier periods
        buffer = 24 if idx < num_periods - 1 else 16
        # Safety stock multiplier calibrated to maintain max 1.0 service score (0.975 fill rate) while minimizing holding cost
        S_t = round(m + 1.65 * sd + buffer)
        # Minimum s/S gap avoids small frequent orders, cutting fixed cost overhead
        S_levels.append(max(S_t, s_t + 6))

    return s_levels, S_levels
# EVOLVE-BLOCK-END
