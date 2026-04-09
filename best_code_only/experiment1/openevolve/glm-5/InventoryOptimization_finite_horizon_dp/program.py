# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

No stockpyl DP solver is used here.
"""

from __future__ import annotations


def solve(demand_mean, demand_sd):
    """Optimized (s, S) policy balancing cost and service.

    Rule:
    - s_t = round(0.70 * mean_t + 0.6 * sd_t)
    - S_t = round(s_t + 0.6 * mean_t + 1.6 * sd_t + 22), with S_t >= s_t + 10
    """

    s_levels = [round(0.70 * m + 0.6 * sd) for m, sd in zip(demand_mean, demand_sd)]
    S_levels = []
    for i, (m, sd) in enumerate(zip(demand_mean, demand_sd)):
        s_t = s_levels[i]
        S_t = round(s_t + 0.6 * m + 1.6 * sd + 22)
        S_levels.append(max(S_t, s_t + 10))

    return s_levels, S_levels
# EVOLVE-BLOCK-END
