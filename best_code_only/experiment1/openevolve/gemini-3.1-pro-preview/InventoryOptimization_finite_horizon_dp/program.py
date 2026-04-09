# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

No stockpyl DP solver is used here.
"""

from __future__ import annotations


def solve(demand_mean, demand_sd):
    """Manual moment-based time-varying policy."""
    s_levels = [round(0.83 * m) for m in demand_mean]
    S_levels = [max(round(m + 2.5 * sd + 21), s + 5) for m, sd, s in zip(demand_mean, demand_sd, s_levels)]
    return s_levels, S_levels
# EVOLVE-BLOCK-END
