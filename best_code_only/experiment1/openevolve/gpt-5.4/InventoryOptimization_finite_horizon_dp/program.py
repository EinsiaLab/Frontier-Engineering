# EVOLVE-BLOCK-START
"""Service-first heuristic for Task 04."""

from __future__ import annotations


def solve(demand_mean, demand_sd):
    n = len(demand_mean)
    if not n:
        return [], []
    avg = sum(demand_mean) / n
    peak = max(demand_mean)
    s_levels, S_levels = [], []
    for i, (m, sd) in enumerate(zip(demand_mean, demand_sd)):
        nxt = demand_mean[i + 1] if i + 1 < n else m
        prev = demand_mean[i - 1] if i else m
        up = nxt > m
        down = prev > m
        late = i >= n - 2
        peakish = m > 0.8 * peak
        high = m > avg
        s = round(m * (0.7 + 0.04 * high + 0.04 * up + 0.02 * late))
        gap = 28 + 1.45 * sd + 0.08 * m + 7 * up + 5 * peakish - 3 * down
        s_levels.append(max(0, s))
        S_levels.append(max(round(m + gap), s + 7 + up))
    return s_levels, S_levels
# EVOLVE-BLOCK-END
