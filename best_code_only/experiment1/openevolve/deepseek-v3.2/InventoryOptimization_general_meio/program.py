# EVOLVE-BLOCK-START
"""Baseline implementation for Task 02.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve() -> dict[int, int]:
    """Manual demand-coverage heuristic for base-stock levels."""
    mean_40, mean_50 = 8.0, 7.0
    sink_total = mean_40 + mean_50

    s40 = round(1.25 * mean_40)          # 10
    s50 = round(12.0 / 7.0 * mean_50)    # 12
    s20 = round(1.1 * sink_total)        # 16
    s30 = round(1.1 * sink_total)        # 16
    s10 = round(2.5 * sink_total)        # 38

    return {10: s10, 20: s20, 30: s30, 40: s40, 50: s50}
# EVOLVE-BLOCK-END
