# EVOLVE-BLOCK-START
"""Baseline implementation for Task 02.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve() -> dict[int, int]:
    """Manual demand-coverage heuristic for base-stock levels."""

    mean_40 = 8.0
    mean_50 = 7.0
    sink_total = mean_40 + mean_50

    # Optimized multipliers: higher sink factors for better service, differentiated upstream factors for robustness & balance
    s40 = round(2.25 * mean_40)
    s50 = round(2.3 * mean_50)  # Optimize sink 50 stock: reduce high-cost holding while maintaining near 100% fill rate for high balance score
    s20 = round(1.1 * sink_total)
    s30 = round(1.2 * sink_total)
    s10 = round(2.1 * sink_total)  # Tiny increase at lowest-cost top node to boost stress scenario robustness with negligible cost impact

    return {10: s10, 20: s20, 30: s30, 40: s40, 50: s50}
# EVOLVE-BLOCK-END
