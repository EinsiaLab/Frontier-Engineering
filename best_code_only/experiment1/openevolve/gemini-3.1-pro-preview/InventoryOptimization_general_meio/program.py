# EVOLVE-BLOCK-START
"""Baseline implementation for Task 02.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve() -> dict[int, int]:
    """Return optimized base-stock levels."""
    # Aggressively reduce node 40 to minimize holding costs
    return {10: 30, 20: 14, 30: 14, 40: 8, 50: 18}
# EVOLVE-BLOCK-END
