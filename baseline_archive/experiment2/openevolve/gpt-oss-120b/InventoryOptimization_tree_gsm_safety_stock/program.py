# EVOLVE-BLOCK-START
"""Baseline implementation for Task 01.

This module intentionally avoids stockpyl and only contains a simple
rule-based CST assignment.
"""

from __future__ import annotations

PROCESSING_TIME = {
    1: 2.0,
    3: 1.0,
    2: 1.0,
    4: 1.0,
}


def solve(_unused=None) -> dict[int, int]:
    """Simple CST policy improving baseline cost.

    Strategy:
    - Keep all CST values at 0 (same as the all‑zero baseline) except
      node 4, which can be set to 1. This respects the SLA
      (node 4 allowed ≤ 1), changes only one node (ComplexityScore = 1),
      and yields a lower inventory cost, leading to higher CostScore
      and RobustnessScore.
    """

    # Initialise all nodes with CST = 0.
    cst: dict[int, int] = {node: 0 for node in PROCESSING_TIME}
    # Adjust node 4 to 1 to reduce cost while staying within its SLA bound.
    cst[4] = 1
    return cst
# EVOLVE-BLOCK-END
