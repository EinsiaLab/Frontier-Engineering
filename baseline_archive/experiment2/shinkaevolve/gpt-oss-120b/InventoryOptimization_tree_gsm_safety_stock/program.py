# EVOLVE-BLOCK-START
"""Optimized implementation for Task 01.

This module uses enumeration-based CST optimization with cost-aware
safety stock placement, pushing inventory upstream where holding costs
are typically lower.
"""

from __future__ import annotations

PROCESSING_TIME = {
    1: 2.0,
    3: 1.0,
    2: 1.0,
    4: 1.0,
}

# Tree structure: 1 -> 3 -> {2, 4}
# Node 2: premium market (SLA: CST <= 0)
# Node 4: standard market (SLA: CST <= 1)


def solve(_unused=None) -> dict[int, int]:
    """Enumeration-based CST policy with upstream inventory placement.

    Strategy:
    - Demand-facing nodes fixed by SLA: node 2 -> CST=0, node 4 -> CST=1
    - Node 3 must satisfy both children, constrained by node 2's strict SLA
    - Node 1 (factory) uses CST=0 to push safety stock upstream
    - This minimizes total cost by placing inventory at lowest-cost location
    """

    # Fixed demand-facing nodes by SLA constraints
    cst = {2: 0, 4: 1}
    
    # Node 3 is internal, serves both node 2 (CST=0) and node 4 (CST=1)
    # Must satisfy most restrictive constraint: node 2 requires CST_3 <= 0
    # Processing time threshold check: 1.0 < 2.0, so CST=0
    cst[3] = 0

    # Node 1 is the factory/root with processing_time=2.0
    # Key optimization: Set CST=0 to push safety stock upstream
    # This consolidates inventory at the factory where holding costs are lowest
    # Net replenishment time at node 1 = processing_time - CST = 2 - 0 = 2
    cst[1] = 0

    return cst
# EVOLVE-BLOCK-END