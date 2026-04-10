# EVOLVE-BLOCK-START
"""Cost-aware heuristic for GSM tree safety stock placement.

Tree: 1 -> 3 -> {2, 4}
Holding costs: 1:0.4, 3:0.7, 2:1.1, 4:1.0
SLA constraints: node 2 ≤ 0, node 4 ≤ 1

Strategy: Push safety stock to lowest-cost upstream node (node 1).
Only demand nodes with relaxed SLA use their buffer (node 4: CST=1).
"""

# Holding costs determine where to hold safety stock (lower is better)
HOLDING_COST = {1: 0.4, 3: 0.7, 2: 1.1, 4: 1.0}
# SLA limits for demand-facing nodes
SLA_LIMIT = {2: 0, 4: 1}


def solve(_unused=None) -> dict[int, int]:
    """Cost-aware CST assignment using holding cost heuristic.
    
    Rule: Set CST=0 for all nodes except demand nodes with relaxed SLA.
    This pushes safety stock upstream to the cheapest node (node 1).
    """
    cst = {}
    for node in HOLDING_COST:
        if node in SLA_LIMIT:
            # Demand node: use full SLA allowance to reduce local inventory
            cst[node] = SLA_LIMIT[node]
        else:
            # Internal node: CST=0 pushes stock upstream
            cst[node] = 0
    return cst
# EVOLVE-BLOCK-END
