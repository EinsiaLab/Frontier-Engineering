# EVOLVE-BLOCK-START
"""Baseline implementation for Task 02.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve(*_, **__) -> dict[int, int]:
    """Simple heuristic for base‑stock levels.

    The heuristic assumes a lead‑time of one period and applies a
    safety factor of 1.5 to each sink node. Up‑stream nodes receive the
    aggregated demand of their children multiplied by a modest safety
    factor to keep service levels high while limiting cost.
    """
    # Mean demand per period for the two sink nodes (given by the task)
    mean_40 = 8.0
    mean_50 = 7.0

    # Safety factor for sink nodes (lead time + safety)
    sink_sf = 1.5

    # Base‑stock for sinks
    s40 = round(mean_40 * sink_sf)
    s50 = round(mean_50 * sink_sf)

    # Aggregate downstream demand
    downstream_total = s40 + s50

    # Safety factors for upstream nodes (slightly lower than sinks)
    upstream_sf = 1.2
    top_sf = 1.4

    # Base‑stock for level‑2 nodes (20, 30)
    s20 = round(downstream_total * upstream_sf)
    s30 = round(downstream_total * upstream_sf)

    # Base‑stock for the top node (10)
    s10 = round(downstream_total * top_sf)

    return {10: s10, 20: s20, 30: s30, 40: s40, 50: s50}
# EVOLVE-BLOCK-END
