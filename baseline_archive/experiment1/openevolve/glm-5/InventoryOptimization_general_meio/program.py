# EVOLVE-BLOCK-START
"""Baseline implementation for Task 02.

No stockpyl optimizer is used here.
"""

from __future__ import annotations


def solve() -> dict[int, int]:
    """Base-stock optimization with tuned service levels and upstream pooling."""
    import math

    # Demand parameters
    mean_40, std_40 = 8.0, 3.0
    mean_50, std_50 = 7.0, 2.5
    sink_total = mean_40 + mean_50

    # Differentiated service level factors for balance
    # Node 50 has higher CV (0.357) than node 40 (0.375), needs more buffer
    # Increase z_50 to close fill rate gap with node 40
    z_40, z_50 = 1.65, 2.00

    # Lead times based on network depth
    L_sink, L_inter, L_root = 1, 2, 3

    # Sink nodes: base stock = mean demand + safety stock
    s40 = round(mean_40 * L_sink + z_40 * std_40 * math.sqrt(L_sink))
    s50 = round(mean_50 * L_sink + z_50 * std_50 * math.sqrt(L_sink))

    # Intermediate nodes: leverage risk pooling, reduce stock (upstream is cheaper)
    # Reduce factor to shift inventory to root (lower holding cost)
    pooled_std = math.sqrt(std_40**2 + std_50**2)
    z_pool = 1.65
    s20 = round(sink_total * L_inter * 0.42 + z_pool * pooled_std * math.sqrt(L_inter) * 0.62)
    s30 = round(sink_total * L_inter * 0.42 + z_pool * pooled_std * math.sqrt(L_inter) * 0.62)

    # Root node: maximize pooling benefit (lowest holding cost = 0.2)
    # Increase factor to absorb inventory shifted from intermediate nodes
    s10 = round(sink_total * L_root * 0.65 + z_pool * pooled_std * math.sqrt(L_root) * 0.50)

    return {10: s10, 20: s20, 30: s30, 40: s40, 50: s50}
# EVOLVE-BLOCK-END
