# EVOLVE-BLOCK-START
from __future__ import annotations

def solve() -> dict[int, int]:
    """Upstream-heavy heuristic with improved sink balance.
    
    Holding costs: 10=0.2, 20/30=0.4, 40/50=0.9
    Strategy: massive upstream buffer at cheapest node 10, lean expensive downstream.
    
    Previous: {10:42, 20:11, 30:11, 40:11, 50:14} -> 0.9911
      fill_40=1.0, fill_50=0.9955, gap=0.0045, BalanceScore=0.91
    
    Improvement: increase node 50 from 14->15 to boost fill_50 closer to 1.0.
    Cost increase ~0.9/period, but CostScore has ~0.88 headroom (stays at 1.0).
    RobustnessScore also has large headroom (stays at 1.0).
    Expected: gap shrinks significantly -> BalanceScore approaches 1.0.
    Also reduce node 40 from 11->10: fill_40 is already 1.0 with huge margin
    from upstream buffer. This saves 0.9/period, offsetting node 50 increase.
    If fill_40 drops slightly toward fill_50, that actually helps balance too.
    """
    return {10: 42, 20: 11, 30: 11, 40: 10, 50: 15}
# EVOLVE-BLOCK-END
