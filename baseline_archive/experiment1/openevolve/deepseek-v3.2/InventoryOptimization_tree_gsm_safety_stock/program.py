# EVOLVE-BLOCK-START
"""Baseline implementation for Task 01.

This module uses a heuristic to choose the best single-node CST change using stockpyl cost evaluation.
"""

from __future__ import annotations
from verification.reference import build_tree
from stockpyl.gsm_helpers import solution_cost_from_cst


def solve(_unused=None) -> dict[int, int]:
    """Rule-based CST policy that selects the best single-node modification via cost evaluation.
    
    Rule:
    - Demand-facing node 2 CST=0 (strict SLA).
    - Consider all single-change alternatives relative to baseline {1:0,3:0,2:0,4:0}:
        Change CST of node 1, 3, or 4 to a valid value within SLA bounds.
    - Use stockpyl's solution_cost_from_cst to compute nominal cost for each candidate.
    - Return the CST with lowest nominal cost.
    Complexity_score remains 1 because only one node differs from baseline.
    """
    baseline_cst = {1: 0, 3: 0, 2: 0, 4: 0}
    # SLA upper bounds: node 2 <=0, node 4 <=1, internal nodes have no explicit SLA
    # For internal nodes, CST can be 0 or 1 (since processing_time >=2 for node1 suggests CST=1 possible)
    candidates = []
    # Candidate: change node 1 CST to 1
    cand1 = {1: 1, 3: 0, 2: 0, 4: 0}
    # Candidate: change node 3 CST to 1 (processing_time=1, but CST=1 allowed)
    cand2 = {1: 0, 3: 1, 2: 0, 4: 0}
    # Candidate: change node 4 CST to 1 (SLA allows)
    cand3 = {1: 0, 3: 0, 2: 0, 4: 1}
    # Also consider CST=2 for node4? SLA upper bound is 1, so not allowed.
    
    nominal_tree = build_tree(1.0)
    costs = []
    for cand in [cand1, cand2, cand3]:
        cost = solution_cost_from_cst(nominal_tree, cand)
        costs.append(cost)
    
    # Find minimum cost candidate
    min_idx = costs.index(min(costs))
    best_cand = [cand1, cand2, cand3][min_idx]
    return best_cand
# EVOLVE-BLOCK-END
