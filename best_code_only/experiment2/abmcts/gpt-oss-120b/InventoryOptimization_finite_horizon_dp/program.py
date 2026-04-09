# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

Improved heuristic: uses a newsvendor‑style safety‑stock factor (≈1.33) that
balances holding and backorder costs, while keeping a modest reorder‑point
offset.  Both levels are rounded to integers and constrained so that S is
at least 6 units larger than s."""
from __future__ import annotations
import math
from typing import List, Tuple

def solve(demand_mean: List[float], demand_sd: List[float]) -> Tuple[List[int], List[int]]:
    """
    Compute time‑varying (s, S) policy parameters.

    Parameters
    ----------
    demand_mean : list of float
        Expected demand for each period.
    demand_sd : list of float
        Standard deviation of demand for each period.

    Returns
    -------
    s_levels : list of int
        Reorder‑point for each period.
    S_levels : list of int
        Order‑up‑to level for each period.

    Heuristic
    ---------
    * Approximate newsvendor critical fractile with holding cost = 1,
      backorder cost = 10 → service factor ≈ 0.909 → z ≈ 1.33.
    * S_t = round(mean_t + 1.33 × sd_t)      (safety stock)
    * s_t = round(mean_t - 0.5 × sd_t)      (lower bound, may be negative)
    * Enforce S_t ≥ s_t + 6 and s_t ≥ 0.
    """
    safety_factor = 1.33  # ≈ norm.ppf(0.909) for h=1, b=10
    s_levels: List[int] = []
    S_levels: List[int] = []

    for m, sd in zip(demand_mean, demand_sd):
        # order‑up‑to level with newsvendor safety stock
        S = round(m + safety_factor * sd)

        # reorder point a bit below the mean
        s = round(m - 0.5 * sd)

        # enforce non‑negative s
        s = max(s, 0)

        # ensure sufficient gap between s and S
        if S < s + 6:
            S = s + 6

        s_levels.append(s)
        S_levels.append(S)

    return s_levels, S_levels
# EVOLVE-BLOCK-END
