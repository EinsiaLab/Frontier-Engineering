# EVOLVE-BLOCK-START
"""Baseline implementation for Task 02.

No stockpyl optimizer is used here.
"""

from __future__ import annotations
import math


def _poisson_cdf(k: int, mean: float) -> float:
    """Compute Poisson CDF P(X <= k) using cumulative calculation."""
    if mean <= 0:
        return 1.0 if k >= 0 else 0.0
    log_pmf = -mean
    cdf = math.exp(log_pmf) if k >= 0 else 0.0
    for i in range(1, k + 1):
        log_pmf += math.log(mean / i)
        cdf += math.exp(log_pmf)
    return min(cdf, 1.0)


def _poisson_inverse_cdf(target: float, mean: float) -> int:
    """Find smallest k such that P(X <= k) >= target."""
    if mean <= 0:
        return 0
    k = max(0, int(mean))
    while _poisson_cdf(k, mean) < target:
        k += 1
    while k > 0 and _poisson_cdf(k - 1, mean) >= target:
        k -= 1
    return k


def solve() -> dict[int, int]:
    """Newsvendor-based base-stock optimization using critical fractile.

    Key algorithmic approach:
    1. Uses Poisson distribution properties directly, not approximations
    2. Calculates exact base-stock for target service levels
    3. Accounts for network lead time structure:
       - Node 10: 2-period lead time, pooled demand from both branches
       - Nodes 20, 30: 2-period effective lead time (upstream + local)
       - Nodes 40, 50: 1-period lead time, direct customer facing
    4. Risk pooling benefit: central node holds less than sum of branches
    5. Optimized service levels for cost-service-robustness-balance tradeoff

    Scoring weights: CostScore (0.30), ServiceScore (0.35), RobustnessScore (0.25), BalanceScore (0.10)
    """

    # Demand parameters (Poisson means)
    mean_40 = 8.0  # Demand at sink node 40
    mean_50 = 7.0  # Demand at sink node 50
    total_mean = mean_40 + mean_50  # Combined demand = 15

    # Target service levels (critical fractile)
    # Optimized for the scoring function:
    # - Node 50 (lower mean, higher CV) needs higher service for balance
    # - Upstream nodes use lower service due to risk pooling benefit
    # - This balances cost reduction with service and robustness
    service_40 = 0.96      # Node 40: high service for fill rate
    service_50 = 0.97      # Node 50: higher to compensate for higher CV, improves balance
    service_10 = 0.88      # Central node: lower due to strong risk pooling benefit (pooled demand CV reduction)
    service_intermediate = 0.91  # Intermediate nodes (20, 30): moderate service for robustness

    # Lead times in the network
    # Node 10: LT = 2 (central warehouse)
    # Nodes 20, 30: LT = 1 each, but depend on node 10 (effective 2 periods)
    # Nodes 40, 50: LT = 1 each (sink nodes)

    # Sink nodes: base-stock covers 1-period lead time demand
    s40 = _poisson_inverse_cdf(service_40, mean_40)
    s50 = _poisson_inverse_cdf(service_50, mean_50)

    # Intermediate nodes: cover 2-period effective lead time
    # They face demand from their downstream sink over 2 periods
    s20 = _poisson_inverse_cdf(service_intermediate, mean_40 * 2)
    s30 = _poisson_inverse_cdf(service_intermediate, mean_50 * 2)

    # Central node: pools demand from both branches
    # Lead time = 2 periods, combined demand mean = 15 * 2 = 30
    # Risk pooling significantly reduces safety stock needs - use lower service level
    s10 = _poisson_inverse_cdf(service_10, total_mean * 2)

    return {10: s10, 20: s20, 30: s30, 40: s40, 50: s50}
# EVOLVE-BLOCK-END