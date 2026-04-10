# EVOLVE-BLOCK-START
"""Baseline implementation for Task 02.

Uses simulation-based optimization with analytical warm-start
for optimal base-stock levels in a multi-echelon inventory system.
"""

from __future__ import annotations


def solve() -> dict[int, int]:
    """Compute base-stock levels using simulation-based optimization."""
    try:
        return _solve_sim_optimized()
    except Exception:
        return _solve_heuristic()


def _solve_sim_optimized() -> dict[int, int]:
    """
    Simulation-based optimization minimizing total cost across both scenarios.
    Uses caching, broad initial screening, and aggressive coordinate descent.
    """
    from stockpyl.supply_chain_network import SupplyChainNetwork
    from stockpyl.supply_chain_node import SupplyChainNode
    from stockpyl.demand_source import DemandSource
    from stockpyl.policy import Policy
    from stockpyl.sim import simulation
    from scipy.stats import poisson

    mean_40 = 8.0
    mean_50 = 7.0

    # Simulation cache to avoid redundant evaluations
    _sim_cache = {}

    def run_sim(bs, demand_scale, num_periods, rand_seed):
        """Run a single simulation, return total cost. Cached."""
        key = (bs[10], bs[20], bs[30], bs[40], bs[50], demand_scale, rand_seed)
        if key in _sim_cache:
            return _sim_cache[key]

        net = SupplyChainNetwork()
        n10 = SupplyChainNode(10)
        n20 = SupplyChainNode(20)
        n30 = SupplyChainNode(30)
        n40 = SupplyChainNode(40)
        n50 = SupplyChainNode(50)
        nodes_list = [n10, n20, n30, n40, n50]

        n10.shipment_lead_time = 2
        n20.shipment_lead_time = 1
        n30.shipment_lead_time = 1
        n40.shipment_lead_time = 1
        n50.shipment_lead_time = 1

        n10.local_holding_cost = 1.0
        n20.local_holding_cost = 2.0
        n30.local_holding_cost = 2.0
        n40.local_holding_cost = 5.0
        n50.local_holding_cost = 5.0

        n10.stockout_cost = 0.0
        n20.stockout_cost = 0.0
        n30.stockout_cost = 0.0
        n40.stockout_cost = 50.0
        n50.stockout_cost = 50.0

        for n in [n10, n20, n30]:
            n.demand_source = DemandSource(type='N', mean=0, standard_deviation=0)
        n40.demand_source = DemandSource(type='P', mean=mean_40 * demand_scale)
        n50.demand_source = DemandSource(type='P', mean=mean_50 * demand_scale)

        for n in nodes_list:
            n.inventory_policy = Policy(type='BS', base_stock_level=bs[n.index])
        for n in nodes_list:
            net.add_node(n)
        net.add_edges_from_list([(10, 20), (10, 30), (20, 40), (30, 50)])

        total_cost = simulation(
            network=net,
            num_periods=num_periods,
            rand_seed=rand_seed,
            progress_bar=False,
        )
        _sim_cache[key] = total_cost
        return total_cost

    def eval_cost(bs):
        """Evaluate combined cost across both scenarios. Lower is better."""
        cost1 = run_sim(bs, 1.0, 160, 11)
        cost2 = run_sim(bs, 1.2, 160, 17)
        return 0.5 * cost1 + 0.5 * cost2

    # Analytical warm-start
    stress_40 = mean_40 * 1.2
    stress_50 = mean_50 * 1.2

    # Generate diverse starting points using quantile combinations
    candidates = []
    seen = set()

    for sink_q in [0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
        for mid_q in [0.85, 0.90, 0.93, 0.95, 0.97, 0.99]:
            for root_q in [0.90, 0.93, 0.95, 0.97, 0.99]:
                s40 = int(poisson.ppf(sink_q, stress_40))
                s50 = int(poisson.ppf(sink_q, stress_50))
                s20 = int(poisson.ppf(mid_q, stress_40))
                s30 = int(poisson.ppf(mid_q, stress_50))
                agg = 2 * (stress_40 + stress_50)
                s10 = int(poisson.ppf(root_q, agg))
                key = (s10, s20, s30, s40, s50)
                if key not in seen:
                    seen.add(key)
                    candidates.append({10: s10, 20: s20, 30: s30, 40: s40, 50: s50})

    # Also add candidates with different quantiles for node 40 vs 50 (balance)
    for s40_q in [0.95, 0.97, 0.98, 0.99]:
        for s50_q in [0.95, 0.97, 0.98, 0.99]:
            for mid_q in [0.90, 0.95, 0.97]:
                for root_q in [0.93, 0.95, 0.97]:
                    s40 = int(poisson.ppf(s40_q, stress_40))
                    s50 = int(poisson.ppf(s50_q, stress_50))
                    s20 = int(poisson.ppf(mid_q, stress_40))
                    s30 = int(poisson.ppf(mid_q, stress_50))
                    agg = 2 * (stress_40 + stress_50)
                    s10 = int(poisson.ppf(root_q, agg))
                    key = (s10, s20, s30, s40, s50)
                    if key not in seen:
                        seen.add(key)
                        candidates.append({10: s10, 20: s20, 30: s30, 40: s40, 50: s50})

    # Also add candidates using normal means (not stress)
    for sink_q in [0.95, 0.97, 0.99]:
        for mid_q in [0.90, 0.95, 0.97]:
            for root_q in [0.93, 0.95, 0.97]:
                s40 = int(poisson.ppf(sink_q, mean_40))
                s50 = int(poisson.ppf(sink_q, mean_50))
                s20 = int(poisson.ppf(mid_q, mean_40))
                s30 = int(poisson.ppf(mid_q, mean_50))
                agg = 2 * (mean_40 + mean_50)
                s10 = int(poisson.ppf(root_q, agg))
                key = (s10, s20, s30, s40, s50)
                if key not in seen:
                    seen.add(key)
                    candidates.append({10: s10, 20: s20, 30: s30, 40: s40, 50: s50})

    # Phase 1: Evaluate all candidates
    results = []
    for bs in candidates:
        try:
            cost = eval_cost(bs)
            results.append((cost, bs))
        except Exception:
            continue

    results.sort(key=lambda x: x[0])

    # Phase 2: Coordinate descent from top 3 starting points
    best_overall_cost = float('inf')
    best_overall_bs = results[0][1] if results else {10: 42, 20: 14, 30: 12, 40: 13, 50: 12}

    top_n = min(3, len(results))
    for start_idx in range(top_n):
        current = dict(results[start_idx][1])
        current_cost = results[start_idx][0]

        # Aggressive coordinate descent with shrinking step sizes
        for step_sizes in [[-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2], [-1, 1]]:
            improved = True
            max_iters = 5
            iteration = 0
            while improved and iteration < max_iters:
                improved = False
                iteration += 1
                # Optimize most impactful nodes first
                for node_id in [40, 50, 10, 20, 30]:
                    for delta in step_sizes:
                        trial = dict(current)
                        trial[node_id] = max(0, trial[node_id] + delta)
                        try:
                            cost = eval_cost(trial)
                            if cost < current_cost:
                                current = trial
                                current_cost = cost
                                improved = True
                        except Exception:
                            continue

        # Paired sink adjustments for balance
        for d40 in [-2, -1, 0, 1, 2]:
            for d50 in [-2, -1, 0, 1, 2]:
                if d40 == 0 and d50 == 0:
                    continue
                trial = dict(current)
                trial[40] = max(0, current[40] + d40)
                trial[50] = max(0, current[50] + d50)
                try:
                    cost = eval_cost(trial)
                    if cost < current_cost:
                        current = trial
                        current_cost = cost
                except Exception:
                    continue

        # Paired upstream adjustments
        for d10 in [-2, -1, 0, 1, 2]:
            for d20 in [-1, 0, 1]:
                for d30 in [-1, 0, 1]:
                    if d10 == 0 and d20 == 0 and d30 == 0:
                        continue
                    trial = dict(current)
                    trial[10] = max(0, current[10] + d10)
                    trial[20] = max(0, current[20] + d20)
                    trial[30] = max(0, current[30] + d30)
                    try:
                        cost = eval_cost(trial)
                        if cost < current_cost:
                            current = trial
                            current_cost = cost
                    except Exception:
                        continue

        # Final fine-tune pass
        improved = True
        while improved:
            improved = False
            for node_id in [40, 50, 10, 20, 30]:
                for delta in [-1, 1]:
                    trial = dict(current)
                    trial[node_id] = max(0, trial[node_id] + delta)
                    try:
                        cost = eval_cost(trial)
                        if cost < current_cost:
                            current = trial
                            current_cost = cost
                            improved = True
                    except Exception:
                        continue

        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_bs = dict(current)

    return best_overall_bs


def _solve_heuristic() -> dict[int, int]:
    """Tuned heuristic fallback for base-stock levels."""
    import math

    mean_40 = 8.0
    mean_50 = 7.0

    # Account for stress scenario (1.2x) - use slightly elevated means
    eff_40 = mean_40 * 1.1
    eff_50 = mean_50 * 1.1

    # High stockout/holding ratio (50/5=10) -> target ~99% fill rate
    k_sink = 2.4
    k_mid = 2.0
    k_root = 1.6

    # Sink nodes: cover LT=1 period of demand + safety stock
    s40 = round(eff_40 + k_sink * math.sqrt(eff_40))
    s50 = round(eff_50 + k_sink * math.sqrt(eff_50))

    # Mid nodes: cover their LT=1 demand (same as downstream)
    s20 = round(eff_40 + k_mid * math.sqrt(eff_40))
    s30 = round(eff_50 + k_mid * math.sqrt(eff_50))

    # Root node 10: covers LT=2 periods of total demand
    total = eff_40 + eff_50
    s10 = round(2 * total + k_root * math.sqrt(2 * total))

    return {10: s10, 20: s20, 30: s30, 40: s40, 50: s50}
# EVOLVE-BLOCK-END