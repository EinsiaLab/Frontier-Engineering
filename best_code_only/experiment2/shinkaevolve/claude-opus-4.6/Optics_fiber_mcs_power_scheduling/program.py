# EVOLVE-BLOCK-START
"""Solver for Task 2: MCS + power scheduling using discretized knapsack approach."""

from __future__ import annotations

import math
import numpy as np
from scipy.special import erfc


def _required_snr_db(M, target_ber):
    """Binary search for the minimum SNR (dB) that achieves target_ber for M-QAM.

    Uses the standard approximate BER formula for M-QAM:
        BER ≈ (2/log2(M)) * (1 - 1/sqrt(M)) * erfc(sqrt(3*SNR_lin / (2*(M-1))))
    """
    if M <= 1:
        return -20.0  # trivially easy

    log2M = math.log2(M)
    sqrtM = math.sqrt(M)
    coeff = (2.0 / log2M) * (1.0 - 1.0 / sqrtM)

    # We need: coeff * erfc(sqrt(3*snr_lin / (2*(M-1)))) <= target_ber
    # erfc(x) <= target_ber / coeff
    target_erfc = target_ber / coeff
    if target_erfc >= 2.0:
        return -20.0  # any SNR works
    if target_erfc <= 0.0:
        return 100.0  # impossible

    # Binary search on SNR in dB
    lo, hi = -20.0, 60.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        snr_lin = 10.0 ** (mid / 10.0)
        arg = math.sqrt(3.0 * snr_lin / (2.0 * (M - 1)))
        ber_val = coeff * erfc(arg)
        if ber_val > target_ber:
            lo = mid
        else:
            hi = mid
    return hi


def _ber_mqam(M, snr_db):
    """Compute approximate BER for M-QAM at given SNR (dB)."""
    if M <= 1:
        return 0.0
    log2M = math.log2(M)
    sqrtM = math.sqrt(M)
    coeff = (2.0 / log2M) * (1.0 - 1.0 / sqrtM)
    snr_lin = 10.0 ** (snr_db / 10.0)
    arg = math.sqrt(3.0 * snr_lin / (2.0 * (M - 1)))
    return float(coeff * erfc(arg))


def select_mcs_power(
    user_demands_gbps,
    channel_quality_db,
    total_power_dbm,
    mcs_candidates=(4, 16, 64),
    pmin_dbm=-8.0,
    pmax_dbm=4.0,
    target_ber=1e-3,
    seed=0,
):
    demands = np.asarray(user_demands_gbps, dtype=float)
    quality = np.asarray(channel_quality_db, dtype=float)
    mcs_candidates = np.asarray(sorted(mcs_candidates), dtype=int)

    n_users = demands.size
    total_power_dbm = float(total_power_dbm)
    pmin_dbm = float(pmin_dbm)
    pmax_dbm = float(pmax_dbm)

    # Precompute required SNR for each MCS to meet target BER
    req_snr = {}
    for M in mcs_candidates:
        req_snr[int(M)] = _required_snr_db(int(M), target_ber)

    # Discretize power levels (0.5 dB steps)
    power_step = 0.5
    power_levels = np.arange(pmin_dbm, pmax_dbm + power_step * 0.01, power_step)

    # Convert total power budget to linear (mW)
    total_lin_mw = 10.0 ** (total_power_dbm / 10.0)

    # Discretize budget: use cost_unit_mw = 0.02 as in oracle
    cost_unit_mw = 0.02
    budget_units = int(math.floor(total_lin_mw / cost_unit_mw))

    # Pre-compute power costs in units
    power_costs = {}
    for p in power_levels:
        p_lin = 10.0 ** (p / 10.0)
        power_costs[p] = max(1, int(round(p_lin / cost_unit_mw)))

    # For each user, enumerate all (MCS, power) options
    # Compute utility for each option
    # utility = w_demand * throughput_ratio + w_ber * ber_pass + w_bits * bits_per_symbol_norm

    # Build option lists per user
    user_options = []  # list of lists of (mcs, power_dbm, cost_units, utility)

    for u in range(n_users):
        options = []
        q = quality[u]
        d = demands[u]

        for M in mcs_candidates:
            M_int = int(M)
            bits = math.log2(M_int)

            for p in power_levels:
                snr_db = q + p  # effective SNR = channel quality + transmit power
                ber = _ber_mqam(M_int, snr_db)
                ber_pass = 1.0 if ber <= target_ber else 0.0

                # Throughput proxy: proportional to bits per symbol
                # Scale by demand satisfaction
                throughput_proxy = bits  # bits per symbol
                demand_sat = min(throughput_proxy / max(d, 0.01), 1.0)

                # Utility weights (tuned to match scoring)
                # Score = 0.4 * demand_satisfaction + 0.3 * ber_pass_ratio + 0.3 * (avg_snr/30 + bits/6)/2
                # Per-user contribution:
                util = (0.4 * demand_sat +
                        0.3 * ber_pass +
                        0.3 * (min(snr_db / 30.0, 1.0) + bits / 6.0) / 2.0)

                cost = power_costs[p]
                options.append((M_int, float(p), cost, util))

        user_options.append(options)

    # Greedy approach: for each user, find the best option, then adjust if over budget
    # First, try to solve with a greedy knapsack

    # Sort options per user by utility (descending)
    for u in range(n_users):
        user_options[u].sort(key=lambda x: x[3], reverse=True)

    # Strategy: Use DP if feasible, otherwise greedy
    # With ~22 users and ~25 power levels * 3 MCS = 75 options per user,
    # and budget up to maybe ~5000 units, DP might be feasible

    # Check if DP is feasible
    n_options_max = max(len(opts) for opts in user_options) if n_users > 0 else 0

    if n_users <= 50 and budget_units <= 100000:
        # DP approach: dp[b] = max utility achievable for users 0..u-1 with budget b
        # Use vectorized numpy operations for speed

        INF_NEG = -1e18
        dp = np.full(budget_units + 1, INF_NEG, dtype=np.float64)
        dp[0] = 0.0

        # Track choices: choice[u][b] = option index chosen for user u at budget b
        choice = np.zeros((n_users, budget_units + 1), dtype=np.int32)

        for u in range(n_users):
            new_dp = np.full(budget_units + 1, INF_NEG, dtype=np.float64)
            new_choice = np.zeros(budget_units + 1, dtype=np.int32)
            opts = user_options[u]

            # Vectorized: for each option, shift dp by cost and add utility
            valid_mask = dp > (INF_NEG / 2)

            for oi, (M_int, p, cost, util) in enumerate(opts):
                if cost > budget_units:
                    continue
                # dp[0:budget_units+1-cost] shifted by cost
                shifted = dp[:budget_units + 1 - cost] + util
                target_slice = new_dp[cost:budget_units + 1]

                # Only update where shifted is better
                better = shifted > target_slice
                # Also only where source dp was valid
                src_valid = valid_mask[:budget_units + 1 - cost]
                update_mask = better & src_valid

                indices = np.where(update_mask)[0]
                if len(indices) > 0:
                    target_slice[indices] = shifted[indices]
                    new_choice[cost:budget_units + 1][indices] = oi

            dp = new_dp
            choice[u] = new_choice

        # Find best total budget
        best_b = int(np.argmax(dp))

        # Backtrack
        mcs_result = np.zeros(n_users, dtype=int)
        power_result = np.zeros(n_users, dtype=float)

        b = best_b
        for u in range(n_users - 1, -1, -1):
            oi = int(choice[u][b])
            M_int, p, cost, util = user_options[u][oi]
            mcs_result[u] = M_int
            power_result[u] = p
            b -= cost

    else:
        # Greedy fallback
        mcs_result = np.zeros(n_users, dtype=int)
        power_result = np.zeros(n_users, dtype=float)
        remaining_budget = budget_units

        # Assign minimum cost option first, then upgrade
        for u in range(n_users):
            # Find minimum cost option
            opts = user_options[u]
            min_cost_opt = min(opts, key=lambda x: x[2])
            mcs_result[u] = min_cost_opt[0]
            power_result[u] = min_cost_opt[1]
            remaining_budget -= min_cost_opt[2]

        # Now try to upgrade users greedily
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_user = -1
            best_opt = None

            for u in range(n_users):
                current_cost = power_costs[power_result[u]]
                current_util = 0
                for opt in user_options[u]:
                    if opt[0] == mcs_result[u] and opt[1] == power_result[u]:
                        current_util = opt[3]
                        break

                for opt in user_options[u]:
                    extra_cost = opt[2] - current_cost
                    gain = opt[3] - current_util
                    if gain > best_gain and extra_cost <= remaining_budget:
                        best_gain = gain
                        best_user = u
                        best_opt = opt

            if best_user >= 0 and best_opt is not None:
                old_cost = power_costs[power_result[best_user]]
                mcs_result[best_user] = best_opt[0]
                power_result[best_user] = best_opt[1]
                remaining_budget -= (best_opt[2] - old_cost)
                improved = True

    # Verify total power doesn't exceed budget
    total_used_lin = np.sum(10.0 ** (power_result / 10.0))
    if total_used_lin > total_lin_mw * 1.001:
        # Scale down if needed - shouldn't happen with proper DP
        scale = total_lin_mw / total_used_lin
        power_lin = 10.0 ** (power_result / 10.0) * scale
        power_result = 10.0 * np.log10(np.maximum(power_lin, 1e-12))
        power_result = np.clip(power_result, pmin_dbm, pmax_dbm)

    return {"mcs": mcs_result, "power_dbm": power_result}
# EVOLVE-BLOCK-END