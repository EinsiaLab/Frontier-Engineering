# EVOLVE-BLOCK-START
"""Baseline solver for Task 2: MCS + power scheduling."""

from __future__ import annotations

import numpy as np
from math import erfc as _math_erfc, log2, sqrt


def _erfc_approx(x):
    """Compute erfc for scalar or array."""
    if np.isscalar(x):
        return _math_erfc(float(x))
    result = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        result[i] = _math_erfc(float(x[i]))
    return result


def _theory_ber_approx(M, EbN0_linear):
    """Approximate BER for M-QAM given Eb/N0 in linear scale."""
    if M == 2:
        return 0.5 * _erfc_approx(sqrt(EbN0_linear))
    k = log2(M)
    return (2.0 / k) * (1.0 - 1.0 / sqrt(M)) * _erfc_approx(sqrt(3.0 * k * EbN0_linear / (2.0 * (M - 1))))


def _ber_for_mcs_snr(M, snr_db):
    """Compute BER for given MCS (M-QAM) and SNR in dB."""
    k = log2(M)
    snr_lin = 10.0 ** (snr_db / 10.0)
    ebn0_lin = snr_lin / k
    return _theory_ber_approx(M, ebn0_lin)


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
    mcs_list = sorted(int(m) for m in mcs_candidates)

    n_users = demands.size
    if n_users == 0:
        return {"mcs": np.array([], dtype=int), "power_dbm": np.array([], dtype=float)}

    total_lin = 10.0 ** (float(total_power_dbm) / 10.0)
    pmin_lin = 10.0 ** (pmin_dbm / 10.0)
    pmax_lin = 10.0 ** (pmax_dbm / 10.0)

    # Discretize power levels (0.25 dB steps for finer granularity)
    power_levels_dbm = np.arange(pmin_dbm, pmax_dbm + 0.01, 0.25)
    power_levels_dbm = np.clip(power_levels_dbm, pmin_dbm, pmax_dbm)
    power_levels_lin = 10.0 ** (power_levels_dbm / 10.0)
    n_powers = len(power_levels_dbm)

    # Precompute per-user feasible options: (mcs, power_idx, throughput_bits, power_lin, ber)
    # We want to maximize a score that combines demand satisfaction, BER pass, and spectral efficiency
    # Score = 0.45*satisfaction + 0.40*ber_pass + 0.15*se_term
    # satisfaction = sum(min(achieved_throughput_i / demand_i, 1)) / n_users
    # ber_pass = fraction of users with ber <= target_ber
    # se_term = avg_bits_per_symbol / log2(max_mcs)

    max_mcs = max(mcs_list)
    max_bits = log2(max_mcs)

    # Build per-user option lists
    user_options = []  # list of lists of (mcs, pidx, bits_per_sym, power_lin, ber, demand_sat)
    for u in range(n_users):
        opts = []
        d = max(demands[u], 1e-12)
        for m in mcs_list:
            k = log2(m)
            for pi in range(n_powers):
                p_dbm = power_levels_dbm[pi]
                snr_db = quality[u] + p_dbm
                ber = _ber_for_mcs_snr(m, snr_db)
                ber_pass = 1.0 if ber <= target_ber else 0.0
                # throughput proxy: bits_per_symbol * (1 - ber) if ber passes, else penalized
                bits = k * (1.0 - ber)
                # demand satisfaction for this user (proxy: higher bits -> higher sat)
                # We approximate demand_sat = min(bits / some_reference, 1.0)
                # But actual demand_sat depends on normalization in the evaluator
                # Use a combined utility that matches the scoring formula
                # utility = 0.45 * demand_sat + 0.40 * ber_pass + 0.15 * (bits / max_bits)
                demand_sat = min(bits / max(d, 1e-9), 1.0) if d > 0 else 1.0
                utility = 0.45 * demand_sat + 0.40 * ber_pass + 0.15 * (bits / max_bits)
                opts.append((m, pi, bits, power_levels_lin[pi], ber, utility))
        if not opts:
            m0 = mcs_list[0]
            opts.append((m0, 0, log2(m0) * 0.5, pmin_lin, 0.5, 0.0))
        user_options.append(opts)

    # DP over users with discretized budget
    # Budget discretization
    n_budget_steps = 2000
    budget_step = total_lin / n_budget_steps

    # For each user, reduce options: for each distinct power bucket, keep only best utility
    user_reduced = []
    for u in range(n_users):
        by_bucket = {}
        for opt in user_options[u]:
            bucket = min(int(round(opt[3] / budget_step)), n_budget_steps)
            if bucket not in by_bucket or opt[5] > by_bucket[bucket][5]:
                by_bucket[bucket] = opt
            # Also keep options with same bucket but different trade-offs
        user_reduced.append(list(by_bucket.values()))

    # DP: dp[b] = (total_utility, choices_list)
    # Use arrays for speed
    dp_util = np.full(n_budget_steps + 1, -1e30, dtype=float)
    dp_util[0] = 0.0
    dp_choice = [None] * (n_budget_steps + 1)
    dp_choice[0] = []

    for u in range(n_users):
        new_util = np.full(n_budget_steps + 1, -1e30, dtype=float)
        new_choice = [None] * (n_budget_steps + 1)
        opts = user_reduced[u]
        for b in range(n_budget_steps + 1):
            if dp_util[b] < -1e29:
                continue
            for opt in opts:
                cost = min(int(round(opt[3] / budget_step)), n_budget_steps)
                nb = b + cost
                if nb > n_budget_steps:
                    continue
                val = dp_util[b] + opt[5]
                if val > new_util[nb]:
                    new_util[nb] = val
                    new_choice[nb] = (b, opt)
        dp_util = new_util
        dp_choice[u] = new_choice  # store per user

    # Actually we need to reconstruct. Let me redo DP with backtracking.
    # Simpler: store per-user choices via backtracking arrays
    # Re-implement with proper backtracking
    prev_dp = np.full(n_budget_steps + 1, -1e30)
    prev_dp[0] = 0.0
    back = []  # back[u][b] = (prev_b, opt)

    for u in range(n_users):
        cur_dp = np.full(n_budget_steps + 1, -1e30)
        cur_back = [None] * (n_budget_steps + 1)
        opts = user_reduced[u]
        for opt in opts:
            cost = min(int(round(opt[3] / budget_step)), n_budget_steps)
            for b in range(n_budget_steps + 1 - cost):
                if prev_dp[b] < -1e29:
                    continue
                nb = b + cost
                val = prev_dp[b] + opt[5]
                if val > cur_dp[nb]:
                    cur_dp[nb] = val
                    cur_back[nb] = (b, opt)
        back.append(cur_back)
        prev_dp = cur_dp

    # Find best final budget
    best_b = int(np.argmax(prev_dp))

    # Backtrack
    mcs_out = np.full(n_users, mcs_list[0], dtype=int)
    power_out_dbm = np.full(n_users, pmin_dbm, dtype=float)

    b = best_b
    for u in range(n_users - 1, -1, -1):
        if back[u][b] is not None:
            prev_b, opt = back[u][b]
            mcs_out[u] = opt[0]
            power_out_dbm[u] = power_levels_dbm[opt[1]]
            b = prev_b
        else:
            mcs_out[u] = mcs_list[0]
            power_out_dbm[u] = pmin_dbm

    power_out_dbm = np.clip(power_out_dbm, pmin_dbm, pmax_dbm)
    return {"mcs": mcs_out, "power_dbm": power_out_dbm}
# EVOLVE-BLOCK-END
