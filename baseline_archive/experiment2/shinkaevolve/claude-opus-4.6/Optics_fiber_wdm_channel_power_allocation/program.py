# EVOLVE-BLOCK-START
"""Solver for Task 1: WDM channel + power allocation.

Crosstalk-aware channel assignment with demand-weighted power allocation
and iterative local search refinement. Fully vectorised scoring for speed.
"""

from __future__ import annotations

import numpy as np


def _dbm_to_lin(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def _lin_to_dbm(lin):
    return 10.0 * np.log10(np.maximum(np.asarray(lin, dtype=float), 1e-30))


def _spread_channels(n_users, n_channels):
    """Select n_users channels from n_channels, maximally spread apart."""
    if n_users >= n_channels:
        return np.arange(n_channels, dtype=int)
    if n_users == 1:
        return np.array([n_channels // 2], dtype=int)
    indices = np.round(np.linspace(0, n_channels - 1, n_users)).astype(int)
    used = set()
    result = []
    for idx in indices:
        while idx in used:
            idx += 1
            if idx >= n_channels:
                idx = 0
        used.add(idx)
        result.append(idx)
    return np.array(sorted(result), dtype=int)


def _compute_snr_lin_vec(active_channels, channel_centers_hz, power_lin):
    """Compute SNR (linear) for each active channel, vectorised."""
    freqs = channel_centers_hz[active_channels]
    powers = power_lin[active_channels]
    n = len(active_channels)
    if n == 0:
        return np.array([])

    # Pairwise frequency differences
    df = np.abs(freqs[:, None] - freqs[None, :])
    df = np.maximum(df, 1e6)

    # Crosstalk: power * (50GHz / df)^2 * 0.01
    xtalk_matrix = powers[None, :] * (50e9 / df) ** 2 * 0.01
    np.fill_diagonal(xtalk_matrix, 0.0)
    xtalk = xtalk_matrix.sum(axis=1)

    noise_floor = 1e-4  # ASE noise proxy
    snr = powers / (noise_floor + xtalk)
    return snr


def _compute_score_vec(assignment, channel_centers_hz, power_lin, demands):
    """Vectorised proxy score: mean demand satisfaction. Higher is better."""
    active_mask = assignment >= 0
    active_users = np.where(active_mask)[0]
    n_active = len(active_users)
    if n_active == 0:
        return -1e9

    active_channels = assignment[active_users]
    active_demands = demands[active_users]

    snr = _compute_snr_lin_vec(active_channels, channel_centers_hz, power_lin)

    # Capacity proxy: log2(1 + SNR) * 32 GBaud
    capacity = np.log2(1.0 + snr) * 32.0
    demand_sat = np.minimum(capacity / np.maximum(active_demands, 1.0), 1.0)

    return demand_sat.mean()


def _allocate_power_waterfill(assignment, demands, channel_centers_hz,
                               n_channels, pmin_lin, pmax_lin, total_power_lin):
    """SNR-aware iterative power allocation.

    Strategy: iteratively increase power for the user with lowest demand satisfaction.
    """
    used_channels = np.unique(assignment[assignment >= 0])
    power_lin = np.full(n_channels, pmin_lin)

    if used_channels.size == 0:
        return power_lin

    # Demand per channel
    demand_per_channel = np.zeros(n_channels)
    for u in range(len(assignment)):
        ch = assignment[u]
        if ch >= 0:
            demand_per_channel[ch] += demands[u]

    n_inactive = n_channels - used_channels.size
    inactive_power = n_inactive * pmin_lin
    active_budget = max(total_power_lin - inactive_power, used_channels.size * pmin_lin)

    # Start with equal power for active channels
    equal_p = active_budget / used_channels.size
    equal_p = np.clip(equal_p, pmin_lin, pmax_lin)
    for ch in used_channels:
        power_lin[ch] = equal_p

    # Ensure we don't exceed budget
    current_total = power_lin.sum()
    if current_total > total_power_lin:
        scale = (total_power_lin - inactive_power) / (current_total - inactive_power + 1e-30)
        power_lin[used_channels] *= scale
        power_lin[used_channels] = np.clip(power_lin[used_channels], pmin_lin, pmax_lin)

    # Iterative: compute SNR-based capacity, boost lowest-satisfaction channels
    for iteration in range(50):
        snr = _compute_snr_lin_vec(used_channels, channel_centers_hz, power_lin)
        capacity = np.log2(1.0 + snr) * 32.0
        dem = demand_per_channel[used_channels]
        dem = np.maximum(dem, 1.0)
        sat = np.minimum(capacity / dem, 1.0)

        # Find the channel with lowest satisfaction that can still be boosted
        can_boost = power_lin[used_channels] < pmax_lin * 0.999
        if not np.any(can_boost):
            break

        remaining = total_power_lin - power_lin.sum()
        if remaining < pmin_lin * 0.001:
            break

        # Boost the least satisfied channel
        sat_masked = sat.copy()
        sat_masked[~can_boost] = 2.0  # exclude already maxed
        worst_idx = np.argmin(sat_masked)
        ch = used_channels[worst_idx]

        # Increase by a fraction of remaining budget
        step = min(remaining, (pmax_lin - power_lin[ch]) * 0.5, remaining * 0.3)
        if step < pmin_lin * 0.001:
            break
        power_lin[ch] += step

    # Final: fill any remaining budget into highest-demand unsaturated channels
    for _ in range(200):
        current_total = power_lin.sum()
        remaining = total_power_lin - current_total
        if remaining < pmin_lin * 0.001:
            break
        below_max = power_lin[used_channels] < pmax_lin * 0.999
        if not np.any(below_max):
            break
        candidates = used_channels[below_max]
        # Pick highest demand
        best_idx = candidates[np.argmax(demand_per_channel[candidates])]
        increase = min(remaining, pmax_lin - power_lin[best_idx])
        power_lin[best_idx] += increase

    # Ensure total constraint
    for _ in range(100):
        current_total = power_lin.sum()
        if current_total <= total_power_lin * 1.001:
            break
        excess = current_total - total_power_lin
        active_powers = power_lin[used_channels]
        max_idx = np.argmax(active_powers)
        reduction = min(excess, active_powers[max_idx] - pmin_lin)
        power_lin[used_channels[max_idx]] -= reduction

    return power_lin


def allocate_wdm(
    user_demands_gbps,
    channel_centers_hz,
    total_power_dbm,
    pmin_dbm=-8.0,
    pmax_dbm=3.0,
    target_ber=1e-3,
    seed=0,
):
    """Allocate one channel per user and per-channel power."""
    rng = np.random.RandomState(seed)
    demands = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)

    n_users = len(demands)
    n_channels = channel_centers_hz.size

    assignment = -np.ones(n_users, dtype=int)
    power_dbm_arr = np.full(n_channels, pmin_dbm, dtype=float)

    if n_users == 0:
        return {"assignment": assignment, "power_dbm": power_dbm_arr}

    pmin_lin = _dbm_to_lin(pmin_dbm)
    pmax_lin = _dbm_to_lin(pmax_dbm)
    total_power_lin = _dbm_to_lin(total_power_dbm)

    n_served = min(n_users, n_channels)

    # ---- Step 1: Select maximally spread channels ----
    selected_channels = _spread_channels(n_served, n_channels)

    # ---- Step 2: Assign users to channels ----
    # Sort users by demand (highest first)
    user_order = np.argsort(-demands)[:n_served]

    # Compute isolation score for each selected channel
    if len(selected_channels) > 1:
        freqs_sel = channel_centers_hz[selected_channels]
        isolation = np.full(len(selected_channels), np.inf)
        sorted_idx = np.argsort(freqs_sel)
        sorted_freqs = freqs_sel[sorted_idx]
        for i in range(len(sorted_idx)):
            min_dist = np.inf
            if i > 0:
                min_dist = min(min_dist, sorted_freqs[i] - sorted_freqs[i - 1])
            if i < len(sorted_idx) - 1:
                min_dist = min(min_dist, sorted_freqs[i + 1] - sorted_freqs[i])
            isolation[sorted_idx[i]] = min_dist
        ch_order = np.argsort(-isolation)
    else:
        ch_order = np.array([0])

    sorted_channels = selected_channels[ch_order]

    # Assign highest-demand users to most isolated channels
    for i, u in enumerate(user_order):
        assignment[u] = sorted_channels[i]

    # ---- Step 3: Initial power allocation ----
    power_lin = _allocate_power_waterfill(
        assignment, demands, channel_centers_hz,
        n_channels, pmin_lin, pmax_lin, total_power_lin
    )

    # ---- Step 4: Exhaustive pairwise swap search ----
    best_score = _compute_score_vec(assignment, channel_centers_hz, power_lin, demands)

    improved = True
    for _round in range(10):
        if not improved:
            break
        improved = False
        for i in range(n_served):
            for j in range(i + 1, n_served):
                u1 = user_order[i]
                u2 = user_order[j]
                if assignment[u1] < 0 or assignment[u2] < 0:
                    continue
                # Swap
                assignment[u1], assignment[u2] = assignment[u2], assignment[u1]
                s = _compute_score_vec(assignment, channel_centers_hz, power_lin, demands)
                if s > best_score + 1e-12:
                    best_score = s
                    improved = True
                else:
                    assignment[u1], assignment[u2] = assignment[u2], assignment[u1]

    # ---- Step 5: Try moving users to unused channels ----
    used_set = set(assignment[assignment >= 0].tolist())
    unused_channels = [c for c in range(n_channels) if c not in used_set]

    if len(unused_channels) > 0:
        for u in range(n_users):
            if assignment[u] < 0:
                continue
            orig_ch = assignment[u]
            best_ch = orig_ch
            best_s = best_score
            for uc in unused_channels:
                assignment[u] = uc
                s = _compute_score_vec(assignment, channel_centers_hz, power_lin, demands)
                if s > best_s + 1e-12:
                    best_s = s
                    best_ch = uc
            assignment[u] = orig_ch
            if best_ch != orig_ch:
                assignment[u] = best_ch
                used_set.discard(orig_ch)
                used_set.add(best_ch)
                unused_channels.remove(best_ch)
                unused_channels.append(orig_ch)
                best_score = best_s

    # ---- Step 6: Assign unserved users ----
    unserved = np.where(assignment < 0)[0]
    used_set = set(assignment[assignment >= 0].tolist())
    unused_channels_list = [c for c in range(n_channels) if c not in used_set]
    for i, u in enumerate(unserved):
        if i < len(unused_channels_list):
            assignment[u] = unused_channels_list[i]

    # ---- Step 7: Final power allocation ----
    power_lin = _allocate_power_waterfill(
        assignment, demands, channel_centers_hz,
        n_channels, pmin_lin, pmax_lin, total_power_lin
    )

    # ---- Step 8: Post-power local search (swaps + moves) ----
    best_score = _compute_score_vec(assignment, channel_centers_hz, power_lin, demands)
    active_users = np.where(assignment >= 0)[0]
    n_active = len(active_users)

    # Exhaustive pairwise swaps with final power
    improved = True
    for _round in range(5):
        if not improved:
            break
        improved = False
        for i in range(n_active):
            for j in range(i + 1, n_active):
                u1 = active_users[i]
                u2 = active_users[j]
                assignment[u1], assignment[u2] = assignment[u2], assignment[u1]
                s = _compute_score_vec(assignment, channel_centers_hz, power_lin, demands)
                if s > best_score + 1e-12:
                    best_score = s
                    improved = True
                else:
                    assignment[u1], assignment[u2] = assignment[u2], assignment[u1]

    # ---- Step 9: Final power re-optimization after swaps ----
    power_lin = _allocate_power_waterfill(
        assignment, demands, channel_centers_hz,
        n_channels, pmin_lin, pmax_lin, total_power_lin
    )

    # ---- Step 10: Try scipy power optimization if available ----
    try:
        from scipy.optimize import minimize

        used_channels = np.unique(assignment[assignment >= 0])
        if used_channels.size > 0:
            demand_per_channel = np.zeros(n_channels)
            for u in range(n_users):
                ch = assignment[u]
                if ch >= 0:
                    demand_per_channel[ch] += demands[u]

            n_used = used_channels.size
            n_inactive = n_channels - n_used
            inactive_power_total = n_inactive * pmin_lin

            def neg_score(x):
                """Negative of demand satisfaction for minimization."""
                pw = np.full(n_channels, pmin_lin)
                pw[used_channels] = x
                if pw.sum() > total_power_lin * 1.001:
                    return 1e6  # penalty
                snr = _compute_snr_lin_vec(used_channels, channel_centers_hz, pw)
                cap = np.log2(1.0 + snr) * 32.0
                dem = demand_per_channel[used_channels]
                dem = np.maximum(dem, 1.0)
                sat = np.minimum(cap / dem, 1.0)
                return -sat.mean()

            x0 = power_lin[used_channels].copy()
            bounds = [(pmin_lin, pmax_lin)] * n_used

            # Linear constraint: sum of active powers <= budget
            budget_active = total_power_lin - inactive_power_total
            from scipy.optimize import LinearConstraint
            A_eq = np.ones((1, n_used))
            lc = LinearConstraint(A_eq, lb=0, ub=budget_active)

            result = minimize(
                neg_score, x0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda x: budget_active - x.sum()},
                options={'maxiter': 200, 'ftol': 1e-10}
            )

            if result.success or result.fun < neg_score(x0):
                candidate_power = np.full(n_channels, pmin_lin)
                candidate_power[used_channels] = np.clip(result.x, pmin_lin, pmax_lin)
                if candidate_power.sum() <= total_power_lin * 1.001:
                    s_new = _compute_score_vec(assignment, channel_centers_hz, candidate_power, demands)
                    s_old = _compute_score_vec(assignment, channel_centers_hz, power_lin, demands)
                    if s_new > s_old:
                        power_lin = candidate_power
    except ImportError:
        pass

    # Convert to dBm
    power_dbm_arr = _lin_to_dbm(power_lin)
    power_dbm_arr = np.clip(power_dbm_arr, pmin_dbm, pmax_dbm)

    return {"assignment": assignment, "power_dbm": power_dbm_arr}
# EVOLVE-BLOCK-END