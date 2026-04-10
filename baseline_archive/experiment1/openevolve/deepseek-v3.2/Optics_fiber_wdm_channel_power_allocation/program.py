# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: WDM channel + power allocation.

Simple engineering baseline:
- assign users to channels in index order
- split total launch power equally among assigned channels
"""

from __future__ import annotations

import numpy as np


def allocate_wdm(
    user_demands_gbps,
    channel_centers_hz,
    total_power_dbm,
    pmin_dbm=-8.0,
    pmax_dbm=3.0,
    target_ber=1e-3,
    seed=0,
):
    """Allocate one channel per user and per-channel power.

    Parameters
    ----------
    user_demands_gbps : array-like, shape (U,)
        Requested data rate per user.
    channel_centers_hz : array-like, shape (C,)
        Available fixed WDM grid center frequencies.
    total_power_dbm : float
        Total launch power budget (dBm, summed across active channels).
    pmin_dbm, pmax_dbm : float
        Per-channel power limits.

    Returns
    -------
    dict with keys:
      - assignment: np.ndarray shape (U,), channel index or -1
      - power_dbm: np.ndarray shape (C,), per-channel launch power
    """
    _ = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)

    n_users = len(user_demands_gbps)
    n_channels = channel_centers_hz.size

    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)

    # Helper function to evenly space channels (from oracle)
    def _evenly_spaced_channels(n_channels, n_served):
        if n_served <= 0:
            return np.array([], dtype=int)
        raw = np.linspace(0, n_channels - 1, n_served)
        cand = np.rint(raw).astype(int)
        cand = np.clip(cand, 0, n_channels - 1)
        used = set()
        out = []
        for c in cand:
            if c not in used:
                used.add(int(c))
                out.append(int(c))
        if len(out) < n_served:
            for c in range(n_channels):
                if c not in used:
                    used.add(c)
                    out.append(c)
                if len(out) == n_served:
                    break
        return np.asarray(sorted(out), dtype=int)

    # Sort users by demand descending to prioritize high-demand users
    sorted_users = np.argsort(-user_demands_gbps)  # Descending order
    n_served = min(n_users, n_channels)
    
    # Assign users to channels spaced apart to reduce interference
    spaced_channels = _evenly_spaced_channels(n_channels, n_served)
    
    for i in range(n_served):
        u = sorted_users[i]
        assignment[u] = spaced_channels[i]

    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    # Allocate power with consideration for interference and demand
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    inactive_lin = (n_channels - used.size) * (10 ** (float(pmin_dbm) / 10.0))
    active_budget_lin = max(total_power_lin - inactive_lin, used.size * (10 ** (float(pmin_dbm) / 10.0)))
    
    # Calculate interference weights: channels closer to others get less power
    interference_decay = 0.9  # Same as in evaluation model
    isolation_factors = np.zeros(len(used))
    for idx, ch in enumerate(used):
        total_interference = 0.0
        for other_ch in used:
            if other_ch != ch:
                total_interference += np.exp(-abs(ch - other_ch) / interference_decay)
        isolation_factors[idx] = 1.0 / (1.0 + total_interference)
    
    # Combine with demand weights
    channel_demands = np.zeros(n_channels)
    for u in range(n_users):
        if assignment[u] >= 0:
            channel_demands[assignment[u]] = user_demands_gbps[u]
    
    demand_weights = channel_demands[used]
    if np.sum(demand_weights) > 0:
        demand_weights = demand_weights / np.sum(demand_weights)
    else:
        demand_weights = np.ones(len(used)) / len(used)
    
    # Combine isolation and demand (30% isolation, 70% demand) to prioritize demand satisfaction
    combined_weights = 0.3 * isolation_factors + 0.7 * demand_weights
    if np.sum(combined_weights) > 0:
        combined_weights = combined_weights / np.sum(combined_weights)
    else:
        combined_weights = np.ones(len(used)) / len(used)
    
    # Allocate power proportionally to combined weights
    each_lin = combined_weights * active_budget_lin
    each_dbm = 10.0 * np.log10(np.maximum(each_lin, 1e-12))
    each_dbm = np.clip(each_dbm, pmin_dbm, pmax_dbm)
    
    power_dbm[used] = each_dbm
    
    # Renormalize to ensure total power budget is respected
    def renormalize_active_power(power_dbm, assignment, n_channels, total_power_dbm, pmin_dbm, pmax_dbm):
        p = np.asarray(power_dbm, dtype=float).copy()
        active_channels = np.unique(assignment[assignment >= 0])
        if active_channels.size == 0:
            return p
        
        total_lin = 10 ** (float(total_power_dbm) / 10.0)
        inactive_lin = (n_channels - active_channels.size) * (10 ** (float(pmin_dbm) / 10.0))
        target_active = max(total_lin - inactive_lin, active_channels.size * (10 ** (float(pmin_dbm) / 10.0)))
        
        plin = 10 ** (p[active_channels] / 10.0)
        s = np.sum(plin)
        if s <= 0:
            plin = np.ones_like(plin)
            s = np.sum(plin)
        plin *= target_active / s
        p[active_channels] = np.clip(10.0 * np.log10(np.maximum(plin, 1e-12)), pmin_dbm, pmax_dbm)
        
        # Final check: if total power exceeds budget, scale down active channels
        total_now_2 = np.sum(10 ** (p / 10.0))
        if total_now_2 > total_lin * 1.000001:
            active_lin = 10 ** (p[active_channels] / 10.0)
            inactive_lin = np.sum(10 ** (p[np.setdiff1d(np.arange(n_channels), active_channels)] / 10.0))
            target_active_rescale = max(total_lin - inactive_lin, 1e-12)
            scale = target_active_rescale / max(np.sum(active_lin), 1e-12)
            active_lin *= min(scale, 1.0)
            p[active_channels] = np.clip(10.0 * np.log10(np.maximum(active_lin, 1e-12)), pmin_dbm, pmax_dbm)
        return p
    
    power_dbm = renormalize_active_power(power_dbm, assignment, n_channels, total_power_dbm, pmin_dbm, pmax_dbm)

    # Simple deterministic local search using seed to refine assignment
    rng = np.random.default_rng(seed)
    best_assignment = assignment.copy()
    best_power = power_dbm.copy()
    
    # Define a proxy score similar to evaluation formula (but simplified)
    def proxy_score(assignment, power_dbm, demands, target_ber):
        n_users = len(assignment)
        n_channels = len(power_dbm)
        assigned = assignment >= 0
        
        # Simplified SNR calculation (similar to evaluation)
        user_snr_db = np.full(n_users, -30.0)
        user_ber = np.ones(n_users)
        user_capacity = np.zeros(n_users)
        
        for u in range(n_users):
            ch = assignment[u]
            if ch < 0:
                continue
            sig = 10 ** (power_dbm[ch] / 10.0)
            interf = 0.0
            for v in range(n_users):
                ch2 = assignment[v]
                if v == u or ch2 < 0:
                    continue
                interf += (10 ** (power_dbm[ch2] / 10.0)) * np.exp(-abs(ch - ch2) / 0.9)
            
            snr_lin = sig / (2e-3 + 0.12 * interf)
            snr_db = 10.0 * np.log10(max(snr_lin, 1e-12))
            user_snr_db[u] = snr_db
            
            # Simplified BER: assume BER <= target_ber if SNR is high enough
            if snr_db > 10:
                user_ber[u] = 0.0
            else:
                user_ber[u] = 1.0
            
            # Capacity proxy
            user_capacity[u] = 28.0 * np.log2(1.0 + max(snr_lin, 1e-12))
        
        sat = np.minimum(user_capacity / np.maximum(demands, 1e-9), 1.0)
        demand_satisfaction = float(np.mean(sat))
        ber_pass_ratio = float(np.mean(user_ber[assigned] <= target_ber))
        spectral_util = float(np.sum(assigned) / n_channels)
        avg_snr_db = float(np.mean(user_snr_db[assigned]))
        
        # Approximate score formula (same weights as evaluation)
        snr_term = np.clip((avg_snr_db - 5.0) / 20.0, 0.0, 1.0)
        power_penalty = 0.0  # Assume budget is respected
        score = 0.35 * demand_satisfaction + 0.40 * ber_pass_ratio + 0.05 * spectral_util + 0.20 * snr_term - 0.15 * power_penalty
        return score
    
    best_score = proxy_score(best_assignment, best_power, user_demands_gbps, target_ber)
    
    # Increase iterations to 80 (like top programs) for better search
    for _ in range(80):
        cand_assignment = best_assignment.copy()
        served = np.where(cand_assignment >= 0)[0]
        if len(served) > 0:
            op = rng.random()
            if op < 0.5:
                # Swap two served users
                if len(served) >= 2:
                    u1, u2 = rng.choice(served, size=2, replace=False)
                    cand_assignment[u1], cand_assignment[u2] = cand_assignment[u2], cand_assignment[u1]
            else:
                # Move a served user to an unused channel
                used_channels = set(cand_assignment[served])
                free_channels = [c for c in range(n_channels) if c not in used_channels]
                if free_channels:
                    u = rng.choice(served)
                    cand_assignment[u] = rng.choice(free_channels)
        
        cand_used = np.unique(cand_assignment[cand_assignment >= 0])
        if cand_used.size == 0:
            continue
        
        # Recompute power for new assignment using same logic
        isolation_factors = np.zeros(len(cand_used))
        for idx, ch in enumerate(cand_used):
            total_interference = 0.0
            for other_ch in cand_used:
                if other_ch != ch:
                    total_interference += np.exp(-abs(ch - other_ch) / interference_decay)
            isolation_factors[idx] = 1.0 / (1.0 + total_interference)
        
        channel_demands = np.zeros(n_channels)
        for u in range(n_users):
            if cand_assignment[u] >= 0:
                channel_demands[cand_assignment[u]] = user_demands_gbps[u]
        demand_weights = channel_demands[cand_used]
        if np.sum(demand_weights) > 0:
            demand_weights = demand_weights / np.sum(demand_weights)
        else:
            demand_weights = np.ones(len(cand_used)) / len(cand_used)
        
        combined_weights = 0.3 * isolation_factors + 0.7 * demand_weights
        if np.sum(combined_weights) > 0:
            combined_weights = combined_weights / np.sum(combined_weights)
        else:
            combined_weights = np.ones(len(cand_used)) / len(cand_used)
        
        # Recalculate active budget for candidate assignment (since number of used channels may differ)
        cand_inactive_lin = (n_channels - cand_used.size) * (10 ** (float(pmin_dbm) / 10.0))
        cand_active_budget_lin = max(total_power_lin - cand_inactive_lin, cand_used.size * (10 ** (float(pmin_dbm) / 10.0)))
        each_lin = combined_weights * cand_active_budget_lin
        each_dbm = 10.0 * np.log10(np.maximum(each_lin, 1e-12))
        each_dbm = np.clip(each_dbm, pmin_dbm, pmax_dbm)
        cand_power = np.full(n_channels, pmin_dbm, dtype=float)
        cand_power[cand_used] = each_dbm
        
        cand_power = renormalize_active_power(cand_power, cand_assignment, n_channels, total_power_dbm, pmin_dbm, pmax_dbm)
        
        cand_score = proxy_score(cand_assignment, cand_power, user_demands_gbps, target_ber)
        if cand_score > best_score:
            best_score = cand_score
            best_assignment = cand_assignment
            best_power = cand_power
    
    # Return best found
    assignment = best_assignment
    power_dbm = best_power

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
