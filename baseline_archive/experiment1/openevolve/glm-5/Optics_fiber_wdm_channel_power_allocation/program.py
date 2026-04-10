# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: WDM channel + power allocation.

Enhanced strategy with local search refinement:
- Interference-aware initial channel assignment
- Demand-proportional power allocation
- Local search to improve assignment and power
"""

from __future__ import annotations

import numpy as np


def _compute_snr(assignment, power_dbm, n_users, noise_floor=2e-3, int_scale=0.12, int_decay=0.9):
    """Compute per-user SNR based on interference model."""
    snr = np.zeros(n_users)
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
            interf += (10 ** (power_dbm[ch2] / 10.0)) * np.exp(-abs(ch - ch2) / int_decay)
        snr[u] = sig / (noise_floor + int_scale * interf)
    return snr


def _compute_objective(assignment, power_dbm, demands, target_ber=1e-3):
    """Compute proxy objective for optimization."""
    n_users = len(assignment)
    assigned = assignment >= 0
    if not np.any(assigned):
        return -1e9
    
    snr = _compute_snr(assignment, power_dbm, n_users)
    snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
    
    capacity = 28.0 * np.log2(1.0 + np.maximum(snr, 1e-12))
    sat = np.minimum(capacity / np.maximum(demands, 1e-9), 1.0)
    demand_sat = np.mean(sat)
    
    M = 4
    ebn0 = snr_db - 10.0 * np.log10(np.log2(M))
    from optic.comm.metrics import theoryBER
    ber_pass = np.mean(np.array([float(theoryBER(M, e, "qam")) for e in ebn0[assigned]]) <= target_ber)
    
    spectral = np.sum(assigned) / max(len(power_dbm), 1)
    avg_snr = np.mean(snr_db[assigned])
    snr_term = np.clip((avg_snr - 5.0) / 20.0, 0.0, 1.0)
    
    return 0.35 * demand_sat + 0.40 * ber_pass + 0.05 * spectral + 0.20 * snr_term


def _allocate_power(assignment, demands, n_channels, total_power_dbm, pmin_dbm, pmax_dbm):
    """Allocate power with floor + proportional split for better demand satisfaction."""
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)
    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return power_dbm
    
    served_demands = demands[assignment >= 0]
    total_demand = np.sum(served_demands)
    
    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    inactive_lin = (n_channels - used.size) * (10 ** (float(pmin_dbm) / 10.0))
    active_budget = max(total_lin - inactive_lin, used.size * (10 ** (float(pmin_dbm) / 10.0)))
    
    if total_demand > 0:
        # Hybrid: 35% equal floor + 65% demand-proportional improves demand satisfaction
        floor_frac = 0.35
        floor_power = active_budget * floor_frac / used.size
        prop_budget = active_budget * (1 - floor_frac)
        for i, ch in enumerate(used):
            frac = served_demands[i] / total_demand
            p_lin = floor_power + prop_budget * frac
            power_dbm[ch] = float(np.clip(10.0 * np.log10(max(p_lin, 1e-12)), pmin_dbm, pmax_dbm))
    return power_dbm


def allocate_wdm(
    user_demands_gbps,
    channel_centers_hz,
    total_power_dbm,
    pmin_dbm=-8.0,
    pmax_dbm=3.0,
    target_ber=1e-3,
    seed=0,
):
    """Allocate one channel per user and per-channel power with local search."""
    demands = np.asarray(user_demands_gbps, dtype=float)
    n_users = len(demands)
    n_channels = len(channel_centers_hz)
    rng = np.random.default_rng(seed)
    
    # Initial interference-aware assignment
    n_served = min(n_users, n_channels)
    spaced = np.linspace(0, n_channels - 1, n_served).astype(int)
    sorted_users = np.argsort(-demands)
    
    assignment = -np.ones(n_users, dtype=int)
    for i, u in enumerate(sorted_users[:n_served]):
        assignment[u] = spaced[i]
    
    power_dbm = _allocate_power(assignment, demands, n_channels, total_power_dbm, pmin_dbm, pmax_dbm)
    best_obj = _compute_objective(assignment, power_dbm, demands, target_ber)
    best_assignment = assignment.copy()
    best_power = power_dbm.copy()
    
    # Simulated annealing-style local search with temperature decay
    temperature = 0.01
    for iteration in range(60):
        cand_assignment = best_assignment.copy()
        users_served = np.where(cand_assignment >= 0)[0]
        if users_served.size < 2:
            break
        
        # Focus on high-demand users for swaps
        if rng.random() < 0.6:
            # Swap: prefer swapping high-demand user with others
            demands_served = demands[users_served]
            probs = demands_served / demands_served.sum()
            u1 = rng.choice(users_served, p=probs)
            u2 = rng.choice(users_served[users_served != u1])
            cand_assignment[u1], cand_assignment[u2] = cand_assignment[u2], cand_assignment[u1]
        else:
            # Move user to free channel
            u = rng.choice(users_served)
            used_ch = set(cand_assignment[users_served])
            free_ch = [c for c in range(n_channels) if c not in used_ch]
            if free_ch:
                cand_assignment[u] = rng.choice(free_ch)
        
        cand_power = _allocate_power(cand_assignment, demands, n_channels, total_power_dbm, pmin_dbm, pmax_dbm)
        cand_obj = _compute_objective(cand_assignment, cand_power, demands, target_ber)
        
        # Accept if better or with small probability for exploration
        if cand_obj > best_obj or (rng.random() < temperature and cand_obj > best_obj - 0.01):
            best_obj = cand_obj
            best_assignment = cand_assignment
            best_power = cand_power
        
        temperature *= 0.95  # Decay temperature
    
    return {"assignment": best_assignment, "power_dbm": best_power}
# EVOLVE-BLOCK-END
