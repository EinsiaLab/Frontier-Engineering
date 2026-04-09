# EVOLVE-BLOCK-START
"""Improved solver for Task 2: MCS + power scheduling with adaptive thresholds."""

from __future__ import annotations

import numpy as np


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
    """Adaptive MCS selection with reliability-aware power allocation."""
    demands = np.asarray(user_demands_gbps, dtype=float)
    quality = np.asarray(channel_quality_db, dtype=float)
    mcs_candidates = np.asarray(mcs_candidates, dtype=int)

    n_users = demands.size
    if n_users == 0:
        return {"mcs": np.array([], dtype=int), "power_dbm": np.array([], dtype=float)}

    # SNR thresholds with reliability margin for target BER ~1e-3
    snr_thresholds = {4: 10.5, 16: 16.5, 64: 23.0}

    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    pmin_lin = 10 ** (pmin_dbm / 10.0)
    pmax_lin = 10 ** (pmax_dbm / 10.0)

    # Calculate minimum power needed for each user at each MCS level
    min_power_lin = {}
    for mc in mcs_candidates:
        power_db = np.clip(snr_thresholds[mc] - quality, pmin_dbm, pmax_dbm)
        min_power_lin[mc] = 10 ** (power_db / 10.0)

    sorted_mcs = sorted(mcs_candidates)
    
    # Start with minimum MCS for all users
    mcs = np.full(n_users, sorted_mcs[0], dtype=int)
    power_lin = min_power_lin[sorted_mcs[0]].copy()

    # Scale down if minimum power exceeds budget
    min_total = np.sum(power_lin)
    if min_total > total_lin:
        scale = total_lin / min_total
        power_lin = power_lin * scale

    current_total = np.sum(power_lin)
    
    # Phase 1: Collect all possible upgrades with channel-quality-weighted benefits
    all_upgrades = []
    for u in range(n_users):
        for m_idx in range(1, len(sorted_mcs)):
            target_mc = sorted_mcs[m_idx]
            for prev_idx in range(m_idx):
                prev_mc = sorted_mcs[prev_idx]
                cost = min_power_lin[target_mc][u] - min_power_lin[prev_mc][u]
                if cost < 0:
                    cost = 0
                # Throughput gain
                throughput_gain = np.log2(target_mc) - np.log2(prev_mc)
                # Weight benefit by demand and channel quality (better channels = more efficient)
                quality_factor = 1.0 + 0.05 * quality[u]  # Quality bonus
                benefit = demands[u] * throughput_gain * quality_factor
                efficiency = benefit / max(cost, 1e-15)
                all_upgrades.append((u, prev_mc, target_mc, cost, benefit, efficiency))
    
    # Sort by efficiency (benefit per unit power)
    all_upgrades.sort(key=lambda x: x[5], reverse=True)
    
    # Phase 2: Apply upgrades greedily
    applied = set()
    for u, prev_mc, target_mc, cost, benefit, efficiency in all_upgrades:
        key = (u, prev_mc, target_mc)
        if key in applied:
            continue
        if mcs[u] == prev_mc and current_total + cost <= total_lin:
            current_total += cost
            power_lin[u] = min_power_lin[target_mc][u]
            mcs[u] = target_mc
            applied.add(key)

    # Phase 3: Try additional single-step upgrades
    for _ in range(5):
        best_upgrade = None
        best_efficiency = 0
        
        for u in range(n_users):
            current_mc_idx = list(sorted_mcs).index(mcs[u])
            if current_mc_idx >= len(sorted_mcs) - 1:
                continue
            target_mc = sorted_mcs[current_mc_idx + 1]
            cost = min_power_lin[target_mc][u] - power_lin[u]
            if cost < 0:
                cost = 0
            if current_total + cost > total_lin:
                continue
            throughput_gain = np.log2(target_mc) - np.log2(mcs[u])
            quality_factor = 1.0 + 0.05 * quality[u]
            benefit = demands[u] * throughput_gain * quality_factor
            efficiency = benefit / max(cost, 1e-15)
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_upgrade = (u, target_mc, cost)
        
        if best_upgrade is None:
            break
        u, target_mc, cost = best_upgrade
        current_total += cost
        power_lin[u] = min_power_lin[target_mc][u]
        mcs[u] = target_mc

    # Phase 4: Redistribute surplus power with SNR-margin awareness
    surplus = total_lin - np.sum(power_lin)
    if surplus > 0:
        # Calculate SNR margin for each user
        snr_margins = np.array([
            quality[u] + 10 * np.log10(max(power_lin[u], 1e-15)) - snr_thresholds[mcs[u]]
            for u in range(n_users)
        ])
        
        # Users with low margins need more power; weight by demand too
        margin_needs = np.maximum(0, 3.0 - snr_margins)
        combined_weights = demands * (1.0 + margin_needs)
        
        if np.sum(combined_weights) > 0:
            combined_weights = combined_weights / np.sum(combined_weights)
        else:
            combined_weights = np.ones(n_users) / n_users
            
        extra_power = surplus * combined_weights
        power_lin = power_lin + extra_power

    # Clip power to allowed range
    power_lin = np.clip(power_lin, pmin_lin, pmax_lin)

    # Convert power back to dBm
    power_dbm = 10 * np.log10(np.maximum(power_lin, 1e-12))

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END