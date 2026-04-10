# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: WDM channel + power allocation.

Advanced solver with interference-aware channel assignment,
demand-weighted power allocation, and local search optimization.
"""

from __future__ import annotations

import numpy as np


def _dbm_to_lin(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def _lin_to_dbm(lin):
    return 10.0 * np.log10(np.maximum(lin, 1e-30))


def _estimate_snr_db(power_dbm_arr, channel_centers_hz, assignment, user_idx):
    """Estimate SNR for a user on its assigned channel.
    
    Simple model: SNR depends on signal power and interference from adjacent channels.
    Uses a simplified Gaussian noise + XPM/FWM interference model.
    """
    ch = assignment[user_idx]
    if ch < 0:
        return -np.inf
    
    sig_power = _dbm_to_lin(power_dbm_arr[ch])
    
    # Noise floor (ASE-like) - approximate
    noise_lin = _dbm_to_lin(-25.0)  # noise floor
    
    # Interference from other active channels (simplified NLI model)
    interference = 0.0
    freq_ch = channel_centers_hz[ch]
    active_channels = set(assignment[assignment >= 0])
    
    for other_ch in active_channels:
        if other_ch == ch:
            continue
        other_power = _dbm_to_lin(power_dbm_arr[other_ch])
        freq_diff = abs(channel_centers_hz[other_ch] - freq_ch)
        # XPM-like interference decreases with frequency separation
        if freq_diff > 0:
            interference += other_power ** 2 * sig_power / (freq_diff / 50e9) ** 2 * 1e-3
    
    snr_lin = sig_power / (noise_lin + interference)
    return 10.0 * np.log10(max(snr_lin, 1e-30))


def _capacity_from_snr(snr_db):
    """Shannon-like capacity estimate in Gbps (simplified)."""
    if snr_db < 0:
        return 0.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    # Assume ~32 GBaud symbol rate, dual-pol
    baud = 32e9
    capacity_bps = 2 * baud * np.log2(1 + snr_lin)
    return capacity_bps / 1e9


def _ber_from_snr(snr_db, modulation_order=4):
    """Approximate BER for QPSK-like modulation."""
    if snr_db < 0:
        return 0.5
    snr_lin = 10.0 ** (snr_db / 10.0)
    # Approximate Q-function based BER for QPSK
    q = np.sqrt(snr_lin)
    ber = 0.5 * np.exp(-q * q / 2.0) / max(q, 0.01) * 0.4
    return min(max(ber, 1e-15), 0.5)


def _compute_score(assignment, power_dbm, channel_centers_hz, user_demands, 
                   total_power_dbm, pmin_dbm, pmax_dbm, target_ber):
    """Approximate the scoring function used by verification."""
    n_users = len(user_demands)
    n_channels = len(channel_centers_hz)
    
    # Demand satisfaction
    total_demand = 0.0
    total_achieved = 0.0
    ber_pass = 0
    served = 0
    
    for u in range(n_users):
        ch = assignment[u]
        if ch < 0:
            continue
        served += 1
        snr = _estimate_snr_db(power_dbm, channel_centers_hz, assignment, u)
        cap = _capacity_from_snr(snr)
        total_demand += user_demands[u]
        total_achieved += min(cap, user_demands[u])
        ber = _ber_from_snr(snr)
        if ber <= target_ber:
            ber_pass += 1
    
    demand_sat = total_achieved / max(total_demand, 1e-9) if total_demand > 0 else 0
    ber_ratio = ber_pass / max(served, 1) if served > 0 else 0
    
    active_channels = len(set(assignment[assignment >= 0]))
    spectral_util = active_channels / n_channels if n_channels > 0 else 0
    
    # Average SNR
    snr_sum = 0.0
    snr_count = 0
    for u in range(n_users):
        if assignment[u] >= 0:
            snr = _estimate_snr_db(power_dbm, channel_centers_hz, assignment, u)
            snr_sum += snr
            snr_count += 1
    avg_snr = snr_sum / max(snr_count, 1)
    snr_term = min(avg_snr / 20.0, 1.0)
    
    # Power penalty
    total_lin = np.sum(_dbm_to_lin(power_dbm))
    budget_lin = _dbm_to_lin(total_power_dbm)
    power_penalty = max(0, (total_lin - budget_lin) / budget_lin)
    
    score = 0.35 * demand_sat + 0.30 * ber_ratio + 0.15 * spectral_util + 0.20 * snr_term - 0.15 * power_penalty
    return score, demand_sat, ber_ratio, spectral_util


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
    rng = np.random.RandomState(seed)
    user_demands = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)

    n_users = len(user_demands)
    n_channels = channel_centers_hz.size

    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)

    if n_users == 0 or n_channels == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    total_power_lin = _dbm_to_lin(total_power_dbm)
    pmin_lin = _dbm_to_lin(pmin_dbm)
    pmax_lin = _dbm_to_lin(pmax_dbm)

    n_served = min(n_users, n_channels)

    # Sort users by demand (highest first) for priority assignment
    user_order = np.argsort(-user_demands)

    # Spread channels: assign users to evenly spaced channels to minimize interference
    # Sort channels by frequency
    ch_sorted = np.argsort(channel_centers_hz)
    
    if n_served <= n_channels:
        # Pick evenly spaced channels
        indices = np.linspace(0, n_channels - 1, n_served, dtype=int)
        # But ensure unique
        selected_channels = []
        used_set = set()
        for idx in indices:
            ch = ch_sorted[idx]
            if ch in used_set:
                # Find nearest unused
                for offset in range(1, n_channels):
                    for candidate in [idx + offset, idx - offset]:
                        if 0 <= candidate < n_channels and ch_sorted[candidate] not in used_set:
                            ch = ch_sorted[candidate]
                            break
                    if ch not in used_set:
                        break
            selected_channels.append(ch)
            used_set.add(ch)
    else:
        selected_channels = list(ch_sorted[:n_served])

    # Assign highest-demand users to center channels (better SNR typically)
    # Actually, assign them spread out to minimize mutual interference
    # Map: user with highest demand -> channel with most isolation
    # For simplicity, just assign in spread order
    for i, u in enumerate(user_order[:n_served]):
        assignment[u] = selected_channels[i]

    # Demand-weighted power allocation
    used_channels = np.unique(assignment[assignment >= 0])
    if used_channels.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    # Calculate demand per channel
    ch_demand = np.zeros(n_channels)
    for u in range(n_users):
        ch = assignment[u]
        if ch >= 0:
            ch_demand[ch] += user_demands[u]

    # Inactive channels get pmin
    inactive_mask = np.ones(n_channels, dtype=bool)
    inactive_mask[used_channels] = False
    inactive_power_lin = np.sum(pmin_lin * inactive_mask)
    
    active_budget_lin = max(total_power_lin - inactive_power_lin, 
                           used_channels.size * pmin_lin)

    # Demand-weighted power distribution
    total_ch_demand = ch_demand[used_channels].sum()
    if total_ch_demand > 0:
        weights = ch_demand[used_channels] / total_ch_demand
    else:
        weights = np.ones(used_channels.size) / used_channels.size

    # Allocate power proportional to demand
    active_power_lin = weights * active_budget_lin
    
    # Clip to per-channel limits
    active_power_lin = np.clip(active_power_lin, pmin_lin, pmax_lin)
    
    # Rescale to fit budget
    current_total = active_power_lin.sum() + inactive_power_lin
    if current_total > total_power_lin:
        excess = current_total - total_power_lin
        # Reduce proportionally from active channels
        reducible = active_power_lin - pmin_lin
        total_reducible = reducible.sum()
        if total_reducible > 0:
            reduction = np.minimum(reducible, reducible / total_reducible * excess)
            active_power_lin -= reduction

    power_dbm[used_channels] = _lin_to_dbm(active_power_lin)
    power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)

    # Local search: try swapping channel assignments to improve score
    best_assignment = assignment.copy()
    best_power = power_dbm.copy()
    best_score, _, _, _ = _compute_score(best_assignment, best_power, channel_centers_hz,
                                          user_demands, total_power_dbm, pmin_dbm, pmax_dbm, target_ber)

    # Try power optimization: iteratively adjust powers
    for iteration in range(50):
        improved = False
        
        # Try adjusting each active channel's power
        for ch in used_channels:
            current_p = best_power[ch]
            for delta in [1.0, 0.5, 0.2, -0.2, -0.5, -1.0]:
                trial_power = best_power.copy()
                new_p = np.clip(current_p + delta, pmin_dbm, pmax_dbm)
                trial_power[ch] = new_p
                
                # Check budget
                trial_total = np.sum(_dbm_to_lin(trial_power))
                if trial_total > total_power_lin * 1.01:
                    continue
                
                score, _, _, _ = _compute_score(best_assignment, trial_power, channel_centers_hz,
                                                 user_demands, total_power_dbm, pmin_dbm, pmax_dbm, target_ber)
                if score > best_score:
                    best_score = score
                    best_power = trial_power
                    improved = True
                    break
        
        if not improved:
            break

    # Try channel swaps for served users
    for iteration in range(30):
        improved = False
        for i in range(n_served):
            u = user_order[i]
            current_ch = best_assignment[u]
            if current_ch < 0:
                continue
            
            # Try unused channels
            all_used = set(best_assignment[best_assignment >= 0])
            unused = [c for c in range(n_channels) if c not in all_used]
            
            for new_ch in unused[:5]:  # Try a few unused channels
                trial_assignment = best_assignment.copy()
                trial_assignment[u] = new_ch
                trial_power = best_power.copy()
                trial_power[new_ch] = best_power[current_ch]
                trial_power[current_ch] = pmin_dbm
                
                score, _, _, _ = _compute_score(trial_assignment, trial_power, channel_centers_hz,
                                                 user_demands, total_power_dbm, pmin_dbm, pmax_dbm, target_ber)
                if score > best_score:
                    best_score = score
                    best_assignment = trial_assignment
                    best_power = trial_power
                    improved = True
                    break
        
        if not improved:
            break

    # Try to serve unserved users if channels are available
    unserved = np.where(best_assignment < 0)[0]
    if len(unserved) > 0:
        all_used = set(best_assignment[best_assignment >= 0])
        unused_chs = [c for c in range(n_channels) if c not in all_used]
        
        for u, ch in zip(unserved, unused_chs):
            trial_assignment = best_assignment.copy()
            trial_assignment[u] = ch
            trial_power = best_power.copy()
            # Give minimum power to new channel
            trial_power[ch] = pmin_dbm
            
            # Check budget
            if np.sum(_dbm_to_lin(trial_power)) <= total_power_lin * 1.01:
                score, _, _, _ = _compute_score(trial_assignment, trial_power, channel_centers_hz,
                                                 user_demands, total_power_dbm, pmin_dbm, pmax_dbm, target_ber)
                if score > best_score:
                    best_score = score
                    best_assignment = trial_assignment
                    best_power = trial_power

    return {"assignment": best_assignment, "power_dbm": best_power}
# EVOLVE-BLOCK-END
