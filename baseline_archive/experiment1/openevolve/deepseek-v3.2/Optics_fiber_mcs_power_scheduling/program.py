# EVOLVE-BLOCK-START
"""Baseline solver for Task 2: MCS + power scheduling."""

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
    demands = np.asarray(user_demands_gbps, dtype=float)
    quality = np.asarray(channel_quality_db, dtype=float)
    mcs_candidates = np.asarray(mcs_candidates, dtype=int)

    n_users = demands.size

    # Conservative MCS selection to improve BER pass ratio, similar to top performing program
    mcs = np.full(n_users, int(mcs_candidates[0]), dtype=int)
    if np.any(mcs_candidates == 16):
        # Assign 16-QAM for good quality, but stricter than baseline
        mcs[quality >= 19.0] = 16
    if np.any(mcs_candidates == 64):
        # Assign 64-QAM only for excellent quality (very rare)
        mcs[quality >= 25.0] = 64

    # Power allocation: prioritize users with higher MCS and higher demand, and also help users with poor channel quality
    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    bits_per_symbol = np.log2(mcs)
    
    # Normalize demands and quality gaps
    max_demand = np.max(demands)
    max_quality = np.max(quality)
    # Weight combines: higher MCS needs more power, higher demand needs more throughput, poor quality needs help
    weights = bits_per_symbol * (demands / max_demand) * (max_quality - quality + 1)
    
    # Normalize weights
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(n_users) / n_users
    
    # Allocate linear power according to weights
    power_lin = total_lin * weights
    
    # Ensure minimum power per user, but make it proportional to MCS needs
    # Users with higher MCS need more minimum power to achieve basic SNR
    min_power_lin = total_lin * 0.005 * bits_per_symbol  # Scale minimum by bits_per_symbol
    power_lin = np.maximum(power_lin, min_power_lin)
    
    # If total exceeds budget due to minimum adjustments, scale down proportionally
    total_allocated = np.sum(power_lin)
    if total_allocated > total_lin:
        power_lin = power_lin * total_lin / total_allocated
    
    # Convert to dBm and clip
    power_dbm = 10.0 * np.log10(np.maximum(power_lin, 1e-12))
    power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)
    
    # Ensure total power constraint after clipping (like top performing program)
    power_lin_final = 10 ** (power_dbm / 10.0)
    total_used = np.sum(power_lin_final)
    if total_used > total_lin * 1.001:
        # Scale down to meet budget
        scale = total_lin / total_used
        power_lin_final = power_lin_final * scale
        power_dbm = 10.0 * np.log10(np.maximum(power_lin_final, 1e-12))
        power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
