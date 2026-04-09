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

    # Adjusted MCS thresholds to improve BER compliance
    mcs = np.full(n_users, int(mcs_candidates[0]), dtype=int)
    if np.any(mcs_candidates == 16):
        mcs[quality >= 19.5] = 16  # Conservative threshold reduces number of 16QAM users, each gets sufficient power for BER compliance
    if np.any(mcs_candidates == 64):
        mcs[quality >= 24.0] = 64

    # Channel-aware weighted power split: allocate more power to worse channels to improve BER and throughput
    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    # Use inverse of linear channel gain + MCS factor as weight: higher MCS needs more power for BER compliance
    channel_gain_lin = 10 ** (quality / 10.0) + 1e-12
    mcs_factor = (np.log2(mcs)) ** 2.2 + 1e-9  # Heavily prioritize higher MCS users to get enough power for BER < target
    weights = (1.0 / channel_gain_lin) * mcs_factor
    weights /= weights.sum() + 1e-12  # Normalize weights sum to 1
    
    # Calculate initial power values
    power_lin = total_lin * weights
    power_dbm = 10.0 * np.log10(power_lin + 1e-12)
    power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)
    
    # Readjust to ensure total power stays within budget after clipping
    total_used_lin = (10 ** (power_dbm / 10.0)).sum()
    if total_used_lin > total_lin + 1e-9:
        scale_factor = total_lin / total_used_lin
        power_lin_scaled = scale_factor * (10 ** (power_dbm / 10.0))
        power_dbm = 10.0 * np.log10(power_lin_scaled + 1e-12)
        power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
