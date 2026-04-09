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
    if n_users == 0:
        return {"mcs": np.array([], dtype=int), "power_dbm": np.array([], dtype=float)}

    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    
    # Conservative SNR thresholds for target BER ~7e-4
    snr_req = {4: 10.5, 16: 17.0, 64: 23.0}
    
    # Phase 1: Conservative MCS selection - only MCS=16 for excellent channels
    mcs = np.full(n_users, int(mcs_candidates[0]), dtype=int)
    if 16 in mcs_candidates:
        mcs[quality >= 19.5] = 16
    
    # Phase 2: Allocate power to achieve required SNR for BER compliance
    power_needed_dbm = np.zeros(n_users)
    for u in range(n_users):
        mc = int(mcs[u])
        req_snr = snr_req.get(mc, 10.5)
        needed = req_snr - quality[u]
        power_needed_dbm[u] = np.clip(needed, pmin_dbm, pmax_dbm)
    
    power_lin_needed = 10 ** (power_needed_dbm / 10.0)
    total_needed = power_lin_needed.sum()
    
    if total_needed <= total_lin:
        power_dbm = power_needed_dbm.copy()
        remaining = total_lin - total_needed
        if remaining > 0:
            for u in np.argsort(-quality):
                extra_lin = remaining * 0.3
                new_lin = 10 ** (power_dbm[u] / 10.0) + extra_lin
                new_dbm = 10.0 * np.log10(max(new_lin, 1e-12))
                power_dbm[u] = np.clip(new_dbm, pmin_dbm, pmax_dbm)
                remaining -= extra_lin
                if remaining <= 0:
                    break
    else:
        scale = total_lin / total_needed
        power_lin = power_lin_needed * scale
        power_dbm = 10.0 * np.log10(np.maximum(power_lin, 1e-12))
        power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
