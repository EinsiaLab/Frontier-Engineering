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

    # Demand-weighted power allocation (favors high-demand users)
    if np.sum(demands) > 0:
        weights = demands / np.sum(demands)
    else:
        weights = np.ones(n_users) / n_users

    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    power_lin = total_lin * weights
    power_dbm = 10.0 * np.log10(np.maximum(power_lin, 1e-12))
    power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)

    # MCS selection based on estimated SNR (quality + allocated power)
    # This balances BER reliability with spectral efficiency better than
    # quality-only thresholds.
    snr_est = quality + power_dbm
    mcs = np.full(n_users, int(mcs_candidates[0]), dtype=int)
    if np.any(mcs_candidates == 16):
        mcs[snr_est >= 15.0] = 16
    if np.any(mcs_candidates == 64):
        mcs[snr_est >= 22.0] = 64

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
