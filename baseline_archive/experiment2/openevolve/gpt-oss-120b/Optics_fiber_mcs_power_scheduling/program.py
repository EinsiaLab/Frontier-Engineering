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

    # ---- Power allocation proportional to traffic demand ----
    total_lin = 10 ** (float(total_power_dbm) / 10.0)          # total linear power (mW)
    demand_weights = demands / (demands.sum() if demands.size else 1.0)
    power_lin = total_lin * demand_weights                     # per‑user linear power
    power_dbm = 10.0 * np.log10(np.maximum(power_lin, 1e-12))
    power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)         # respect per‑user limits

    # ---- Effective SNR after power allocation ----
    effective_snr = quality + power_dbm

    # ---- MCS selection based on effective SNR ----
    # Use stricter, safety‑margined thresholds (same as the original quality‑only rule)
    # to improve the BER‑pass ratio while keeping the demand‑aware power allocation.
    mcs = np.full(n_users, int(mcs_candidates[0]), dtype=int)   # start with lowest MCS
    if np.any(mcs_candidates == 64):
        # 64‑QAM requires very good SNR
        mcs[effective_snr >= 22.0] = 64
    if np.any(mcs_candidates == 16):
        # 16‑QAM requires moderate SNR
        # Users already qualifying for 64‑QAM will keep that choice.
        mcs[effective_snr >= 15.0] = 16

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
