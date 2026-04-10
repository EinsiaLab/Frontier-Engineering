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
    d, q = np.asarray(user_demands_gbps), np.asarray(channel_quality_db)
    mcs = np.full(d.size, mcs_candidates[0], dtype=int)
    if 16 in mcs_candidates: mcs[q >= 19.5] = 16
    if 64 in mcs_candidates: mcs[q >= 26.5] = 64
    p = 10 ** (total_power_dbm / 10.0) * d / max(d.sum(), 1e-12)
    return {"mcs": mcs, "power_dbm": np.clip(10.0 * np.log10(p + 1e-12), pmin_dbm, pmax_dbm)}
# EVOLVE-BLOCK-END
