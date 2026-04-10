# EVOLVE-BLOCK-START
"""Heuristic solver for Task 2: conservative MCS + weighted power."""

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
    d = np.asarray(user_demands_gbps, float)
    q = np.asarray(channel_quality_db, float)
    c = np.sort(np.asarray(mcs_candidates, int))
    n = d.size
    if n == 0:
        return {"mcs": c[:0], "power_dbm": q[:0]}

    lo = int(c[0])
    has16 = np.any(c == 16)
    has64 = np.any(c == 64)
    tot = 10 ** (float(total_power_dbm) / 10.0)

    m = np.full(n, lo, int)
    if has16:
        m[(q >= 18.0) & (d >= np.median(d))] = 16
    if has64:
        m[(q >= 24.0) & (d >= np.mean(d))] = 64

    w = np.maximum(d, 1e-6) / np.maximum(q, 1e-6)
    p = np.clip(10.0 * np.log10(np.maximum(tot * w / np.maximum(w.sum(), 1e-12), 1e-12)), pmin_dbm, pmax_dbm)

    lin = 10 ** (p / 10.0)
    s = lin.sum()
    if s > tot:
        p = 10.0 * np.log10(np.maximum(lin * (tot / s), 1e-12))
        p = np.clip(p, pmin_dbm, pmax_dbm)
    else:
        rem = tot - s
        free = (p > pmin_dbm + 1e-9) & (p < pmax_dbm - 1e-9)
        if rem > 1e-12 and np.any(free):
            p[free] = np.clip(
                10.0 * np.log10(10 ** (p[free] / 10.0) + rem / free.sum()),
                pmin_dbm,
                pmax_dbm,
            )

    e = q + p
    if has64 and has16:
        m[(m == 64) & (e < 24.0)] = 16
    if has16:
        m[(m == 16) & (e < 14.0)] = lo
    return {"mcs": m, "power_dbm": p}
# EVOLVE-BLOCK-END
