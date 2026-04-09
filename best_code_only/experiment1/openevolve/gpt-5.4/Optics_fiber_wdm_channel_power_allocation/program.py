# EVOLVE-BLOCK-START
"""Compact greedy WDM allocation."""

from __future__ import annotations

import numpy as np


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
    d = np.asarray(user_demands_gbps, float)
    n_users = d.size
    n_channels = np.asarray(channel_centers_hz, float).size
    assignment = -np.ones(n_users, int)
    power_dbm = np.full(n_channels, pmin_dbm, float)

    if n_channels == 0 or n_users == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    total_lin = 10 ** (0.1 * total_power_dbm)
    pmin_lin = 10 ** (0.1 * pmin_dbm)
    pmax_lin = 10 ** (0.1 * pmax_dbm)
    served = min(n_users, n_channels, max(1, int(total_lin / max(pmin_lin, 1e-12))))
    order = np.argsort(-d)[:served]
    c = (n_channels - 1) / 2.0
    s = np.arange(served)
    used = np.clip(np.rint(c + (s - (served - 1) / 2.0) * n_channels / max(served, 1)).astype(int), 0, n_channels - 1)
    used = np.unique(used)
    if used.size < served:
        used = np.r_[used, np.setdiff1d(np.arange(n_channels), used)[:served - used.size]]
    used = used[np.argsort(np.abs(used - c))]
    assignment[order] = used

    avail_lin = max(total_lin - (n_channels - served) * pmin_lin, served * pmin_lin)
    w = np.maximum(d[order], 1e-9) ** 0.6
    w /= w.sum()
    plin = np.clip(avail_lin * w, pmin_lin, pmax_lin)
    if plin.sum() > avail_lin:
        plin *= avail_lin / plin.sum()
    power_dbm[used] = np.clip(10 * np.log10(np.maximum(plin, 1e-12)), pmin_dbm, pmax_dbm)

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
