# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: WDM channel + power allocation.

Simple engineering baseline:
- assign users to channels in index order
- split total launch power equally among assigned channels
"""

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
    demands = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)
    n_users = len(demands)
    n_channels = channel_centers_hz.size
    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)
    n_served = min(n_users, n_channels)
    if n_served > 0:
        idx = np.arange(n_served)
        spaced = np.round(idx * n_channels / n_served).astype(int)
        spaced = np.clip(spaced, 0, n_channels - 1)
        spaced = np.unique(spaced)
        if len(spaced) < n_served:
            extra = np.setdiff1d(np.arange(n_channels), spaced)
            spaced = np.concatenate((spaced, extra[:n_served - len(spaced)]))
        center = (n_channels - 1) / 2.0
        ch_order = spaced[np.argsort(np.abs(spaced - center))]
        user_order = np.argsort(-demands)
        for k in range(n_served):
            assignment[user_order[k]] = int(ch_order[k])
    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    inactive_lin = (n_channels - used.size) * (10 ** (float(pmin_dbm) / 10.0))
    active_budget_lin = max(total_power_lin - inactive_lin, used.size * (10 ** (float(pmin_dbm) / 10.0)))
    served_demands = demands[assignment >= 0]
    if served_demands.sum() > 0:
        weights = served_demands / served_demands.sum()
        alloc_lin = weights * active_budget_lin
    else:
        alloc_lin = np.full(used.size, active_budget_lin / used.size)
    alloc_dbm = 10.0 * np.log10(np.maximum(alloc_lin, 1e-12))
    alloc_dbm = np.clip(alloc_dbm, pmin_dbm, pmax_dbm)
    power_dbm[used] = alloc_dbm
    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
