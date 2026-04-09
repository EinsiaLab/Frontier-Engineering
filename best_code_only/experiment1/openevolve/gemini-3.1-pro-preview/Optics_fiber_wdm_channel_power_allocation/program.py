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
    _ = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)

    n_users = len(user_demands_gbps)
    n_channels = channel_centers_hz.size

    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)

    # Space out channels to minimize adjacent channel interference
    n_served = min(n_users, n_channels)
    if n_served > 0:
        sorted_users = np.argsort(-np.asarray(user_demands_gbps))
        spaced = np.linspace(0, n_channels - 1, n_served).round().astype(int)
        
        # Prioritize edge channels (furthest from center)
        center = (n_channels - 1) / 2.0
        channel_order = spaced[np.argsort(-np.abs(spaced - center))]
        assignment[sorted_users[:n_served]] = channel_order

    used = assignment[assignment >= 0]
    if used.size > 0:
        tot_lin = 10 ** (total_power_dbm / 10.0)
        inact_lin = (n_channels - used.size) * (10 ** (pmin_dbm / 10.0))
        act_lin = max(tot_lin - inact_lin, used.size * (10 ** (pmin_dbm / 10.0)))
        power_dbm[used] = float(np.clip(10.0 * np.log10(max(act_lin / used.size, 1e-12)), pmin_dbm, pmax_dbm))

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
