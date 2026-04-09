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
    user_demands = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)

    n_users = len(user_demands_gbps)
    n_channels = channel_centers_hz.size

    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)

    # Prioritize higher-demand users for channel assignment, use evenly spaced channels to minimize interference
    sorted_user_indices = np.argsort(-user_demands)  # Sort descending by demand
    n_served = min(n_users, n_channels)
    if n_served > 0:
        # Sort channels by frequency first to ensure even spacing minimizes cross-talk correctly even if input channels are unsorted
        sorted_channel_indices = np.argsort(channel_centers_hz)
        # Generate evenly spaced indices from the sorted frequency list
        spaced_indices = np.linspace(0, n_channels - 1, n_served, dtype=int)
        spaced_channels = sorted_channel_indices[spaced_indices]
        # Remove duplicates if any, fill gaps with remaining free channels
        spaced_channels = np.unique(spaced_channels)
        if len(spaced_channels) < n_served:
            all_channels = np.arange(n_channels)
            free_channels = np.setdiff1d(all_channels, spaced_channels)
            spaced_channels = np.concatenate([spaced_channels, free_channels[:n_served - len(spaced_channels)]])
        # Assign highest demand users first to the spaced channels
        for idx in range(n_served):
            u = sorted_user_indices[idx]
            assignment[u] = spaced_channels[idx]

    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    # Proportional power split: allocate more power to higher-demand users
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    inactive_lin = (n_channels - used.size) * (10 ** (float(pmin_dbm) / 10.0))
    active_budget_lin = max(total_power_lin - inactive_lin, used.size * (10 ** (float(pmin_dbm) / 10.0)))
    
    # Calculate per-channel demand weights
    channel_demand = np.zeros(n_channels, dtype=float)
    for u in range(n_users):
        c = assignment[u]
        if c >= 0:
            channel_demand[c] = user_demands[u]
    total_active_demand = channel_demand[used].sum()

    if total_active_demand <= 1e-9:
        # Fallback to equal split if no valid demand values
        each_lin = active_budget_lin / used.size
        each_dbm = 10.0 * np.log10(max(each_lin, 1e-12))
        each_dbm = float(np.clip(each_dbm, pmin_dbm, pmax_dbm))
        power_dbm[used] = each_dbm
    else:
        # Allocate power with sqrt demand weight to favor higher demand users more strongly (matches logarithmic capacity curve)
        demand_sqrt = np.sqrt(channel_demand[used])
        total_sqrt = demand_sqrt.sum()
        for idx, c in enumerate(used):
            weight = demand_sqrt[idx] / total_sqrt if total_sqrt > 1e-9 else 1 / len(used)
            p_lin = weight * active_budget_lin
            p_dbm = 10.0 * np.log10(max(p_lin, 1e-12))
            power_dbm[c] = float(np.clip(p_dbm, pmin_dbm, pmax_dbm))

    # Renormalize: prioritize higher demand channels to use leftover budget first, up to pmax
    active_lin_total = np.sum(10 ** (power_dbm[used] / 10.0))
    remaining_lin = max(active_budget_lin - active_lin_total, 0.0)
    if remaining_lin > 1e-12 and used.size > 0:
        # Sort used channels descending by demand to prioritize higher demand users first
        sorted_used = sorted(used, key=lambda c: -channel_demand[c])
        pmax_lin = 10 ** (pmax_dbm / 10.0)
        # Allocate leftover power to higher demand users first, up to pmax
        for c in sorted_used:
            if remaining_lin <= 1e-12:
                break
            curr_lin = 10 ** (power_dbm[c] / 10.0)
            available_add = pmax_lin - curr_lin
            if available_add <= 1e-12:
                continue
            add_amount = min(remaining_lin, available_add)
            new_lin = curr_lin + add_amount
            power_dbm[c] = float(10.0 * np.log10(new_lin))
            remaining_lin -= add_amount
        # Distribute any remaining budget equally across all active channels if leftover
        if remaining_lin > 1e-12:
            add_per_channel = remaining_lin / used.size
            for c in used:
                curr_lin = 10 ** (power_dbm[c] / 10.0)
                new_lin = min(curr_lin + add_per_channel, pmax_lin)
                power_dbm[c] = float(10.0 * np.log10(new_lin))

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
