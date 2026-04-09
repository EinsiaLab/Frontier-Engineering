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

    # Serve highest‑demand users first, assigning them to spaced channels
    # to reduce inter‑channel interference (mirrors the oracle strategy).
    n_served = min(n_users, n_channels)
    demand_arr = np.asarray(user_demands_gbps, dtype=float)
    sorted_idx = np.argsort(-demand_arr)  # descending order (largest demand first)

    # ---- generate evenly‑spaced channel indices ----
    # Use a linear spacing across the full channel grid.
    raw = np.linspace(0, n_channels - 1, n_served)
    cand = np.rint(raw).astype(int)
    cand = np.clip(cand, 0, n_channels - 1)

    # Remove duplicates while preserving order, then fill any missing slots
    # with the first still‑free channels.
    used_set = set()
    ch_order = []
    for c in cand:
        if c not in used_set:
            used_set.add(int(c))
            ch_order.append(int(c))
    if len(ch_order) < n_served:
        for c in range(n_channels):
            if c not in used_set:
                used_set.add(c)
                ch_order.append(c)
            if len(ch_order) == n_served:
                break
    # ---- end of channel generation ----

    # ------------------------------------------------------------------
    # Re‑order the selected channels by isolation (largest nearest‑neighbour
    # distance first).  This gives the most isolated channels to the
    # highest‑demand users, helping to lower inter‑channel interference.
    # ------------------------------------------------------------------
    if len(ch_order) > 1:
        ch_arr = np.asarray(ch_order, dtype=int)
        # distance to the closest other selected channel
        nearest_dist = np.full(ch_arr.shape, np.inf, dtype=float)
        for i, ch in enumerate(ch_arr):
            dists = np.abs(ch_arr - ch).astype(float)
            dists[i] = np.inf
            nearest_dist[i] = dists.min()
        # sort channels so that the most isolated come first
        ordered_idx = np.argsort(-nearest_dist)
        ch_order = ch_arr[ordered_idx].tolist()

    # Assign the highest‑demand users to the spaced channels.
    for user_idx, ch in zip(sorted_idx[:n_served], ch_order):
        assignment[user_idx] = ch

    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    # Power allocation that blends demand proportion with channel isolation.
    # The total linear power budget (mW) and per‑channel minimum.
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    pmin_lin = 10 ** (float(pmin_dbm) / 10.0)

    # Power that must stay on inactive channels (they keep pmin).
    inactive_lin = (n_channels - used.size) * pmin_lin

    # Power that can be shared among the *active* channels.
    # Never less than the mandatory minimum per active channel.
    active_budget_lin = max(total_power_lin - inactive_lin,
                           used.size * pmin_lin)

    # Extra power above the mandatory minimum for each active channel.
    extra_budget_lin = max(active_budget_lin - used.size * pmin_lin, 0.0)

    # --------------------------------------------------------------
    # Demand‑based weights (one per served user)
    # --------------------------------------------------------------
    served_mask = assignment >= 0
    served_user_idxs = np.where(served_mask)[0]

    served_demands = demand_arr[served_user_idxs]
    demand_sum = served_demands.sum()

    if demand_sum > 0:
        demand_weights = served_demands / demand_sum
    else:
        demand_weights = np.full_like(served_demands,
                                      1.0 / used.size, dtype=float)

    # --------------------------------------------------------------
    # Distance‑based weights for each active channel
    # --------------------------------------------------------------
    if used.size > 1:
        channel_positions = used.astype(float)
        nearest_dist = np.full(used.shape, np.inf, dtype=float)
        for i, ch in enumerate(used):
            dists = np.abs(channel_positions - ch)
            dists[i] = np.inf
            nearest_dist[i] = dists.min()
        distance_weights = nearest_dist / (nearest_dist.sum() + 1e-12)
    else:
        # Single active channel → treat as fully isolated
        distance_weights = np.full(used.shape, 1.0, dtype=float)

    # Map distance weights to the order of served users (one per channel)
    dist_per_user = np.array(
        [distance_weights[np.where(used == assignment[u])[0][0]]
         for u in served_user_idxs]
    )

    # --------------------------------------------------------------
    # Blend the two weight vectors.
    # α = 0.6 gives a slight emphasis to demand while rewarding isolation.
    # --------------------------------------------------------------
    # Reduce the emphasis on raw demand a little and reward channel isolation more.
    # This tends to improve SNR for well‑separated channels while keeping demand
    # satisfaction high.
    # Give a slightly larger emphasis to channel isolation (distance weighting)
    alpha = 0.4
    combined_weights = alpha * demand_weights + (1.0 - alpha) * dist_per_user
    combined_weights = combined_weights / (combined_weights.sum() + 1e-12)

    # --------------------------------------------------------------
    # Power allocation:
    #   – start from the mandatory minimum per channel,
    #   – add the demand‑/distance‑based share of the extra budget,
    #   – enforce the per‑channel max limit,
    #   – if some budget is left (because of clipping), redistribute it
    #     to channels that are still below the max, again weighted by the
    #     combined_weights.
    # --------------------------------------------------------------
    # Initial linear power per active channel
    chan_lin = pmin_lin + combined_weights * extra_budget_lin

    # Enforce the per‑channel maximum (convert dBm limit to linear mW)
    pmax_lin = 10 ** (float(pmax_dbm) / 10.0)
    chan_lin = np.minimum(chan_lin, pmax_lin)

    # If clipping left unused budget, give it back proportionally
    total_active_lin = np.sum(chan_lin)
    if total_active_lin < active_budget_lin:
        leftover = active_budget_lin - total_active_lin
        # Channels that are NOT already at the maximum
        not_max = chan_lin < pmax_lin - 1e-12
        if np.any(not_max):
            # Distribute leftover according to the weights of those channels
            weights = combined_weights * not_max
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                addition = leftover * (weights / weight_sum)
                chan_lin = np.minimum(chan_lin + addition, pmax_lin)

    # Convert back to dBm for the final output
    chan_dbm = 10.0 * np.log10(np.maximum(chan_lin, 1e-12))

    # Write the final powers into the full channel array.
    power_dbm[used] = chan_dbm

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
