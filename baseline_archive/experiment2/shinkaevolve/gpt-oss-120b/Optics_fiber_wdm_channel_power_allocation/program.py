# EVOLVE-BLOCK-START
"""Optimized WDM channel + power allocation solver.

Key improvements:
1. Aggressive isolation weighting with high exponent for better channel differentiation
2. Quadratic gap penalty with threshold=3 to strongly discourage adjacent channels
3. Demand-weighted power redistribution prioritizing high-demand users on good channels
4. Higher edge bonus to leverage asymmetric interference at spectrum edges
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
    """Allocate one channel per user and per-channel power with crosstalk awareness.

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
    target_ber : float
        Target BER threshold (used for optimization).

    Returns
    -------
    dict with keys:
      - assignment: np.ndarray shape (U,), channel index or -1
      - power_dbm: np.ndarray shape (C,), per-channel launch power
    """
    user_demands = np.asarray(user_demands_gbps, dtype=float)
    channel_centers = np.asarray(channel_centers_hz, dtype=float)

    n_users = len(user_demands)
    n_channels = channel_centers.size

    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)

    if n_users == 0 or n_channels == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    n_serve = min(n_users, n_channels)

    # Evenly-spaced channel selection for minimal crosstalk
    if n_serve == 1:
        selected_channels = [n_channels // 2]
    else:
        step = (n_channels - 1) / (n_serve - 1) if n_serve > 1 else 0
        selected_channels = [int(round(i * step)) for i in range(n_serve)]
        selected_channels = list(dict.fromkeys(selected_channels))
        available = [c for c in range(n_channels) if c not in selected_channels]
        while len(selected_channels) < n_serve and available:
            best = max(available, key=lambda c: min(abs(c - s) for s in selected_channels) if selected_channels else 0)
            selected_channels.append(best)
            available.remove(best)
        selected_channels = sorted(selected_channels[:n_serve])

    # Sort users by demand (descending)
    user_order = np.argsort(-user_demands)

    # Initial assignment: center channels prioritized for high-demand users
    channel_priority = sorted(selected_channels, key=lambda c: abs(c - n_channels // 2))

    for i, user_idx in enumerate(user_order[:n_serve]):
        assignment[user_idx] = channel_priority[i]

    assigned_users = user_order[:n_serve]

    def get_min_isolation(ch, all_used):
        """Get minimum distance to nearest neighbor channel."""
        others = [c for c in all_used if c != ch]
        return min(abs(ch - o) for o in others) if others else n_channels

    def compute_assignment_score():
        """Score assignment with aggressive isolation weighting and quadratic gap penalty."""
        used_channels = [assignment[u] for u in assigned_users]
        score = 0.0
        gap_penalty = 0.0
        for u in assigned_users:
            ch = assignment[u]
            isolation = get_min_isolation(ch, used_channels)
            # Aggressive isolation factor: lower base, higher exponent
            isolation_factor = 0.4 + 0.18 * isolation ** 2.8
            score += user_demands[u] * isolation_factor
            # Quadratic gap penalty with threshold=3
            if isolation < 3:
                gap_penalty += 150.0 * (3 - isolation) ** 2 * user_demands[u]
        return score - gap_penalty

    # Local search with more iterations
    for _ in range(15):
        improved = False
        current_score = compute_assignment_score()

        for i in range(len(assigned_users)):
            for j in range(i + 1, len(assigned_users)):
                u1, u2 = assigned_users[i], assigned_users[j]
                c1, c2 = assignment[u1], assignment[u2]

                assignment[u1], assignment[u2] = c2, c1
                new_score = compute_assignment_score()

                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    assignment[u1], assignment[u2] = c1, c2

        if not improved:
            break

    # Post-process: resolve adjacent channel conflicts
    used_channels = set(assignment[assignment >= 0])
    available_channels = [c for c in range(n_channels) if c not in used_channels]

    for _ in range(5):
        moved = False
        assigned_list = [(u, assignment[u]) for u in assigned_users]

        for u, ch in assigned_list:
            has_adjacent = (ch - 1 in used_channels) or (ch + 1 in used_channels)

            if has_adjacent and available_channels:
                best_ch = max(available_channels,
                             key=lambda c: min(abs(c - uc) for uc in used_channels if uc != ch))
                best_isolation = min(abs(best_ch - uc) for uc in used_channels if uc != ch)
                current_isolation = min(abs(ch - uc) for uc in used_channels if uc != ch)

                if best_isolation > current_isolation and best_isolation >= 2:
                    old_ch = assignment[u]
                    assignment[u] = best_ch
                    used_channels.remove(old_ch)
                    used_channels.add(best_ch)
                    available_channels.remove(best_ch)
                    available_channels.append(old_ch)
                    moved = True

        if not moved:
            break

    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    # Power allocation with aggressive isolation scaling
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    pmin_lin = 10 ** (float(pmin_dbm) / 10.0)
    pmax_lin = 10 ** (float(pmax_dbm) / 10.0)

    inactive_power = (n_channels - used.size) * pmin_lin
    active_budget = max(total_power_lin - inactive_power, used.size * pmin_lin)

    demands_assigned = np.array([user_demands[np.where(assignment == c)[0][0]] for c in used])
    total_demand = np.sum(demands_assigned)

    # Aggressive isolation factor: well-isolated channels get much more power
    used_list = list(used)
    isolation_factors = np.ones(len(used))
    for i, ch in enumerate(used):
        min_gap = n_channels
        for other in used_list:
            if other != ch:
                gap = abs(ch - other)
                min_gap = min(min_gap, gap)
        # Aggressive power scaling for better crosstalk tolerance
        isolation_factors[i] = 0.5 + 0.2 * min_gap ** 2.5

    # Higher edge bonus for asymmetric interference advantage
    edge_bonus = np.array([1.0 + 0.7 * (1 if c == 0 or c == n_channels - 1 else 0) for c in used])

    # Combined weight: demand * isolation * edge_bonus
    weights = demands_assigned * isolation_factors * edge_bonus
    weight_sum = np.sum(weights)

    # Allocate power proportionally
    for i, ch_idx in enumerate(used):
        demand_weight = weights[i] / weight_sum if weight_sum > 0 else 1.0 / used.size
        power_lin = np.clip(demand_weight * active_budget, pmin_lin, pmax_lin)
        power_dbm[ch_idx] = 10.0 * np.log10(max(power_lin, 1e-12))

    # Iterative power refinement with demand-weighted redistribution
    for _ in range(20):
        current_powers = np.array([10 ** (power_dbm[c] / 10.0) for c in used])
        total_used = np.sum(current_powers)

        if total_used < active_budget * 0.9999:
            headroom = pmax_lin - current_powers
            headroom = np.maximum(headroom, 0)
            total_headroom = np.sum(headroom)

            if total_headroom > 0:
                remaining = active_budget - total_used
                # Demand-weighted redistribution: high-demand users on good channels get priority
                weighted_headroom = headroom * weights
                total_weighted = np.sum(weighted_headroom)

                if total_weighted > 0:
                    for i, ch_idx in enumerate(used):
                        if headroom[i] > 0 and weighted_headroom[i] > 0:
                            extra = min(remaining * (weighted_headroom[i] / total_weighted), headroom[i])
                            new_power = np.clip(current_powers[i] + extra, pmin_lin, pmax_lin)
                            power_dbm[ch_idx] = 10.0 * np.log10(max(new_power, 1e-12))

                current_powers = np.array([10 ** (power_dbm[c] / 10.0) for c in used])
            else:
                break
        else:
            break

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END