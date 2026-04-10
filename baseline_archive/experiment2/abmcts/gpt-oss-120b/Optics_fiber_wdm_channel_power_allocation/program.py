# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: WDM channel + power allocation.

Improved engineering baseline:
- assign users to channels in order of descending demand, spreading them evenly across the spectrum
- allocate per‑channel launch power proportionally to user demand
- respect total power budget and per‑channel limits with iterative redistribution
"""

from __future__ import annotations

import numpy as np


def _redistribute_power(
    lin_vals: np.ndarray,
    min_lin: float,
    max_lin: float,
    total_budget_lin: float,
    max_iter: int = 10,
) -> np.ndarray:
    """Iteratively clamp values to [min_lin, max_lin] while preserving total budget.

    Parameters
    ----------
    lin_vals : np.ndarray
        Initial linear power values for each active channel.
    min_lin, max_lin : float
        Linear lower/upper bounds corresponding to pmin/pmax.
    total_budget_lin : float
        Desired sum of linear powers after redistribution.
    max_iter : int
        Maximum number of redistribution passes.

    Returns
    -------
    np.ndarray
        Power values respecting bounds and (approximately) the budget.
    """
    vals = lin_vals.copy()
    for _ in range(max_iter):
        vals = np.clip(vals, min_lin, max_lin)
        diff = total_budget_lin - vals.sum()
        if np.abs(diff) < 1e-12:
            break

        adjustable = (vals > min_lin + 1e-12) & (vals < max_lin - 1e-12)

        if not np.any(adjustable):
            slack_up = max_lin - vals
            slack_down = vals - min_lin
            total_slack = slack_up.sum() + slack_down.sum()
            if total_slack == 0:
                break
            vals += diff * (slack_up - slack_down) / total_slack
            continue

        vals[adjustable] += diff / adjustable.sum()
    return np.clip(vals, min_lin, max_lin)


def allocate_wdm(
    user_demands_gbps,
    channel_centers_hz,
    total_power_dbm,
    pmin_dbm: float = -8.0,
    pmax_dbm: float = 3.0,
    target_ber: float = 1e-3,
    seed: int = 0,
):
    """Allocate one channel per user and per‑channel launch power.

    Parameters
    ----------
    user_demands_gbps : array‑like, shape (U,)
        Requested data rate per user (Gbps).
    channel_centers_hz : array‑like, shape (C,)
        Fixed WDM grid centre frequencies (Hz).
    total_power_dbm : float
        Total launch power budget (dBm, summed across *all* channels).
    pmin_dbm, pmax_dbm : float, optional
        Per‑channel power limits (default –8 dBm … +3 dBm).
    target_ber : float, optional
        Desired BER threshold (currently unused by the baseline).
    seed : int, optional
        Random seed for any stochastic tie‑breaking (deterministic).

    Returns
    -------
    dict
        ``{\"assignment\": np.ndarray, \"power_dbm\": np.ndarray}`` where
        ``assignment`` maps each user to a channel index (or -1 if unserved)
        and ``power_dbm`` gives per‑channel launch powers.
    """
    rng = np.random.default_rng(seed)

    user_demands = np.asarray(user_demands_gbps, dtype=float)
    channel_centers = np.asarray(channel_centers_hz, dtype=float)

    n_users = user_demands.size
    n_channels = channel_centers.size

    # ------------------------------------------------------------------
    # Channel assignment – serve up to the number of available channels.
    # Users with higher demand are served first, and channels are spread
    # evenly across the spectrum to improve utilization.
    # ------------------------------------------------------------------
    assignment = -np.ones(n_users, dtype=int)

    # Sort users by descending demand, break ties randomly but deterministically
    sorted_idx = np.lexsort((rng.random(n_users), -user_demands))
    n_served = min(n_users, n_channels)

    # Choose channel indices that are as evenly spaced as possible
    used_channels = np.linspace(0, n_channels - 1, n_served, dtype=int)

    for i, u in enumerate(sorted_idx[:n_served]):
        assignment[u] = used_channels[i]

    n_used = used_channels.size

    if n_used == 0:
        power_dbm = np.full(n_channels, pmin_dbm, dtype=float)
        return {"assignment": assignment, "power_dbm": power_dbm}

    # ------------------------------------------------------------------
    # Power allocation – proportional to user demand while respecting limits.
    # ------------------------------------------------------------------
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    pmin_lin = 10 ** (float(pmin_dbm) / 10.0)
    pmax_lin = 10 ** (float(pmax_dbm) / 10.0)

    # Reserve minimum power for inactive channels
    inactive_budget_lin = (n_channels - n_used) * pmin_lin

    # Remaining budget that can be flexibly distributed among active channels
    active_budget_lin = max(total_power_lin - inactive_budget_lin,
                            n_used * pmin_lin)

    # Demand‑weighted initial linear powers for the served users
    demands_used = user_demands[assignment >= 0]
    if demands_used.sum() == 0:
        init_lin = np.full(n_used, active_budget_lin / n_used)
    else:
        weights = demands_used / demands_used.sum()
        init_lin = active_budget_lin * weights

    # Adjust to respect per‑channel bounds while keeping total budget
    final_lin = _redistribute_power(
        init_lin,
        min_lin=pmin_lin,
        max_lin=pmax_lin,
        total_budget_lin=active_budget_lin,
    )

    # Convert back to dBm and fill the full channel vector
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)
    power_dbm[used_channels] = 10.0 * np.log10(final_lin)

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
