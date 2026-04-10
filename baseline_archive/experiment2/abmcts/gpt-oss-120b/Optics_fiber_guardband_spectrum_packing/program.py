# EVOLVE-BLOCK-START
"""Baseline for Task 4: spectrum packing with guard bands."""

from __future__ import annotations

import numpy as np


def _pack_given_order(demands: np.ndarray, order: np.ndarray, n_slots: int, guard_slots: int) -> np.ndarray:
    """Pack users following a specific order using a best‑fit placement heuristic.

    Parameters
    ----------
    demands : np.ndarray
        Integer array of requested slot widths per user.
    order : np.ndarray
        Indices defining the placement order.
    n_slots : int
        Total number of spectrum slots.
    guard_slots : int
        Number of guard slots required on each side of an allocation.

    Returns
    -------
    np.ndarray
        Allocation array of shape (n_users, 2) with (start, width) or (-1, 0) for rejected users.
    """
    n_users = demands.size
    alloc = np.full((n_users, 2), fill_value=(-1, 0), dtype=int)
    occupied = np.zeros(n_slots, dtype=bool)

    for u in order:
        width = int(demands[u])
        if width <= 0 or width > n_slots:
            continue

        best_start = -1
        best_interval = n_slots + 1  # larger than any possible interval

        # Scan all feasible start positions
        for s in range(0, n_slots - width + 1):
            left = max(0, s - guard_slots)
            right = min(n_slots, s + width + guard_slots)
            if not occupied[left:right].any():
                interval_len = right - left
                if interval_len < best_interval or (interval_len == best_interval and s < best_start):
                    best_interval = interval_len
                    best_start = s

        if best_start >= 0:
            occupied[best_start: best_start + width] = True
            alloc[u] = (best_start, width)

    return alloc


def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """Improved spectrum packing heuristic.

    The function evaluates two ordering strategies (large‑first and small‑first)
    using a best‑fit placement rule and selects the allocation with the highest
    acceptance ratio (and, as a tie‑breaker, higher utilization).

    Returns
    -------
    dict
        ``{'alloc': np.ndarray}`` where each row is ``(start_slot, width)`` or
        ``(-1, 0)`` for an unallocated request.
    """
    d = np.asarray(user_demand_slots, dtype=int)
    n_users = d.size

    # Ordering strategies
    orders = [
        np.argsort(-d),  # descending (large first)
        np.argsort(d)    # ascending (small first)
    ]

    best_alloc = None
    best_accept = -1
    best_util = -1.0

    for order in orders:
        alloc = _pack_given_order(d, order, n_slots, guard_slots)

        allocated_mask = alloc[:, 0] != -1
        accept = allocated_mask.sum()
        util = alloc[allocated_mask, 1].sum() / n_slots if n_slots > 0 else 0.0

        if (accept > best_accept) or (accept == best_accept and util > best_util):
            best_accept = accept
            best_util = util
            best_alloc = alloc

    return {"alloc": best_alloc}
# EVOLVE-BLOCK-END
