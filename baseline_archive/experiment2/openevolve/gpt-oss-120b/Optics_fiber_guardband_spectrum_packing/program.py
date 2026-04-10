# EVOLVE-BLOCK-START
"""Baseline for Task 4: spectrum packing with guard bands."""

from __future__ import annotations

import numpy as np
import time


def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """First-Fit Decreasing baseline.

    Returns
    -------
    dict with key `alloc`:
      alloc[i] = (start_slot, width)
      if not allocated: (-1, 0)
    """
    d = np.asarray(user_demand_slots, dtype=int)
    n_users = d.size

    alloc = [(-1, 0) for _ in range(n_users)]

    # -----------------------------------------------------------------
    # Helper: allocate using a best‑fit rule that minimises guard‑band waste.
    # -----------------------------------------------------------------
    def _allocate(ordering):
        occ = np.zeros(n_slots, dtype=bool)            # core slots only
        alloc_local = [(-1, 0) for _ in range(n_users)]

        for u in ordering:
            width = int(d[u])
            if width <= 0 or width > n_slots:
                continue

            best_s   = -1
            best_waste = n_slots + 1                     # larger than any possible waste

            for s in range(0, n_slots - width + 1):
                left  = max(0, s - guard_slots)
                right = min(n_slots, s + width + guard_slots)

                # interval (including guards) must be free
                if np.any(occ[left:right]):
                    continue

                waste = (right - left) - width            # guard‑band space that would stay unused
                if waste < best_waste or (waste == best_waste and s > best_s):
                    best_waste = waste
                    best_s = s

            if best_s != -1:
                occ[best_s:best_s + width] = True
                alloc_local[u] = (best_s, width)

        return np.asarray(alloc_local, dtype=int)

    # -----------------------------------------------------------------
    # Try two simple orderings: large‑first and small‑first, with
    # deterministic tie‑shuffling of users that have the same demand.
    # This reduces fragmentation while keeping the algorithm fully
    # reproducible (the RNG is seeded).
    # -----------------------------------------------------------------
    rng = np.random.default_rng(seed)

    def _shuffle_ties(order, values):
        i = 0
        n = len(order)
        while i < n:
            j = i + 1
            while j < n and values[order[i]] == values[order[j]]:
                j += 1
            if j - i > 1:                              # more than one with same demand
                segment = order[i:j].copy()
                rng.shuffle(segment)
                order[i:j] = segment
            i = j
        return order

    order_desc = _shuffle_ties(np.argsort(-d, kind='stable'), d)   # larger first
    order_asc  = _shuffle_ties(np.argsort(d, kind='stable'), d)    # smaller first

    alloc_desc = _allocate(order_desc)
    alloc_asc  = _allocate(order_asc)

    # -----------------------------------------------------------------
    # Choose the allocation with higher acceptance; tie‑break on total width.
    # Then perform a tiny stochastic local‑search to try to improve it.
    # -----------------------------------------------------------------
    def _acceptance(a):
        return np.mean(a[:, 0] >= 0)

    def _total_width(a):
        return np.sum(a[:, 1])

    # Start from the better of the two deterministic orderings
    best_order = order_desc
    best_alloc = alloc_desc
    if _acceptance(alloc_asc) > _acceptance(best_alloc):
        best_order = order_asc
        best_alloc = alloc_asc
    elif _acceptance(alloc_asc) == _acceptance(best_alloc):
        if _total_width(alloc_asc) > _total_width(best_alloc):
            best_order = order_asc
            best_alloc = alloc_asc

    # ----- tiny hill‑climbing search (swap two users) -----
    rng_ls = np.random.default_rng(seed + 1)   # deterministic but different sequence
    start_time = time.time()
    max_steps = 2000               # safety cap
    n = n_users

    for step in range(max_steps):
        # time‑budget stop (≈ 4 s)
        if time.time() - start_time > 4.0:
            break

        i, j = rng_ls.integers(0, n, size=2)
        if i == j:
            continue
        cand_order = best_order.copy()
        cand_order[i], cand_order[j] = cand_order[j], cand_order[i]

        cand_alloc = _allocate(cand_order)

        # keep if strictly better acceptance, or equal acceptance but larger width
        if _acceptance(cand_alloc) > _acceptance(best_alloc):
            best_order = cand_order
            best_alloc = cand_alloc
        elif _acceptance(cand_alloc) == _acceptance(best_alloc):
            if _total_width(cand_alloc) > _total_width(best_alloc):
                best_order = cand_order
                best_alloc = cand_alloc

    alloc = best_alloc

    # -------------------------------------------------------------
    # Post‑process: try to fit any still‑unallocated users into the
    # remaining free core slots (respecting guard‑band constraints).
    # -------------------------------------------------------------
    def _fill_remaining(allocation, occupied, order=None):
        """
        Fill free gaps with still‑unallocated users.

        Parameters
        ----------
        allocation : np.ndarray
            Current allocation matrix.
        occupied : np.ndarray
            Boolean occupancy mask for core slots.
        order : iterable of int, optional
            Sequence of user indices to try.  If ``None`` the users are tried
            *small‑first* (ascending demand).  Supplying a list allows a
            *large‑first* (descending demand) strategy.
        """
        alloc_filled = allocation.copy()
        occ = occupied.copy()

        # Determine which users are still unallocated
        unassigned = [i for i, (s, w) in enumerate(alloc_filled) if s == -1]

        if order is None:
            # default: smallest demand first (more likely to fit)
            unassigned.sort(key=lambda i: d[i])
        else:
            # respect the supplied order, but keep only still‑unassigned users
            unassigned = [i for i in order if i in unassigned]

        for u in unassigned:
            width = int(d[u])
            if width <= 0 or width > n_slots:
                continue
            for s in range(0, n_slots - width + 1):
                left = max(0, s - guard_slots)
                right = min(n_slots, s + width + guard_slots)
                if np.any(occ[left:right]):
                    continue
                occ[s:s + width] = True
                alloc_filled[u] = (s, width)
                break
        return alloc_filled, occ

    # Build occupancy mask for the chosen allocation
    occupied = np.zeros(n_slots, dtype=bool)
    for s, w in alloc:
        if s >= 0:
            occupied[s:s + w] = True

    # -----------------------------------------------------------------
    # Run two filling strategies (small‑first and large‑first) and keep
    # the one that yields the higher acceptance/width score.
    # -----------------------------------------------------------------
    # 1) Small‑first (default)
    alloc_small, _ = _fill_remaining(alloc, occupied)

    # 2) Large‑first (descending demand)
    still_unassigned = [i for i, (s, w) in enumerate(alloc) if s == -1]
    large_order = sorted(still_unassigned, key=lambda i: -d[i])
    alloc_large, _ = _fill_remaining(alloc, occupied, order=large_order)

    # Helper to compare two filled allocations
    def _score_filled(a):
        accepted = a[:, 0] >= 0
        return (int(np.sum(accepted)), int(np.sum(a[accepted, 1])))

    # Choose the better filled allocation
    if _score_filled(alloc_large) > _score_filled(alloc_small):
        alloc = alloc_large
    else:
        alloc = alloc_small

    return {"alloc": alloc}
# EVOLVE-BLOCK-END
