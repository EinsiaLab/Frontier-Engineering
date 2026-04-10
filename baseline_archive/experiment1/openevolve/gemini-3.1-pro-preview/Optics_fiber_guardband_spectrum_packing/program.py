# EVOLVE-BLOCK-START
"""Baseline for Task 4: spectrum packing with guard bands."""

from __future__ import annotations

import numpy as np


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

    best_alloc = None
    best_score = -1

    rng = np.random.default_rng(seed)
    
    base_order = np.argsort(d)
    orders = [np.argsort(-d), base_order]
    
    for _ in range(250):
        o = base_order.copy()
        i, j = sorted(rng.integers(0, n_users, size=2))
        rng.shuffle(o[i:j+1])
        orders.append(o)

    for order in orders:
        alloc = [(-1, 0) for _ in range(n_users)]
        occupied = np.zeros(n_slots, dtype=bool)
        score = 0

        for u in order:
            width = int(d[u])
            if width <= 0 or width > n_slots:
                continue

            best_s = -1
            best_frag = 1000
            for s in range(0, n_slots - width + 1):
                left = max(0, s - guard_slots)
                right = min(n_slots, s + width + guard_slots)
                if not np.any(occupied[left:right]):
                    occ_tmp = occupied.copy()
                    occ_tmp[s : s + width] = True
                    frag = int(np.sum(~occ_tmp[1:] & occ_tmp[:-1])) + int(not occ_tmp[0])
                    if frag < best_frag:
                        best_frag = frag
                        best_s = s
            
            if best_s != -1:
                occupied[best_s : best_s + width] = True
                alloc[u] = (best_s, width)
                score += 10000 + width

        score -= int(np.sum(~occupied[1:] & occupied[:-1])) + int(not occupied[0])

        if score > best_score:
            best_score = score
            best_alloc = alloc

    return {"alloc": np.asarray(best_alloc, dtype=int)}
# EVOLVE-BLOCK-END
