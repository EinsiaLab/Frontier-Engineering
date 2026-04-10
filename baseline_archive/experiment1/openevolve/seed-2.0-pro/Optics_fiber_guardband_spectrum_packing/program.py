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

    alloc = [(-1, 0) for _ in range(n_users)]

    # Hybrid sort: large requests (>=7 slots) first descending, then small requests ascending
    # Balances high acceptance ratio (small users fill gaps) with better BER performance (large users placed first to avoid interference)
    # Lower threshold places medium allocations earlier to avoid gaps becoming too small for them, improving acceptance ratio
    sort_keys = np.array([(-w, w) if w >= 7 else (0, w) for w in d])
    order = np.lexsort(sort_keys.T)

    occupied = np.zeros(n_slots, dtype=bool)

    for u in order:
        width = int(d[u])
        if width <= 0 or width > n_slots:
            continue

        # Best Fit with BER-optimized tiebreaker: minimize wasted gap space, then maximize distance to existing allocations to reduce interference
        valid_starts = []
        occ_pos = np.where(occupied)[0]
        for s in range(0, n_slots - width + 1):
            left = max(0, s - guard_slots)
            right = min(n_slots, s + width + guard_slots)
            if not np.any(occupied[left:right]):
                # Calculate full size of the free gap containing this candidate position
                gap_left = left
                while gap_left > 0 and not occupied[gap_left - 1]:
                    gap_left -= 1
                gap_right = right
                while gap_right < n_slots and not occupied[gap_right]:
                    gap_right += 1
                gap_size = gap_right - gap_left
                
                # Calculate minimal distance to existing occupied slots to measure interference risk
                if len(occ_pos) == 0:
                    nearest_dist = float('inf')
                else:
                    dist_left = np.min(np.abs(occ_pos - s))
                    dist_right = np.min(np.abs(occ_pos - (s + width - 1)))
                    nearest_dist = min(dist_left, dist_right)
                
                # Sort keys: smallest gap first, then largest weighted distance first (negative for ascending sort), then earliest position for determinism
                # Weight distance by allocation width: larger allocations are more sensitive to interference, so prioritize more distance for them to improve BER pass rate
                valid_starts.append((gap_size, -(nearest_dist * width), s))
        
        if valid_starts:
            valid_starts.sort()
            _, _, s = valid_starts[0]
            occupied[s : s + width] = True
            alloc[u] = (s, width)
        else:
            alloc[u] = (-1, 0)

    return {"alloc": np.asarray(alloc, dtype=int)}
# EVOLVE-BLOCK-END
