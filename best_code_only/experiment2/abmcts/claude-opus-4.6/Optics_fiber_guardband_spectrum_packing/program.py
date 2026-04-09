# EVOLVE-BLOCK-START
"""Baseline for Task 4: spectrum packing with guard bands."""

from __future__ import annotations

import numpy as np


def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """Advanced spectrum packing with multiple strategies.

    Returns
    -------
    dict with key `alloc`:
      alloc[i] = (start_slot, width)
      if not allocated: (-1, 0)
    """
    d = np.asarray(user_demand_slots, dtype=int)
    n_users = d.size

    best_alloc = None
    best_score = -1.0

    # Try multiple strategies and pick the best
    strategies = []

    # Strategy 1: Small-first best-fit (like oracle heuristic)
    strategies.append(('small_first_bestfit', np.argsort(d)))

    # Strategy 2: Large-first best-fit
    strategies.append(('large_first_bestfit', np.argsort(-d)))

    # Strategy 3: Medium-first (interleave small and large)
    sorted_asc = np.argsort(d)
    mid_order = []
    lo, hi = 0, len(sorted_asc) - 1
    toggle = True
    while lo <= hi:
        if toggle:
            mid_order.append(sorted_asc[lo])
            lo += 1
        else:
            mid_order.append(sorted_asc[hi])
            hi -= 1
        toggle = not toggle
    strategies.append(('interleave', np.array(mid_order)))

    # Strategy 4: Random permutations (local search)
    rng = np.random.RandomState(seed)
    for trial in range(200):
        perm = rng.permutation(n_users)
        strategies.append(('random_' + str(trial), perm))

    # Strategy 5: Hybrid local search - start from small-first, do swaps
    base_order = np.argsort(d).copy()
    for trial in range(100):
        order = base_order.copy()
        # Random swaps
        n_swaps = rng.randint(1, max(2, n_users // 3))
        for _ in range(n_swaps):
            i, j = rng.randint(0, n_users, size=2)
            order[i], order[j] = order[j], order[i]
        strategies.append(('hybrid_' + str(trial), order))

    for name, order in strategies:
        alloc = _pack_with_order_bestfit(d, n_slots, guard_slots, order)
        score = _proxy_score(alloc, d, n_slots, guard_slots)
        if score > best_score:
            best_score = score
            best_alloc = alloc

    return {"alloc": np.asarray(best_alloc, dtype=int)}


def _pack_with_order_bestfit(d, n_slots, guard_slots, order):
    """Pack users in given order using best-fit placement."""
    n_users = len(d)
    alloc = [(-1, 0)] * n_users
    occupied = np.zeros(n_slots, dtype=bool)

    for u in order:
        width = int(d[u])
        if width <= 0 or width > n_slots:
            continue

        best_start = -1
        best_waste = n_slots + 1

        # Find all feasible positions and pick best-fit
        s = 0
        while s <= n_slots - width:
            left = max(0, s - guard_slots)
            right = min(n_slots, s + width + guard_slots)
            if not np.any(occupied[left:right]):
                # Calculate waste: size of the free block containing this position
                # Find the free block boundaries
                fb_start = s
                while fb_start > 0 and not occupied[fb_start - 1]:
                    fb_start -= 1
                fb_end = s + width
                while fb_end < n_slots and not occupied[fb_end]:
                    fb_end += 1
                waste = (fb_end - fb_start) - width
                if waste < best_waste:
                    best_waste = waste
                    best_start = s
                    if waste == 0:
                        break
                # Skip to next possible position
                s += 1
            else:
                # Skip past the occupied region
                s += 1

        if best_start >= 0:
            occupied[best_start: best_start + width] = True
            alloc[u] = (best_start, width)

    return alloc


def _proxy_score(alloc, d, n_slots, guard_slots):
    """Compute proxy score matching the evaluation metric."""
    n_users = len(d)
    accepted = sum(1 for s, w in alloc if s >= 0)
    acceptance = accepted / max(n_users, 1)

    total_demand = sum(d[i] for i in range(n_users) if alloc[i][0] >= 0)
    if accepted > 0:
        max_end = max(s + w for s, w in alloc if s >= 0)
        utilization = total_demand / max(max_end, 1)
    else:
        utilization = 0.0

    # Count free blocks for compactness
    occupied = np.zeros(n_slots, dtype=bool)
    for s, w in alloc:
        if s >= 0:
            occupied[s:s + w] = True
    free_blocks = 0
    in_free = False
    for i in range(n_slots):
        if not occupied[i]:
            if not in_free:
                free_blocks += 1
                in_free = True
        else:
            in_free = False
    compactness = 1.0 / max(free_blocks, 1)

    # BER pass ratio assumed ~1.0 for reasonable placements with guard bands
    ber_pass = 1.0

    score = 0.80 * acceptance + 0.05 * utilization + 0.05 * compactness + 0.10 * ber_pass
    return score
# EVOLVE-BLOCK-END
