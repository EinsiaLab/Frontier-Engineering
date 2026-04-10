# EVOLVE-BLOCK-START
"""Baseline for Task 4: spectrum packing with guard bands."""

from __future__ import annotations

import numpy as np


def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """Spectrum packing with first-fit and local search on user ordering.

    Uses first-fit placement (better for acceptance) and performs local
    search to explore user orderings systematically.

    Returns
    -------
    dict with key `alloc`:
      alloc[i] = (start_slot, width)
      if not allocated: (-1, 0)
    """
    d = np.asarray(user_demand_slots, dtype=int)
    n_users = d.size

    best_alloc = [(-1, 0) for _ in range(n_users)]
    best_score = (0, 0)  # (accepted, utilized)

    def first_fit_placement(order):
        """Place users in order using first-fit strategy."""
        alloc = [(-1, 0) for _ in range(n_users)]
        occupied = np.zeros(n_slots, dtype=bool)

        for u in order:
            width = int(d[u])
            if width <= 0 or width > n_slots:
                continue

            # First-fit: take first valid position
            for s in range(0, n_slots - width + 1):
                left = max(0, s - guard_slots)
                right = min(n_slots, s + width + guard_slots)
                if not np.any(occupied[left:right]):
                    occupied[s:s + width] = True
                    alloc[u] = (s, width)
                    break

        return alloc

    def score_allocation(alloc):
        """Score by acceptance (primary) and utilization (secondary)."""
        accepted = sum(1 for a in alloc if a[0] >= 0)
        utilized = sum(d[u] for u, a in enumerate(alloc) if a[0] >= 0)
        return (accepted, utilized)

    def local_search(initial_order, rng, max_iterations=30):
        """Hill-climbing local search on user ordering."""
        current_order = initial_order.copy()
        current_alloc = first_fit_placement(current_order)
        current_score = score_allocation(current_alloc)

        for _ in range(max_iterations):
            improved = False
            # Try all pairwise swaps
            for i in range(n_users):
                for j in range(i + 1, n_users):
                    new_order = current_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_alloc = first_fit_placement(new_order)
                    new_score = score_allocation(new_alloc)
                    if new_score > current_score:
                        current_order = new_order
                        current_alloc = new_alloc
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return current_alloc, current_score

    rng = np.random.RandomState(seed)

    # Starting orderings for local search
    starting_orders = [
        np.argsort(d),      # smallest first
        np.argsort(-d),     # largest first
    ]

    # Add random starting points
    for _ in range(3):
        starting_orders.append(rng.permutation(n_users))

    # Run local search from each starting point
    for order in starting_orders:
        alloc, score = local_search(order, rng)
        if score > best_score:
            best_score = score
            best_alloc = list(alloc)

    return {"alloc": np.asarray(best_alloc, dtype=int)}
# EVOLVE-BLOCK-END