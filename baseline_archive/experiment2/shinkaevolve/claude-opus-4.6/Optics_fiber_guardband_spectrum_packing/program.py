# EVOLVE-BLOCK-START
"""Fast spectrum packing with guard bands – BER-aware, multi-strategy search."""

from __future__ import annotations

import numpy as np
import time


def _place_order(order, demands, n_slots, guard_slots, mode='first_fit', extra_guard=0):
    """Place users in given order using gap-list tracking.

    extra_guard: additional spacing beyond minimum guard_slots to improve BER.
    """
    n_users = len(demands)
    alloc_starts = [-1] * n_users
    alloc_widths = [0] * n_users
    gaps = [(0, n_slots)]
    n_accepted = 0
    max_slot = 0
    total_width = 0
    g = guard_slots + extra_guard

    for idx in range(len(order)):
        u = order[idx]
        width = demands[u]
        if width <= 0 or width > n_slots:
            continue

        best_pos = -1
        best_gap_idx = -1
        best_metric = n_slots + 1 if mode in ('first_fit', 'best_fit') else -1

        for gi in range(len(gaps)):
            gs, ge = gaps[gi]
            gap_len = ge - gs

            eff_start = gs if gs == 0 else gs + g
            eff_end = ge if ge == n_slots else ge - g

            if eff_end - eff_start < width:
                # Try with minimum guard as fallback
                if extra_guard > 0:
                    eff_start2 = gs if gs == 0 else gs + guard_slots
                    eff_end2 = ge if ge == n_slots else ge - guard_slots
                    if eff_end2 - eff_start2 < width:
                        continue
                    candidate_pos = eff_start2
                    candidate_gap_len = gap_len
                else:
                    continue
            else:
                if mode == 'spread':
                    candidate_pos = (eff_start + eff_end - width) // 2
                else:
                    candidate_pos = eff_start
                candidate_gap_len = gap_len

            if mode == 'first_fit':
                best_pos = candidate_pos
                best_gap_idx = gi
                break
            elif mode == 'best_fit':
                if candidate_gap_len < best_metric:
                    best_metric = candidate_gap_len
                    best_pos = candidate_pos
                    best_gap_idx = gi
            elif mode == 'worst_fit' or mode == 'spread':
                if candidate_gap_len > best_metric:
                    best_metric = candidate_gap_len
                    best_pos = candidate_pos
                    best_gap_idx = gi

        if best_pos >= 0:
            s = best_pos
            e = s + width
            alloc_starts[u] = s
            alloc_widths[u] = width
            n_accepted += 1
            if e > max_slot:
                max_slot = e
            total_width += width

            gs, ge = gaps[best_gap_idx]
            gaps.pop(best_gap_idx)
            if s > gs:
                gaps.insert(best_gap_idx, (gs, s))
                best_gap_idx += 1
            if e < ge:
                gaps.insert(best_gap_idx, (e, ge))

    if max_slot > 0 and total_width > 0:
        compactness = total_width / max_slot
    else:
        compactness = 0.0

    alloc = list(zip(alloc_starts, alloc_widths))
    return n_accepted, compactness, alloc


def _score_solution(n_accepted, compactness, alloc, n_users, n_slots, guard_slots):
    """Score that closely approximates the actual evaluation metric.

    Final score = 0.35*acceptance + 0.25*utilization + 0.15*compactness + 0.25*ber_pass
    """
    if n_accepted == 0:
        return -1.0

    intervals = []
    total_width = 0
    for i in range(len(alloc)):
        s, w = alloc[i]
        if s >= 0 and w > 0:
            intervals.append((s, s + w))
            total_width += w

    n_alloc = len(intervals)
    if n_alloc == 0:
        return -1.0

    intervals.sort()
    max_slot = intervals[-1][1]

    # Estimate BER pass ratio based on spacing to neighbors
    n_pass = 0.0
    for i in range(n_alloc):
        min_neighbor_gap = 9999
        if i > 0:
            gap = intervals[i][0] - intervals[i - 1][1]
            if gap < min_neighbor_gap:
                min_neighbor_gap = gap
        if i < n_alloc - 1:
            gap = intervals[i + 1][0] - intervals[i][1]
            if gap < min_neighbor_gap:
                min_neighbor_gap = gap
        # BER model: larger gaps -> better SNR -> lower BER
        if min_neighbor_gap >= guard_slots * 3:
            n_pass += 1.0
        elif min_neighbor_gap >= guard_slots * 2:
            n_pass += 0.9
        elif min_neighbor_gap >= guard_slots:
            n_pass += 0.6
        else:
            n_pass += 0.1

    ber_pass = n_pass / n_alloc
    acceptance = n_accepted / n_users
    utilization = total_width / n_slots if n_slots > 0 else 0.0
    comp = total_width / max_slot if max_slot > 0 else 0.0

    score = 0.35 * acceptance + 0.25 * utilization + 0.15 * comp + 0.25 * ber_pass
    return score


def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """Spectrum packing with BER-aware placement and intensive local search."""
    start_time = time.time()
    time_limit = 10.0

    d = np.asarray(user_demand_slots, dtype=int)
    n_users = d.size
    demands = d.tolist()
    rng = np.random.RandomState(seed)

    if n_users == 0:
        return {"alloc": np.zeros((0, 2), dtype=int)}

    # Precompute orderings
    sorted_desc = np.argsort(-d, kind='mergesort').tolist()
    sorted_asc = np.argsort(d, kind='mergesort').tolist()

    candidates = []
    # 1. Largest first (FFD)
    candidates.append(sorted_desc[:])
    # 2. Smallest first
    candidates.append(sorted_asc[:])
    # 3. Original order
    candidates.append(list(range(n_users)))
    # 4. Interleave large-small
    interleaved = []
    for i in range(n_users):
        if i % 2 == 0:
            interleaved.append(sorted_desc[i // 2])
        else:
            interleaved.append(sorted_asc[i // 2])
    candidates.append(interleaved[:])
    # 5. Reverse interleave
    interleaved2 = []
    for i in range(n_users):
        if i % 2 == 0:
            interleaved2.append(sorted_asc[i // 2])
        else:
            interleaved2.append(sorted_desc[i // 2])
    candidates.append(interleaved2[:])
    # 6. Medium first (sort by distance from median)
    median_d = np.median(d)
    dist_from_med = np.abs(d - median_d)
    candidates.append(np.argsort(dist_from_med, kind='mergesort').tolist())
    candidates.append(np.argsort(-dist_from_med, kind='mergesort').tolist())

    # 7. Random permutations
    n_random = min(60, max(25, n_users * 3))
    for _ in range(n_random):
        candidates.append(rng.permutation(n_users).tolist())

    modes = ['first_fit', 'best_fit']
    extra_guards = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    best_score = -1.0
    best_alloc = None
    best_order = None
    best_mode = 'first_fit'
    best_eg = 0

    # Phase 1: Evaluate candidates
    for order in candidates:
        if time.time() - start_time > time_limit * 0.2:
            break
        for mode in modes:
            for eg in extra_guards:
                n_acc, comp, alloc = _place_order(order, demands, n_slots, guard_slots, mode=mode, extra_guard=eg)
                sc = _score_solution(n_acc, comp, alloc, n_users, n_slots, guard_slots)
                if sc > best_score:
                    best_score = sc
                    best_alloc = alloc
                    best_order = list(order)
                    best_mode = mode
                    best_eg = eg

    # Phase 2: Local search on best order
    improved = True
    iteration = 0
    while improved and iteration < 300:
        if time.time() - start_time > time_limit * 0.45:
            break
        improved = False
        iteration += 1

        for i in range(n_users - 1):
            if time.time() - start_time > time_limit * 0.45:
                break
            best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
            n_acc, comp, alloc = _place_order(best_order, demands, n_slots, guard_slots, mode=best_mode, extra_guard=best_eg)
            sc = _score_solution(n_acc, comp, alloc, n_users, n_slots, guard_slots)
            if sc > best_score:
                best_score = sc
                best_alloc = alloc
                improved = True
            else:
                best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]

        # Random swaps
        n_rand = min(n_users * 4, 200)
        for _ in range(n_rand):
            if time.time() - start_time > time_limit * 0.45:
                break
            i = rng.randint(0, n_users)
            j = rng.randint(0, n_users)
            if i != j:
                best_order[i], best_order[j] = best_order[j], best_order[i]
                n_acc, comp, alloc = _place_order(best_order, demands, n_slots, guard_slots, mode=best_mode, extra_guard=best_eg)
                sc = _score_solution(n_acc, comp, alloc, n_users, n_slots, guard_slots)
                if sc > best_score:
                    best_score = sc
                    best_alloc = alloc
                    improved = True
                else:
                    best_order[i], best_order[j] = best_order[j], best_order[i]

    # Phase 3: Try all modes/extra_guards on best order
    for mode in modes:
        for eg in extra_guards:
            if time.time() - start_time > time_limit * 0.50:
                break
            n_acc, comp, alloc = _place_order(best_order, demands, n_slots, guard_slots, mode=mode, extra_guard=eg)
            sc = _score_solution(n_acc, comp, alloc, n_users, n_slots, guard_slots)
            if sc > best_score:
                best_score = sc
                best_alloc = alloc
                best_mode = mode
                best_eg = eg

    # Phase 4: Perturbation with mode exploration
    for p in range(3000):
        if time.time() - start_time > time_limit * 0.88:
            break
        trial_order = list(best_order)
        portion_size = max(2, rng.randint(2, max(3, n_users // 2 + 1)))
        si = rng.randint(0, max(1, n_users - portion_size + 1))
        subset = trial_order[si:si + portion_size]
        rng.shuffle(subset)
        trial_order[si:si + portion_size] = subset

        # Try current best mode/eg first
        n_acc, comp, alloc = _place_order(trial_order, demands, n_slots, guard_slots, mode=best_mode, extra_guard=best_eg)
        sc = _score_solution(n_acc, comp, alloc, n_users, n_slots, guard_slots)
        if sc > best_score:
            best_score = sc
            best_alloc = alloc
            best_order = trial_order
            # Explore other configs on improved order
            for mode in modes:
                for eg in extra_guards:
                    if time.time() - start_time > time_limit * 0.88:
                        break
                    n_acc2, comp2, alloc2 = _place_order(best_order, demands, n_slots, guard_slots, mode=mode, extra_guard=eg)
                    sc2 = _score_solution(n_acc2, comp2, alloc2, n_users, n_slots, guard_slots)
                    if sc2 > best_score:
                        best_score = sc2
                        best_alloc = alloc2
                        best_mode = mode
                        best_eg = eg

    # Phase 5: Final local search
    improved = True
    iteration = 0
    while improved and iteration < 200:
        if time.time() - start_time > time_limit * 0.95:
            break
        improved = False
        iteration += 1
        for i in range(n_users - 1):
            if time.time() - start_time > time_limit * 0.95:
                break
            best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
            n_acc, comp, alloc = _place_order(best_order, demands, n_slots, guard_slots, mode=best_mode, extra_guard=best_eg)
            sc = _score_solution(n_acc, comp, alloc, n_users, n_slots, guard_slots)
            if sc > best_score:
                best_score = sc
                best_alloc = alloc
                improved = True
            else:
                best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]

    return {"alloc": np.asarray(best_alloc, dtype=int)}
# EVOLVE-BLOCK-END