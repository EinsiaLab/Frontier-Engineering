# EVOLVE-BLOCK-START
"""Spectrum packing with local search and fragmentation-aware placement."""

from __future__ import annotations
import numpy as np

def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """Small-first best-fit with local search for maximum acceptance."""
    d = np.asarray(user_demand_slots, dtype=int)
    n_users = d.size
    rng = np.random.RandomState(seed)

    def count_frag(occ):
        blocks, in_free = 0, False
        for x in occ:
            if not x and not in_free:
                blocks += 1
                in_free = True
            elif x:
                in_free = False
        return blocks

    def pack(order):
        alloc = [(-1, 0) for _ in range(n_users)]
        occ = np.zeros(n_slots, dtype=bool)
        for u in order:
            w = int(d[u])
            if w <= 0 or w > n_slots:
                continue
            best_s, best_frag = -1, n_slots + 1
            for s in range(n_slots - w + 1):
                l, r = max(0, s - guard_slots), min(n_slots, s + w + guard_slots)
                if not np.any(occ[l:r]):
                    tmp = occ.copy()
                    tmp[s:s + w] = True
                    frag = count_frag(tmp)
                    if frag < best_frag:
                        best_s, best_frag = s, frag
            if best_s >= 0:
                occ[best_s:best_s + w] = True
                alloc[u] = (best_s, w)
        return alloc

    def score(a):
        return (sum(1 for s, _ in a if s >= 0), sum(w for _, w in a if w > 0))

    order = np.argsort(d)  # Small-first for high acceptance
    best_alloc = pack(order)
    best_sc = score(best_alloc)
    curr_order, curr_sc = order.copy(), best_sc

    for step in range(600):
        cand = curr_order.copy()
        i, j = rng.randint(0, n_users), rng.randint(0, n_users)
        cand[i], cand[j] = cand[j], cand[i]
        a = pack(cand)
        sc = score(a)
        temp = max(0.02, 0.12 * (1 - step / 600))
        if sc > curr_sc or rng.rand() < np.exp((sc[0] - curr_sc[0]) / temp):
            curr_order, curr_sc = cand, sc
            if sc > best_sc:
                best_sc, best_alloc = sc, a

    return {"alloc": np.asarray(best_alloc, dtype=int)}
# EVOLVE-BLOCK-END
