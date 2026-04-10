# EVOLVE-BLOCK-START
"""Compact deterministic spectrum packing."""
from __future__ import annotations
import numpy as np

def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    d = np.asarray(user_demand_slots, int); n = d.size
    best = np.full((n, 2), (-1, 0), int); best_key = (-1, -1, -1)
    rng = np.random.default_rng(seed)

    def run(order):
        occ = np.zeros(n_slots, bool)
        alloc = np.full((n, 2), (-1, 0), int)
        for u in order:
            w = int(d[u])
            if not 0 < w <= n_slots: 
                continue
            pos = -1; pick = None
            for s in range(n_slots - w + 1):
                l = max(0, s - guard_slots); r = min(n_slots, s + w + guard_slots)
                if occ[l:r].any(): 
                    continue
                x = occ.copy(); x[s:s + w] = 1
                cur = (((~x) & np.r_[True, x[:-1]]).sum(), min(s, n_slots - s - w), s)
                if pick is None or cur < pick:
                    pick, pos = cur, s
            if pos >= 0:
                occ[pos:pos + w] = 1
                alloc[u] = (pos, w)
        return alloc, occ

    orders = [np.argsort(d), np.argsort(-d), rng.permutation(n)]
    a = np.argsort(d); orders += [a[np.argsort(d[a] % max(1, guard_slots + 2), kind="stable")]]
    for order in orders:
        alloc, occ = run(order)
        key = ((alloc[:, 0] >= 0).sum(), alloc[:, 1].sum(), -((~occ) & np.r_[True, occ[:-1]]).sum())
        if key > best_key:
            best_key, best = key, alloc
    return {"alloc": best}
# EVOLVE-BLOCK-END
