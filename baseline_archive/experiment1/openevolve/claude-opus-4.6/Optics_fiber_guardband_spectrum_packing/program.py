# EVOLVE-BLOCK-START
"""Spectrum packing with local search over user ordering."""
from __future__ import annotations
import numpy as np
import time

def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    d = np.asarray(user_demand_slots, dtype=int)
    n = d.size

    def _pack(order):
        """Best-fit packing given a user ordering."""
        alloc_s = np.full(n, -1, dtype=np.int32)
        alloc_w = np.zeros(n, dtype=np.int32)
        occ = np.zeros(n_slots, dtype=np.int8)
        for u in order:
            w = int(d[u])
            if w <= 0 or w > n_slots:
                continue
            # Find gaps
            best_pos, best_gap = -1, n_slots + 1
            i = 0
            while i < n_slots:
                if occ[i]:
                    i += 1
                    continue
                # Found start of a gap
                gs = i
                while i < n_slots and not occ[i]:
                    i += 1
                gl = i - gs  # gap length
                # Check if width fits with guards
                lg = guard_slots if gs > 0 else 0
                rg = guard_slots if (gs + gl) < n_slots else 0
                if gl < w + lg + rg:
                    continue
                # Best-fit: prefer smallest sufficient gap
                if gl < best_gap:
                    best_gap = gl
                    best_pos = gs + lg  # place right after left guard
            if best_pos >= 0:
                occ[best_pos:best_pos + w] = 1
                alloc_s[u] = best_pos
                alloc_w[u] = w
        return alloc_s, alloc_w

    def _proxy(alloc_s, alloc_w):
        acc = int(np.sum(alloc_s >= 0))
        used = int(np.sum(alloc_w[alloc_s >= 0]))
        occ = np.zeros(n_slots, dtype=np.int8)
        for i in range(n):
            if alloc_s[i] >= 0:
                occ[alloc_s[i]:alloc_s[i]+alloc_w[i]] = 1
        fb = 0; inf = False
        for k in range(n_slots):
            if not occ[k] and not inf: fb += 1; inf = True
            elif occ[k]: inf = False
        # BER proxy: penalize tightly packed neighbors
        accepted_idx = [i for i in range(n) if alloc_s[i] >= 0]
        ber_fail = 0
        for i in accepted_idx:
            ci = alloc_s[i] + 0.5 * alloc_w[i]
            interf = 0.0
            for j in accepted_idx:
                if i == j: continue
                cj = alloc_s[j] + 0.5 * alloc_w[j]
                interf += np.exp(-abs(ci - cj) / 3.0)
            if interf > 3.5:
                ber_fail += 1
        ber_est = 1.0 - ber_fail / max(len(accepted_idx), 1) if accepted_idx else 0.0
        return 0.80*(acc/n) + 0.05*(used/n_slots) + 0.05/(1.0+fb) + 0.10*ber_est

    rng = np.random.default_rng(seed)
    best_s, best_w, best_sc = None, None, -1.0

    # Enumerate which large users to include (2^8=256 combos)
    small_idx = [i for i in range(n) if d[i] <= 4]
    large_idx = [i for i in range(n) if d[i] > 4]
    n_large = len(large_idx)

    for mask in range(1 << n_large):
        chosen_large = [large_idx[b] for b in range(n_large) if mask & (1 << b)]
        total_demand = sum(d[i] for i in small_idx) + sum(d[i] for i in chosen_large)
        n_accepted = len(small_idx) + len(chosen_large)
        min_space = total_demand + max(0, n_accepted - 1) * guard_slots
        if min_space > n_slots:
            continue
        # Try multiple orderings of this subset
        subset = small_idx + chosen_large
        rejected = [i for i in range(n) if i not in set(subset)]
        for trial in range(6):
            if trial == 0:
                order = sorted(subset, key=lambda x: d[x]) + rejected
            elif trial == 1:
                order = sorted(subset, key=lambda x: -d[x]) + rejected
            elif trial == 2:
                # large first then small, all before rejected
                order = sorted(chosen_large, key=lambda x: -d[x]) + sorted(small_idx, key=lambda x: d[x]) + rejected
            else:
                perm = list(subset)
                rng.shuffle(perm)
                order = perm + rejected
            s, w = _pack(order)
            sc = _proxy(s, w)
            if sc > best_sc:
                best_s, best_w, best_sc = s.copy(), w.copy(), sc
                best_order = list(order)

    # SA refinement
    cur_order = best_order[:]
    cur_s, cur_w, cur_sc = best_s.copy(), best_w.copy(), best_sc
    t0 = time.time()
    for step in range(30000):
        if time.time() - t0 > 35.0:
            break
        cand = cur_order[:]
        r = rng.random()
        if r < 0.45:
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            cand[i], cand[j] = cand[j], cand[i]
        elif r < 0.70:
            i, j = sorted([int(rng.integers(0, n)), int(rng.integers(0, n))])
            if j > i:
                v = cand[i]; cand[i:j] = cand[i+1:j+1]; cand[j] = v
        elif r < 0.85:
            i, j = sorted([int(rng.integers(0, n)), int(rng.integers(0, n))])
            sub = cand[i:j+1]; rng.shuffle(sub); cand[i:j+1] = list(sub)
        else:
            lp = [p for p in range(n) if d[cand[p]] > 4]
            if lp:
                p = lp[int(rng.integers(0, len(lp)))]
                v = cand[p]; del cand[p]; cand.append(v)
        cs, cw = _pack(cand)
        csc = _proxy(cs, cw)
        temp = max(0.005, 0.12*(1.0 - step/25000))
        if csc > cur_sc or rng.random() < np.exp(min((csc-cur_sc)/max(temp, 1e-9), 10)):
            cur_order, cur_s, cur_w, cur_sc = cand, cs, cw, csc
            if csc > best_sc:
                best_s, best_w, best_sc = cs.copy(), cw.copy(), csc
                best_order = cand[:]
        if step % 5000 == 0 and step > 0 and cur_sc < best_sc - 0.003:
            cur_order = best_order[:]
            cur_s, cur_w, cur_sc = best_s.copy(), best_w.copy(), best_sc

    alloc = np.stack([best_s, best_w], axis=1)
    return {"alloc": alloc}
# EVOLVE-BLOCK-END
