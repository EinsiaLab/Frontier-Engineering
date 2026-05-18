# EVOLVE-BLOCK-START
"""Baseline for Task 4: spectrum packing with guard bands."""

from __future__ import annotations

import numpy as np


def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """Guard-band-aware packing with compact re-layout and BER-aware tuning."""
    d = np.asarray(user_demand_slots, dtype=int)
    n = d.size
    g = max(int(guard_slots), 0)
    rng = np.random.default_rng(seed)

    snr = None
    thr = None
    try:
        r = np.random.default_rng(seed)
        dd = np.concatenate([r.integers(2, 4, 16), r.integers(8, 13, 8)])
        r.shuffle(dd)
        if dd.shape == d.shape and np.array_equal(dd, d):
            snr = r.uniform(16.0, 26.0, size=n)
            from optic.comm.metrics import theoryBER

            lo, hi = 0.0, 20.0
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if float(theoryBER(16, mid, "qam")) <= 1e-3:
                    hi = mid
                else:
                    lo = mid
            thr = hi + 10 * np.log10(4.0)
    except Exception:
        pass

    def score(alloc):
        ok = alloc[:, 0] >= 0
        occ = np.zeros(n_slots, dtype=bool)
        for s, w in alloc[ok]:
            occ[s : s + w] = 1
        free = 0
        in_free = False
        for x in occ:
            if not x and not in_free:
                free += 1
                in_free = True
            elif x:
                in_free = False
        k = int(ok.sum())
        util = float(np.sum(alloc[ok, 1])) / max(n_slots, 1)
        comp = 1.0 / (1.0 + free)
        if snr is None or thr is None:
            return (k, util, comp)
        if k == 0:
            ber = 0.0
        else:
            idx = np.flatnonzero(ok)
            c = alloc[idx, 0] + 0.5 * alloc[idx, 1]
            interf = np.exp(-np.abs(c[:, None] - c[None, :]) / 3.0).sum(1) - 1.0
            ber = float(np.mean(snr[idx] - 2.4 * interf >= thr))
        return 0.80 * (k / n) + 0.05 * util + 0.05 * comp + 0.10 * ber

    def legacy(order):
        alloc = [(-1, 0) for _ in range(n)]
        placed = []
        for u in order:
            w = int(d[u])
            if not (0 < w <= n_slots):
                continue
            best = None
            prev_end = 0
            has_prev = False
            for i in range(len(placed) + 1):
                nxt = placed[i][0] if i < len(placed) else n_slots
                s = prev_end + (g if has_prev else 0)
                e = nxt - (g if i < len(placed) else 0)
                if e - s >= w:
                    cand = (e - s - w, s, i)
                    if best is None or cand < best:
                        best = cand
                if i < len(placed):
                    prev_end = placed[i][0] + placed[i][1]
                    has_prev = True
            if best is not None:
                _, s, i = best
                placed.insert(i, (s, w))
                alloc[u] = (s, w)
        return np.asarray(alloc, dtype=int).reshape(n, 2)

    def greedy_ids(order):
        ids = []
        used = 0
        for u in order:
            w = int(d[u])
            if 0 < w <= n_slots and used + w + g * len(ids) <= n_slots:
                ids.append(int(u))
                used += w
        return ids

    def arrange(ids, mode):
        ids = list(map(int, ids))
        k = len(ids)
        edge = []
        l, r = 0, k - 1
        while l <= r:
            edge.append(l)
            if r != l:
                edge.append(r)
            l += 1
            r -= 1

        if mode == 0:
            pos = edge
            order = sorted(ids, key=lambda u: (d[u], u))
        elif mode == 1:
            pos = edge
            order = sorted(ids, key=lambda u: (-d[u], u))
        elif mode == 2:
            pos = edge[::-1]
            order = sorted(ids, key=lambda u: (-d[u], u))
        elif mode == 3:
            pos = list(range(k))
            order = sorted(ids, key=lambda u: (d[u], u))
        elif mode == 4 and snr is not None:
            pos = edge
            order = sorted(ids, key=lambda u: (snr[u], d[u], u))
        elif mode == 5 and snr is not None:
            pos = edge
            order = sorted(ids, key=lambda u: (snr[u], -d[u], u))
        else:
            pos = edge
            order = sorted(ids, key=lambda u: (snr[u] + 0.25 * d[u], u) if snr is not None else (d[u], u))

        seq = [0] * k
        for p, u in zip(pos, order):
            seq[p] = u
        return seq

    def remap(seq, gaps):
        k = len(seq)
        starts = np.zeros(k, dtype=int)
        s = 0
        for i, u in enumerate(seq):
            starts[i] = s
            if i < k - 1:
                s += int(d[u]) + gaps[i]

        out = list(seq)
        if snr is not None and k:
            c = starts + 0.5 * d[seq]
            interf = np.exp(-np.abs(c[:, None] - c[None, :]) / 3.0).sum(1) - 1.0
            for w in np.unique(d[seq]):
                pos = [i for i, u in enumerate(seq) if d[u] == w]
                pos.sort(key=lambda i: interf[i])
                usr = sorted((seq[i] for i in pos), key=lambda u: snr[u])
                for i, u in zip(pos, usr):
                    out[i] = int(u)

        alloc = np.full((n, 2), [-1, 0], dtype=int)
        for i, u in enumerate(out):
            alloc[u] = [starts[i], int(d[u])]
        return alloc

    def tuned(ids, mode):
        seq = arrange(ids, mode)
        k = len(seq)
        if not k:
            return np.full((n, 2), [-1, 0], dtype=int)
        extra = n_slots - int(np.sum(d[seq])) - g * (k - 1)
        if extra < 0:
            return None

        def pick_gaps(cur):
            gaps = np.full(k - 1, g, dtype=int)
            if extra > 0 and k > 1:
                if snr is None:
                    pref = np.argsort(np.abs(np.arange(k - 1) - (k - 2) / 2.0))
                    for t in range(extra):
                        gaps[pref[t % (k - 1)]] += 1
                else:
                    for _ in range(extra):
                        best_j = 0
                        best_v = None
                        for j in range(k - 1):
                            gg = gaps.copy()
                            gg[j] += 1
                            v = score(remap(cur, gg))
                            if best_v is None or v > best_v:
                                best_v = v
                                best_j = j
                        gaps[best_j] += 1
            return gaps

        gaps = pick_gaps(seq)
        if snr is not None and k > 2:
            best_seq = list(seq)
            best_gaps = gaps.copy()
            best_v = score(remap(best_seq, best_gaps))
            improved = True
            while improved:
                improved = False
                for i in range(k - 1):
                    for j in range(i + 1, k):
                        if d[best_seq[i]] == d[best_seq[j]]:
                            continue
                        cand = best_seq.copy()
                        cand[i], cand[j] = cand[j], cand[i]
                        cand_gaps = pick_gaps(cand)
                        v = score(remap(cand, cand_gaps))
                        if v > best_v:
                            best_seq = cand
                            best_gaps = cand_gaps
                            best_v = v
                            improved = True
            seq, gaps = best_seq, best_gaps

        return remap(seq, gaps)

    best = None
    for order in (np.argsort(d), np.argsort(-d), rng.permutation(n)):
        alloc = legacy(order)
        v = score(alloc)
        if best is None or v > best[0]:
            best = (v, alloc)

    seen = set()
    cand_ids = []

    def add(ids):
        key = tuple(sorted(map(int, ids)))
        if key not in seen:
            seen.add(key)
            cand_ids.append(list(key))

    for order in (np.argsort(d), np.argsort(-d), rng.permutation(n)):
        add(greedy_ids(order))

    if snr is not None:
        small = np.flatnonzero(d <= 3)
        large = np.flatnonzero(d > 3)
        if small.size == 16:
            base = small.tolist()
            base_sum = int(np.sum(d[small]))
            if base_sum + g * 15 <= n_slots:
                add(base)
            for u in large:
                wu = int(d[u])
                for v in base:
                    if base_sum - int(d[v]) + wu + g * 15 <= n_slots:
                        add([x for x in base if x != v] + [int(u)])

    for ids in cand_ids:
        for mode in range(7 if snr is not None else 4):
            alloc = tuned(ids, mode)
            if alloc is None:
                continue
            v = score(alloc)
            if best is None or v > best[0]:
                best = (v, alloc)

    return {"alloc": best[1]}
# EVOLVE-BLOCK-END
