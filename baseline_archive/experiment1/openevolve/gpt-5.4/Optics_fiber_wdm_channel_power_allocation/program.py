# EVOLVE-BLOCK-START
"""Heuristic WDM channel and power allocator."""

from __future__ import annotations

import numpy as np

try:
    from optic.comm.metrics import theoryBER
except Exception:
    theoryBER = None


def allocate_wdm(
    user_demands_gbps,
    channel_centers_hz,
    total_power_dbm,
    pmin_dbm=-8.0,
    pmax_dbm=3.0,
    target_ber=1e-3,
    seed=0,
):
    """Map users to channels and choose per-channel power."""
    user_demands_gbps = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)

    n_users, n_channels = user_demands_gbps.size, channel_centers_hz.size
    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)

    n_served = min(n_users, n_channels)
    if n_served == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    base = 10 ** (float(pmin_dbm) / 10.0)
    cap = 10 ** (float(pmax_dbm) / 10.0)
    total = 10 ** (float(total_power_dbm) / 10.0)
    extra = max(0.0, total - n_channels * base)
    easy = np.argsort(user_demands_gbps, kind="stable")
    k0 = min(n_served, max(1, int(np.ceil(0.55 * n_channels))))
    best_s = -1e9
    best_ch = np.array([], dtype=int)
    best_p = np.full(n_channels, base, dtype=float)

    def spread(raw):
        out = []
        for x in raw:
            x = int(np.clip(x, 0, n_channels - 1))
            if x not in out:
                out.append(x)
        for x in range(n_channels):
            if x not in out:
                out.append(x)
            if len(out) == len(raw):
                break
        return np.array(sorted(out), dtype=int)

    def pack(ch, w):
        lim = max(0.0, cap - base)
        amt = min(extra, lim * ch.size)
        w = np.maximum(np.asarray(w, dtype=float), 1e-12)
        e = amt * w / np.sum(w)
        for _ in range(ch.size):
            hi = e > lim
            if not np.any(hi):
                break
            over = float(np.sum(e[hi]) - lim * np.sum(hi))
            e[hi] = lim
            lo = ~hi
            if over <= 1e-12 or not np.any(lo):
                break
            ww = w[lo]
            e[lo] += over * ww / np.sum(ww)
        p = np.full(n_channels, base, dtype=float)
        p[ch] += e
        return p

    def score(ch, p):
        a = p[ch]
        d = np.abs(ch[:, None] - ch[None, :])
        g = np.exp(-d / 0.9)
        np.fill_diagonal(g, 0.0)
        snr = a / (2e-3 + 0.12 * g.dot(a))
        snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
        capu = np.sort(28.0 * np.log2(1.0 + np.maximum(snr, 1e-12)))[::-1]
        k = ch.size
        sat = float(np.sum(np.minimum(capu / np.maximum(user_demands_gbps[easy[:k]], 1e-9), 1.0)) / n_users)
        if np.min(snr_db) >= 10.0:
            ber_pass = 1.0
        elif theoryBER is None:
            ber_pass = float(np.mean(snr_db >= 9.5))
        else:
            ber_pass = float(np.mean([float(theoryBER(4, s - 3.010299956639812, "qam")) <= target_ber for s in snr_db]))
        snr_term = float(np.clip((np.mean(snr_db) - 5.0) / 20.0, 0.0, 1.0))
        return 0.35 * sat + 0.40 * ber_pass + 0.05 * k / n_channels + 0.20 * snr_term

    def consider(ch):
        nonlocal best_s, best_ch, best_p
        d = np.abs(ch[:, None] - ch[None, :])
        g = np.exp(-d / 0.9)
        np.fill_diagonal(g, 0.0)
        iso = 1.0 / (0.15 + np.sum(g, axis=1))
        for w in (np.ones(ch.size), np.sqrt(iso), iso, iso * iso, iso * iso * iso):
            p = pack(ch, w)
            s = score(ch, p)
            if s > best_s:
                best_s, best_ch, best_p = s, ch.copy(), p.copy()

    def refine(ch, p):
        s = score(ch, p)
        if ch.size < 2:
            return p, s
        for frac in (0.5, 0.2):
            while True:
                cur = p[ch]
                best_local = s
                best_q = None
                for i in range(ch.size):
                    give = cur[i] - base
                    if give <= 1e-12:
                        continue
                    for j in range(ch.size):
                        if i == j:
                            continue
                        take = cap - cur[j]
                        step = frac * min(give, take)
                        if step <= 1e-12:
                            continue
                        q = p.copy()
                        q[ch[i]] -= step
                        q[ch[j]] += step
                        cand = score(ch, q)
                        if cand > best_local + 1e-12:
                            best_local, best_q = cand, q
                if best_q is None:
                    break
                p, s = best_q, best_local
        return p, s

    alt0 = np.r_[np.arange(0, n_channels, 2), np.arange(n_channels - 1, -1, -2)]
    alt1 = np.r_[np.arange(1, n_channels, 2), np.arange(0, n_channels, 2)]

    for k in range(k0, n_served + 1):
        for raw in (
            np.rint(np.linspace(0, n_channels - 1, k)).astype(int),
            np.floor((np.arange(k) + 0.5) * n_channels / k).astype(int),
            alt0[:k],
            alt1[:k],
        ):
            consider(spread(raw))

    improved = True
    while improved and best_ch.size:
        improved = False
        free = [c for c in range(n_channels) if c not in set(best_ch.tolist())]
        prev = best_s
        for i in range(best_ch.size):
            for c in free:
                ch = best_ch.copy()
                ch[i] = c
                if np.unique(ch).size != ch.size:
                    continue
                consider(np.sort(ch))
        if best_s > prev + 1e-12:
            best_p, best_s = refine(best_ch, best_p)
            improved = True

    if best_ch.size:
        best_p, best_s = refine(best_ch, best_p)
        a = best_p[best_ch]
        d = np.abs(best_ch[:, None] - best_ch[None, :])
        g = np.exp(-d / 0.9)
        np.fill_diagonal(g, 0.0)
        capu = 28.0 * np.log2(1.0 + a / (2e-3 + 0.12 * g.dot(a)))
        assignment[easy[: best_ch.size]] = best_ch[np.argsort(-capu, kind="stable")]
        power_dbm = 10.0 * np.log10(np.maximum(best_p, 1e-12))

    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
