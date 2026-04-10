# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: WDM channel + power allocation.

Simple engineering baseline:
- assign users to channels in index order
- split total launch power equally among assigned channels
"""

from __future__ import annotations

import numpy as np


def allocate_wdm(
    user_demands_gbps,
    channel_centers_hz,
    total_power_dbm,
    pmin_dbm=-8.0,
    pmax_dbm=3.0,
    target_ber=1e-3,
    seed=0,
):
    """Allocate one channel per user and per-channel power.

    Parameters
    ----------
    user_demands_gbps : array-like, shape (U,)
        Requested data rate per user.
    channel_centers_hz : array-like, shape (C,)
        Available fixed WDM grid center frequencies.
    total_power_dbm : float
        Total launch power budget (dBm, summed across active channels).
    pmin_dbm, pmax_dbm : float
        Per-channel power limits.

    Returns
    -------
    dict with keys:
      - assignment: np.ndarray shape (U,), channel index or -1
      - power_dbm: np.ndarray shape (C,), per-channel launch power
    """
    demands = np.asarray(user_demands_gbps, dtype=float)
    channel_centers_hz = np.asarray(channel_centers_hz, dtype=float)

    n_users = len(demands)
    n_channels = channel_centers_hz.size

    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)

    if n_users == 0 or n_channels == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    # Sort users by demand descending - prioritize high-demand users
    user_order = np.argsort(-demands)

    # Sort channels by frequency to spread assignments across spectrum
    chan_order = np.argsort(channel_centers_hz)

    n_served = min(n_users, n_channels)

    # Spread users evenly across available channels for better spectral utilization
    # Assign users to evenly-spaced channels across the spectrum
    if n_served <= n_channels:
        # Pick evenly spaced channels
        indices = np.round(np.linspace(0, n_channels - 1, n_served)).astype(int)
        selected_chans = chan_order[indices]
    else:
        selected_chans = chan_order[:n_served]

    for i in range(n_served):
        assignment[user_order[i]] = selected_chans[i]

    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}

    # Power allocation: proportional to demand, within constraints
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    pmin_lin = 10 ** (float(pmin_dbm) / 10.0)
    pmax_lin = 10 ** (float(pmax_dbm) / 10.0)

    # Budget for inactive channels at pmin
    inactive_count = n_channels - used.size
    inactive_lin = inactive_count * pmin_lin
    active_budget_lin = max(total_power_lin - inactive_lin, used.size * pmin_lin)

    # Compute demand per channel
    chan_demand = np.zeros(n_channels)
    for u in range(n_users):
        ch = assignment[u]
        if ch >= 0:
            chan_demand[ch] += demands[u]

    # Allocate power proportional to demand on active channels
    total_demand_active = chan_demand[used].sum()
    if total_demand_active > 0:
        fracs = chan_demand[used] / total_demand_active
    else:
        fracs = np.ones(used.size) / used.size

    alloc_lin = fracs * active_budget_lin
    # Clip to per-channel limits
    alloc_lin = np.clip(alloc_lin, pmin_lin, pmax_lin)

    # Re-normalize to respect total budget
    total_alloc = alloc_lin.sum() + inactive_lin
    if total_alloc > total_power_lin:
        excess = total_alloc - total_power_lin
        # Scale down active channels proportionally
        scale = (active_budget_lin - excess) / alloc_lin.sum() if alloc_lin.sum() > 0 else 1.0
        alloc_lin = np.clip(alloc_lin * scale, pmin_lin, pmax_lin)

    for i, ch in enumerate(used):
        p_dbm = 10.0 * np.log10(max(alloc_lin[i], 1e-12))
        power_dbm[ch] = float(np.clip(p_dbm, pmin_dbm, pmax_dbm))

    # Local search to refine assignment and power
    nfl, isc, idc, csc = 2e-3, 0.12, 0.9, 28.0

    def _renorm(p, a):
        p = p.copy(); ac = np.unique(a[a >= 0])
        if ac.size == 0: return p
        tgt = max(total_power_lin - (n_channels - ac.size) * pmin_lin, ac.size * pmin_lin)
        pl = 10 ** (p[ac] / 10.0); s = pl.sum()
        if s > 0: pl *= tgt / s
        p[ac] = np.clip(10.0 * np.log10(np.clip(pl, 1e-12, None)), pmin_dbm, pmax_dbm)
        if np.sum(10 ** (p / 10.0)) > total_power_lin * 1.000001:
            al = 10 ** (p[ac] / 10.0); il = np.sum(10 ** (p / 10.0)) - al.sum()
            al *= min(max(total_power_lin - il, 1e-12) / max(al.sum(), 1e-12), 1.0)
            p[ac] = np.clip(10.0 * np.log10(np.clip(al, 1e-12, None)), pmin_dbm, pmax_dbm)
        return p

    def _obj(a, p):
        uidx = np.where(a >= 0)[0]
        if uidx.size == 0: return -1e12
        chs = a[uidx].astype(int)
        plin = 10 ** (p / 10.0)
        sigs = plin[chs]
        # Vectorized interference
        ch_diff = np.abs(chs[:, None] - chs[None, :])
        itf_mat = plin[chs][None, :] * np.exp(-ch_diff / idc)
        np.fill_diagonal(itf_mat, 0.0)
        itfs = itf_mat.sum(axis=1)
        snr_lin = sigs / (nfl + isc * itfs)
        snr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
        caps = csc * np.log2(1.0 + np.maximum(snr_lin, 1e-12))
        sat_served = np.minimum(caps / np.maximum(demands[uidx], 1e-9), 1.0)
        # Unserved users contribute 0 to satisfaction
        ds = float(sat_served.sum() / n_users)
        bp = float(np.mean(snr_db > 9.8))
        su = float(uidx.size / n_channels)
        st = float(np.clip((np.mean(snr_db) - 5.0) / 20.0, 0.0, 1.0))
        return 0.35 * ds + 0.40 * bp + 0.05 * su + 0.20 * st

    rng = np.random.default_rng(seed + 42)
    ba, bp_ = assignment.copy(), power_dbm.copy(); bs = _obj(ba, bp_)
    for _ in range(2000):
        ca, cp = ba.copy(), bp_.copy(); r = rng.random()
        if r < 0.20:
            us = np.where(ca >= 0)[0]
            if us.size > 0:
                free = [c for c in range(n_channels) if c not in set(ca[us].tolist())]
                if free: ca[int(rng.choice(us))] = int(rng.choice(free))
        elif r < 0.40:
            us = np.where(ca >= 0)[0]
            if us.size >= 2:
                u1, u2 = rng.choice(us, 2, replace=False); ca[u1], ca[u2] = ca[u2], ca[u1]
        elif r < 0.60:
            ac = np.unique(ca[ca >= 0])
            if ac.size > 0: cp[int(rng.choice(ac))] += rng.normal(0.0, 0.5)
        elif r < 0.75:
            ac = np.unique(ca[ca >= 0])
            if ac.size > 1:
                c1, c2 = rng.choice(ac, 2, replace=False); d = rng.uniform(0.2, 1.2); cp[c1] += d; cp[c2] -= d
        elif r < 0.85:
            ac = np.unique(ca[ca >= 0])
            if ac.size > 0: cp[ac] += rng.normal(0.0, 0.15, size=ac.size)
        elif r < 0.93:
            ac = np.unique(ca[ca >= 0])
            if ac.size > 0: avg = np.mean(cp[ac]); cp[ac] = cp[ac] * 0.5 + avg * 0.5
        else:
            ac = np.unique(ca[ca >= 0])
            if ac.size > 2: cp[ac[[0, -1]]] += rng.uniform(0.1, 0.5); cp[ac[len(ac)//2]] -= rng.uniform(0.1, 0.3)
        cp = _renorm(cp, ca); s = _obj(ca, cp)
        if s > bs: bs, ba, bp_ = s, ca, cp

    return {"assignment": ba, "power_dbm": bp_}
# EVOLVE-BLOCK-END
