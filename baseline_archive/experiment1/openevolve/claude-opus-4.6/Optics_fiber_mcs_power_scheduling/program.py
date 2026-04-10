# EVOLVE-BLOCK-START
"""Baseline solver for Task 2: MCS + power scheduling."""

from __future__ import annotations

import numpy as np


def select_mcs_power(
    user_demands_gbps,
    channel_quality_db,
    total_power_dbm,
    mcs_candidates=(4, 16, 64),
    pmin_dbm=-8.0,
    pmax_dbm=4.0,
    target_ber=1e-3,
    seed=0,
):
    demands = np.asarray(user_demands_gbps, dtype=float)
    quality = np.asarray(channel_quality_db, dtype=float)
    mcs_candidates = np.asarray(mcs_candidates, dtype=int)

    n_users = demands.size

    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    sorted_mcs = np.sort(mcs_candidates)

    # DP-based optimization with finer power discretization than oracle
    # Use 0.125 dB steps for better granularity
    power_levels_dbm = np.arange(pmin_dbm, pmax_dbm + 1e-12, 0.125)
    cost_unit = 0.005  # mW (finer cost unit for finer power steps)
    budget_cap = int(np.floor(total_lin / cost_unit + 1e-9))
    # Cap budget_cap to keep DP tractable
    if budget_cap > 2000000:
        cost_unit = total_lin / 2000000.0
        budget_cap = 2000000

    mcs_max = int(np.max(sorted_mcs))

    # Import the same BER function used by verification for exact match
    import sys
    from pathlib import Path
    proj_root = Path(__file__).resolve().parents[3]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from optic.comm.metrics import theoryBER

    # Build per-user option table
    options_per_user = []
    for u in range(n_users):
        opts = []
        for m_val in sorted_mcs:
            m_int = int(m_val)
            for p_dbm in power_levels_dbm:
                cost_lin = 10 ** (p_dbm / 10.0)
                cost_int = int(np.ceil(cost_lin / cost_unit - 1e-12))
                snr_db = quality[u] + p_dbm
                ebn0_db = snr_db - 10.0 * np.log10(np.log2(m_int))
                ber_val = float(theoryBER(m_int, ebn0_db, "qam"))

                cap = 32.0 * np.log2(m_int)
                if ber_val <= target_ber:
                    reliability = 1.0
                else:
                    reliability = np.exp(-(ber_val - target_ber) * 15.0)
                achieved = min(demands[u], cap * reliability)

                sat = min(achieved / max(demands[u], 1e-9), 1.0)
                ber_ok = 1.0 if ber_val <= target_ber else 0.0
                se = np.log2(m_int) / np.log2(max(mcs_max, 2))
                # Match verification weights exactly
                util = 0.45 * sat + 0.40 * ber_ok + 0.15 * se
                # Tiny tiebreaker: prefer saving power budget for others
                util -= 1e-8 * cost_int

                opts.append((cost_int, util, m_int, float(p_dbm)))
        # Prune dominated options
        opts.sort(key=lambda x: (x[0], -x[1]))
        pruned = []
        best_util = -1e18
        for item in opts:
            if item[1] > best_util + 1e-12:
                pruned.append(item)
                best_util = item[1]
        options_per_user.append(pruned)

    # DP solve - use rolling arrays to save memory
    dp_prev = np.full(budget_cap + 1, -1e18, dtype=float)
    dp_curr = np.full(budget_cap + 1, -1e18, dtype=float)
    choice = np.zeros((n_users, budget_cap + 1), dtype=np.int16)
    dp_prev[:] = 0.0

    for i in range(n_users):
        opts = options_per_user[i]
        dp_curr[:] = -1e18
        for b in range(budget_cap + 1):
            best = -1e18
            best_k = 0
            for k, item in enumerate(opts):
                c, u_val, _, _ = item
                if c <= b:
                    v = dp_prev[b - c] + u_val
                    if v > best:
                        best = v
                        best_k = k
            dp_curr[b] = best
            choice[i, b] = best_k
        dp_prev[:] = dp_curr[:]

    b = int(np.argmax(dp_curr))
    mcs = np.zeros(n_users, dtype=int)
    power_dbm = np.zeros(n_users, dtype=float)

    for i in range(n_users - 1, -1, -1):
        item = options_per_user[i][choice[i, b]]
        c, _, m_val, p_dbm = item
        mcs[i] = int(m_val)
        power_dbm[i] = float(p_dbm)
        b -= c

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
