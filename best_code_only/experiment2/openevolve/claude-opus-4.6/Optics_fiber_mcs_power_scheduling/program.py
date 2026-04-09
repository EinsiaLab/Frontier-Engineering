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
    mcs_max = int(np.max(mcs_candidates))

    # DP knapsack approach matching oracle utility
    total_lin = 10 ** (float(total_power_dbm) / 10.0)
    cost_unit = 0.01  # mW granularity (finer for better resolution)
    cap = int(np.floor(total_lin / cost_unit + 1e-9))
    # Cap the DP budget dimension to keep computation tractable
    if cap > 1000:
        cost_unit = total_lin / 1000.0
        cap = 1000

    # Use finer power grid (0.125 dB) for better granularity than oracle's 0.5 dB
    power_levels_dbm = np.arange(pmin_dbm, pmax_dbm + 1e-12, 0.125)

    # Import the exact BER function used by the verifier
    try:
        import sys
        from pathlib import Path
        pr = Path(__file__).resolve().parents[2]
        if str(pr) not in sys.path:
            sys.path.insert(0, str(pr))
        from optic.comm.metrics import theoryBER as _theoryBER
        _has_optic = True
    except Exception:
        _has_optic = False
        from scipy.special import erfc as _erfc

    def _compute_utility(demand, quality_db, M, p_dbm):
        snr_db = quality_db + p_dbm
        ebn0_db = snr_db - 10.0 * np.log10(np.log2(M))
        if _has_optic:
            ber = max(float(_theoryBER(M, ebn0_db, "qam")), 0.0)
        else:
            ebn0_lin = 10 ** (ebn0_db / 10.0)
            if M == 4:
                ber = 0.5 * _erfc(np.sqrt(ebn0_lin))
            else:
                k = np.log2(M)
                ber = (2.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * _erfc(
                    np.sqrt(3.0 * k * ebn0_lin / (2.0 * (M - 1.0)))
                )
            ber = max(float(ber), 0.0)
        cap_thr = 32.0 * np.log2(M)
        reliability = 1.0 if ber <= target_ber else np.exp(-(ber - target_ber) * 15.0)
        achieved = min(demand, cap_thr * reliability)
        sat = min(achieved / max(demand, 1e-9), 1.0)
        ber_ok = 1.0 if ber <= target_ber else 0.0
        se = np.log2(M) / np.log2(max(mcs_max, 2))
        return 0.45 * sat + 0.40 * ber_ok + 0.15 * se

    # Build per-user option tables
    options_per_user = []
    for u in range(n_users):
        opts = []
        for M in mcs_candidates:
            for p_dbm in power_levels_dbm:
                cost_lin = 10 ** (p_dbm / 10.0)
                cost_int = int(np.ceil(cost_lin / cost_unit - 1e-12))
                util = _compute_utility(demands[u], quality[u], int(M), float(p_dbm))
                opts.append((cost_int, util, int(M), float(p_dbm)))
        # Prune dominated options (sort by cost, keep only increasing utility)
        opts.sort(key=lambda x: (x[0], -x[1]))
        pruned = []
        best_util = -1e18
        for item in opts:
            if item[1] > best_util + 1e-12:
                pruned.append(item)
                best_util = item[1]
        options_per_user.append(pruned)

    # DP solve
    dp = np.full((n_users + 1, cap + 1), -1e18, dtype=float)
    choice = np.zeros((n_users + 1, cap + 1), dtype=int)
    dp[0, :] = 0.0

    for i in range(1, n_users + 1):
        opts = options_per_user[i - 1]
        # Pre-extract costs and utilities for faster inner loop
        n_opts = len(opts)
        costs = np.array([o[0] for o in opts], dtype=int)
        utils = np.array([o[1] for o in opts], dtype=float)
        prev_dp = dp[i - 1]
        cur_dp = dp[i]
        cur_ch = choice[i]
        for b in range(cap + 1):
            best = -1e18
            best_k = 0
            for k in range(n_opts):
                c = costs[k]
                if c <= b:
                    v = prev_dp[b - c] + utils[k]
                    if v > best:
                        best = v
                        best_k = k
            cur_dp[b] = best
            cur_ch[b] = best_k

    # Backtrack
    b = int(np.argmax(dp[n_users, :]))
    mcs = np.zeros(n_users, dtype=int)
    power_dbm_out = np.zeros(n_users, dtype=float)
    for i in range(n_users, 0, -1):
        item = options_per_user[i - 1][choice[i, b]]
        c, _, M, p_dbm = item
        mcs[i - 1] = int(M)
        power_dbm_out[i - 1] = float(p_dbm)
        b -= c

    return {"mcs": mcs, "power_dbm": power_dbm_out}
# EVOLVE-BLOCK-END
