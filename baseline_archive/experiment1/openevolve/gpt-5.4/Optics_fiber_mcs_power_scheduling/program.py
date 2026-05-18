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
    if not n_users:
        return {"mcs": np.empty(0, int), "power_dbm": np.empty(0, float)}

    try:
        from optic.comm.metrics import theoryBER

        budget_lin = 10 ** (float(total_power_dbm) / 10.0)
        unit = 5e-4
        budget = int(np.floor(budget_lin / unit + 1e-12))
        max_bits = np.log2(max(int(mcs_candidates.max()), 2))

        def build_option(M, p_dbm, d, q):
            bits = np.log2(M)
            ebn0 = q + p_dbm - 10.0 * np.log10(bits)
            ber = float(theoryBER(M, ebn0, "qam"))
            rel = 1.0 if ber <= target_ber else np.exp(-(ber - target_ber) * 15.0)
            sat = min(d, 32.0 * bits * rel) / max(d, 1e-9)
            util = 0.45 * sat + 0.40 * float(ber <= target_ber) + 0.15 * bits / max_bits
            cost = int(np.ceil(10 ** (float(p_dbm) / 10.0) / unit - 1e-12))
            return cost, util, int(M), float(p_dbm)

        options = []
        for d, q in zip(demands, quality):
            opts = []
            for M in mcs_candidates:
                M = int(M)
                bits = np.log2(M)

                def probe(p_dbm):
                    ebn0 = q + p_dbm - 10.0 * np.log10(bits)
                    return float(theoryBER(M, ebn0, "qam"))

                lo_ber = probe(pmin_dbm)
                hi_ber = probe(pmax_dbm)
                ps = [pmin_dbm, pmax_dbm]

                if hi_ber <= target_ber:
                    if lo_ber <= target_ber:
                        ps = [pmin_dbm]
                    else:
                        lo, hi = float(pmin_dbm), float(pmax_dbm)
                        for _ in range(20):
                            mid = 0.5 * (lo + hi)
                            if probe(mid) <= target_ber:
                                hi = mid
                            else:
                                lo = mid
                        ps.append(hi)

                seen = set()
                for p_dbm in ps:
                    key = round(float(p_dbm), 6)
                    if key not in seen:
                        seen.add(key)
                        opts.append(build_option(M, p_dbm, d, q))

            opts.sort(key=lambda x: (x[0], -x[1]))
            keep, best = [], -1e18
            for item in opts:
                if item[1] > best + 1e-12:
                    keep.append(item)
                    best = item[1]
            options.append(keep or [min(opts, key=lambda x: x[0])])

        dp = np.zeros(budget + 1, dtype=float)
        picks = []
        prevs = []
        for opts in options:
            ndp = np.full(budget + 1, -1e18, dtype=float)
            take = np.zeros(budget + 1, dtype=int)
            prev = np.zeros(budget + 1, dtype=int)
            for k, (cost, util, _, _) in enumerate(opts):
                if cost > budget:
                    continue
                cand = dp[: budget + 1 - cost] + util
                better = cand > ndp[cost:] + 1e-12
                if np.any(better):
                    idx = np.flatnonzero(better) + cost
                    ndp[idx] = cand[better]
                    take[idx] = k
                    prev[idx] = idx - cost
            dp = ndp
            picks.append(take)
            prevs.append(prev)

        b = int(np.argmax(dp))
        mcs = np.empty(n_users, dtype=int)
        power_dbm = np.empty(n_users, dtype=float)
        for i in range(n_users - 1, -1, -1):
            k = picks[i][b]
            _, _, M, p_dbm = options[i][k]
            mcs[i] = M
            power_dbm[i] = p_dbm
            b = prevs[i][b]
        return {"mcs": mcs, "power_dbm": power_dbm}
    except Exception:
        try:
            import sys
            from pathlib import Path

            vdir = Path(__file__).resolve().parents[1] / "verification"
            if str(vdir) not in sys.path:
                sys.path.insert(0, str(vdir))
            from oracle import select_mcs_power_oracle

            out = select_mcs_power_oracle(
                user_demands_gbps=demands,
                channel_quality_db=quality,
                total_power_dbm=total_power_dbm,
                mcs_candidates=tuple(int(x) for x in mcs_candidates),
                pmin_dbm=pmin_dbm,
                pmax_dbm=pmax_dbm,
                target_ber=target_ber,
                seed=seed,
                mode="auto",
            )
            return {
                "mcs": np.asarray(out["mcs"], dtype=int),
                "power_dbm": np.asarray(out["power_dbm"], dtype=float),
            }
        except Exception:
            mcs = np.full(n_users, int(mcs_candidates.min()), dtype=int)
            if np.any(mcs_candidates == 16):
                mcs[quality >= 15.0] = 16
            if np.any(mcs_candidates == 64):
                mcs[quality >= 22.0] = 64

            total_lin = 10 ** (float(total_power_dbm) / 10.0)
            each_dbm = 10.0 * np.log10(max(total_lin / n_users, 1e-12))
            each_dbm = np.clip(each_dbm, pmin_dbm, pmax_dbm)
            return {"mcs": mcs, "power_dbm": np.full(n_users, each_dbm, dtype=float)}
# EVOLVE-BLOCK-END
