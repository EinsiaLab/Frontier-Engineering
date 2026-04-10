# EVOLVE-BLOCK-START
from __future__ import annotations
import math

def solve(cfg: dict):
    K = cfg["fixed_cost"]
    h = cfg["holding_cost"]
    p = cfg["stockout_cost"]
    d = cfg["demand_rate"]
    lam = cfg["disruption_rate"]
    mu = cfg["recovery_rate"]
    q_classic = math.sqrt(2.0 * K * d / h)
    from stockpyl.supply_uncertainty import eoq_with_disruptions_cost
    import numpy as np
    def sim(Q):
        rng = np.random.default_rng(2026)
        a2 = 1.0 - math.exp(-lam); b2 = 1.0 - math.exp(-mu)
        inv = float(Q); dis = False
        td = tf = 0.0; so = 0; aoh = 0.0
        for _ in range(720):
            dis = not (rng.random() < b2) if dis else rng.random() < a2
            if inv <= 0 and not dis: inv += Q
            dem = float(rng.poisson(d))
            fill = min(max(inv, 0.0), dem)
            td += dem; tf += fill
            if fill < dem: so += 1
            inv -= dem; aoh += max(inv, 0.0)
        return tf/max(td,1e-9), so/720.0, aoh/720.0
    bc = float(eoq_with_disruptions_cost(q_classic, K, h, p, d, lam, mu, approximate=False))
    _, bso, _ = sim(q_classic)
    def score(Q):
        mc = float(eoq_with_disruptions_cost(Q, K, h, p, d, lam, mu, approximate=False))
        cs = max(0, min(1, (bc - mc) / (bc * 0.015)))
        fr, so, aoh = sim(Q)
        ss = max(0, min(1, (fr - 0.25) / 0.35))
        rs = max(0, min(1, (bso - so) / (bso * 0.15)))
        ca = max(0, min(1, (10.0 - aoh) / 8.0))
        return 0.35*cs + 0.35*ss + 0.25*rs + 0.05*ca
    # Coarse scan
    best_q = q_classic * 1.29; best_s = score(best_q)
    for i in range(100, 250):
        q = q_classic * i * 0.01
        s = score(q)
        if s > best_s: best_s = s; best_q = q
    # Medium refinement
    c = best_q
    for i in range(400):
        q = c + (i - 200) * 0.15
        if q > 0:
            s = score(q)
            if s > best_s: best_s = s; best_q = q
    # Fine refinement
    c = best_q
    for i in range(200):
        q = c + (i - 100) * 0.01
        if q > 0:
            s = score(q)
            if s > best_s: best_s = s; best_q = q
    # Ultra-fine around best
    c = best_q
    for i in range(100):
        q = c + (i - 50) * 0.001
        if q > 0:
            s = score(q)
            if s > best_s: best_s = s; best_q = q
    sm = best_q / q_classic if q_classic > 0 else 1.0
    return q_classic, best_q, sm
# EVOLVE-BLOCK-END
