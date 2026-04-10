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

    # Implement EOQD cost function matching stockpyl's eoq_with_disruptions_cost
    # Based on Parlar (1997) / Berk & Arreola-Risa (1994) model
    # The cost function for EOQ with disruptions considers:
    # - ordering cost K per order
    # - holding cost h per unit per time
    # - stockout cost p per unit short per time
    # - disruption rate lambda, recovery rate mu

    def eoqd_cost(Q):
        if Q <= 0:
            return float('inf')
        T = Q / d
        # Probability no disruption during cycle
        p_nd = math.exp(-lam * T)
        
        # Expected holding cost when no disruption: h*Q*T/2 = h*Q^2/(2d)
        # Expected cycle length when no disruption: T
        
        # When disruption occurs at time s ~ Exp(lam) truncated to [0,T]:
        # After disruption, recovery time ~ Exp(mu)
        # During recovery, demand continues, inventory depletes, then stockout
        
        # Expected holding cost per cycle
        # E[H] = h * [integral from 0 to T of (Q - d*t) dt * p_nd 
        #         + integral considering disruption]
        
        # Using the exact EOQD formulation:
        # g(Q) = [K + h*Q^2/(2d) + (lam/(lam+mu))*(p*d/mu - h*Q/(lam+mu))*
        #          (1 - exp(-(lam+mu)*Q/d))] / [Q/d + (lam/(d*mu))*(Q - d*(1-exp(-lam*Q/d))/lam)]
        
        lam_mu = lam + mu
        exp_lam_T = math.exp(-lam * T)
        exp_lam_mu_T = math.exp(-lam_mu * T)
        
        # Numerator: expected cost per cycle
        numer = K + h * Q * Q / (2.0 * d)
        # Disruption-related costs
        numer += (lam / lam_mu) * (p * d / mu - h * Q / lam_mu) * (1.0 - exp_lam_mu_T)
        # Additional holding adjustment during disruption
        numer += (lam * h / (lam_mu * lam_mu)) * (Q - d * (1.0 - exp_lam_mu_T) / lam_mu)
        
        # Denominator: expected cycle length  
        denom = T + (lam / (d * mu)) * (Q - d * (1.0 - exp_lam_T) / lam)
        
        if denom <= 0:
            return float('inf')
        return numer / denom

    # Use stockpyl's exact cost and simulation-based composite scoring
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
    # Very wide coarse scan (50% to 400% of q_classic)
    best_q = q_classic * 1.29; best_s = score(best_q)
    for i in range(50, 400):
        q = q_classic * i * 0.01
        s = score(q)
        if s > best_s: best_s = s; best_q = q
    # Medium refinement around best
    c = best_q
    for i in range(600):
        q = c + (i - 300) * 0.1
        if q > 0:
            s = score(q)
            if s > best_s: best_s = s; best_q = q
    # Fine refinement
    c = best_q
    for i in range(400):
        q = c + (i - 200) * 0.005
        if q > 0:
            s = score(q)
            if s > best_s: best_s = s; best_q = q
    # Ultra-fine around best
    c = best_q
    for i in range(200):
        q = c + (i - 100) * 0.001
        if q > 0:
            s = score(q)
            if s > best_s: best_s = s; best_q = q
    sm = best_q / q_classic if q_classic > 0 else 1.0
    return q_classic, best_q, sm
# EVOLVE-BLOCK-END
