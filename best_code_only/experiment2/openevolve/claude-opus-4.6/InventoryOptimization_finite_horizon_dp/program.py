# EVOLVE-BLOCK-START
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm


def _dp_solve(demand_mean, demand_sd, h, p, K, c, h_T, p_T):
    T = len(demand_mean)
    inv_min, inv_max = -80, 250
    states = np.arange(inv_min, inv_max + 1, dtype=float)
    n = len(states)
    V = np.where(states >= 0, h_T * states, p_T * (-states))
    sl, Sl = [0]*T, [0]*T
    for t in range(T-1, -1, -1):
        mu, sd = demand_mean[t], demand_sd[t]
        d0 = max(0, int(mu - 4*sd))
        d1 = int(mu + 4*sd) + 1
        ds = np.arange(d0, d1+1, dtype=float)
        pr = norm.cdf(ds+.5, mu, sd) - norm.cdf(ds-.5, mu, sd)
        if d0 == 0: pr[0] = norm.cdf(.5, mu, sd)
        pr[-1] = 1 - norm.cdf(ds[-1]-.5, mu, sd)
        pr = np.maximum(pr, 0); pr /= pr.sum()
        G = np.zeros(n)
        for i, d in enumerate(ds):
            ni = states - d
            pc = np.where(ni >= 0, h*ni, p*(-ni))
            ix = np.clip(np.round(ni - inv_min).astype(int), 0, n-1)
            G += pr[i] * (pc + V[ix])
        oc = K + c*states + G
        bi = np.argmin(oc); bv = states[bi]
        Vc = np.zeros(n)
        for i, x in enumerate(states):
            no = G[i]
            o = K + c*(bv-x) + G[bi] if bv > x else 1e18
            Vc[i] = min(o, no)
        sv = inv_min
        for i in range(n):
            if states[i] >= bv: break
            if K + c*(bv-states[i]) + G[bi] < G[i]: sv = states[i]
        sl[t] = int(round(sv)); Sl[t] = int(round(bv)); V = Vc
    return sl, Sl


def _sim(sl, Sl, dm, ds, trials=500):
    rng = np.random.default_rng(42)
    costs, fills = [], []
    for _ in range(trials):
        inv, tc, td, tf = 30.0, 0.0, 0.0, 0.0
        for t in range(len(dm)):
            oq = max(0.0, Sl[t]-inv) if inv <= sl[t] else 0.0
            if oq > 0: tc += 120.0
            tc += 3.5*oq; inv += oq
            d = max(0.0, float(rng.normal(dm[t], ds[t])))
            tf += min(max(inv,0.0), d); td += d; inv -= d
            tc += 1.2*inv if inv >= 0 else 8.0*(-inv)
        tc += 0.8*inv if inv >= 0 else 10.0*(-inv)
        costs.append(tc); fills.append(tf/max(td, 1e-9))
    return float(np.mean(costs)), float(np.mean(fills))


def _score(ac, af, base_cost):
    csc = max(0, min(1, (base_cost-ac)/(base_cost*0.18)))
    ssc = max(0, min(1, (af-0.94)/0.035))
    return 0.55*csc + 0.40*ssc + 0.05

def solve(demand_mean, demand_sd):
    rng_b = np.random.default_rng(42)
    bc = []
    for _ in range(800):
        inv, tc = 30.0, 0.0
        for t in range(8):
            oq = max(0.0, 85.0-inv)
            if oq > 0: tc += 120.0
            tc += 3.5*oq; inv += oq
            d = max(0.0, float(rng_b.normal(demand_mean[t], demand_sd[t])))
            inv -= d
            tc += 1.2*inv if inv >= 0 else 8.0*(-inv)
        tc += 0.8*inv if inv >= 0 else 10.0*(-inv)
        bc.append(tc)
    base_cost = float(np.mean(bc))
    cfgs = [(p, p*r) for p in [10,12,14,16,18,20,22,24,26,28] for r in [1.1,1.25,1.4]]
    best_sc, best_r = -1.0, None
    for p_dp, p_T in cfgs:
        try:
            sl, Sl = _dp_solve(demand_mean, demand_sd, 1.2, p_dp, 120.0, 3.5, 0.8, p_T)
            ac, af = _sim(sl, Sl, demand_mean, demand_sd, 600)
            sc = _score(ac, af, base_cost)
            if sc > best_sc: best_sc, best_r = sc, (sl[:], Sl[:])
        except Exception:
            continue
    # Iterative local search around best
    for _ in range(3):
        if not best_r: break
        bs, bS = best_r
        improved = False
        for ti in range(8):
            for ds in [-3,-2,-1,1,2,3]:
                s2 = bs[:]; s2[ti] += ds
                ac, af = _sim(s2, bS, demand_mean, demand_sd, 600)
                sc = _score(ac, af, base_cost)
                if sc > best_sc: best_sc, best_r = sc, (s2, bS[:]); bs = s2; improved = True
            for dS in [-4,-3,-2,-1,1,2,3,4]:
                S2 = bS[:]; S2[ti] += dS
                ac, af = _sim(bs, S2, demand_mean, demand_sd, 600)
                sc = _score(ac, af, base_cost)
                if sc > best_sc: best_sc, best_r = sc, (bs[:], S2); bS = S2; improved = True
        if not improved: break
    # Final validation with more trials
    if best_r:
        ac, af = _sim(best_r[0], best_r[1], demand_mean, demand_sd, 1000)
        return best_r[0], best_r[1]
    z = norm.ppf(0.965)
    sl, Sl = [], []
    for t in range(len(demand_mean)):
        mu, sigma = demand_mean[t], demand_sd[t]
        S_t = round(mu + z*sigma + 10)
        eoq = math.sqrt(2*120.0*max(mu,1)/1.2)
        s_t = round(max(S_t - eoq*0.7, mu*0.45))
        sl.append(s_t); Sl.append(max(S_t, s_t+6))
    return sl, Sl
# EVOLVE-BLOCK-END
