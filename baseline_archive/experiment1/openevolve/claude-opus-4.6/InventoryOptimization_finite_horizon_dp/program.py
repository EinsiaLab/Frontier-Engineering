# EVOLVE-BLOCK-START
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm


def _dp_solve(demand_mean, demand_sd, h, p, K, c, h_T, p_T, init_inv):
    """Finite-horizon DP for (s,S) inventory policy."""
    T = len(demand_mean)
    inv_min = -100
    inv_max = 300
    states = np.arange(inv_min, inv_max + 1, dtype=float)
    n_states = len(states)

    # Terminal cost
    V_next = np.where(states >= 0, h_T * states, p_T * (-states))

    s_levels = [0] * T
    S_levels = [0] * T

    for t in range(T - 1, -1, -1):
        mu = demand_mean[t]
        sd = demand_sd[t]
        # Discretize demand
        d_min = max(0, int(mu - 4 * sd))
        d_max = int(mu + 4 * sd) + 1
        demands = np.arange(d_min, d_max + 1, dtype=float)
        # Demand probabilities (normal, truncated at 0)
        cdf_upper = norm.cdf(demands + 0.5, mu, sd)
        cdf_lower = norm.cdf(demands - 0.5, mu, sd)
        probs = cdf_upper - cdf_lower
        if d_min == 0:
            probs[0] = norm.cdf(demands[0] + 0.5, mu, sd)
        probs[-1] = 1.0 - norm.cdf(demands[-1] - 0.5, mu, sd)
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()

        # For each post-order inventory y, compute expected cost
        y_vals = np.arange(inv_min, inv_max + 1, dtype=float)
        # Expected single-period cost G(y) = E[h*max(y-d,0) + p*max(d-y,0) + V_next(y-d)]
        G = np.zeros(len(y_vals))
        for i, d in enumerate(demands):
            new_inv = y_vals - d  # inventory after demand
            # Holding/stockout cost
            period_cost = np.where(new_inv >= 0, h * new_inv, p * (-new_inv))
            # Future cost: interpolate V_next
            idx = np.clip(np.round(new_inv - inv_min).astype(int), 0, n_states - 1)
            period_cost += V_next[idx]
            G += probs[i] * period_cost

        # For each state x, find optimal action
        V_curr = np.zeros(n_states)
        best_s = 0
        best_S = 0

        # Find optimal S* = argmin(K + c*y + G(y)) for ordering case
        order_cost = K + c * y_vals + G
        best_S_idx = np.argmin(order_cost)
        best_S_val = y_vals[best_S_idx]
        min_order_cost = order_cost[best_S_idx]

        for i, x in enumerate(states):
            # Cost of not ordering
            x_idx = i  # x maps to index i
            no_order_cost = c * x + G[i]  # no fixed cost, just variable + future
            # Actually, if not ordering, y = x, cost = G(x) (no purchase cost for existing stock)
            no_order_cost_actual = G[i]
            # If ordering to S*, cost = K + c*(S*-x) + G(S*)
            if best_S_val > x:
                order_cost_actual = K + c * (best_S_val - x) + G[best_S_idx]
            else:
                order_cost_actual = float('inf')

            if order_cost_actual < no_order_cost_actual:
                V_curr[i] = order_cost_actual
            else:
                V_curr[i] = no_order_cost_actual

        # Find s: largest x where ordering is still optimal
        s_val = inv_min
        for i in range(n_states):
            x = states[i]
            if x >= best_S_val:
                break
            if K + c * (best_S_val - x) + G[best_S_idx] < G[i]:
                s_val = x

        s_levels[t] = int(round(s_val))
        S_levels[t] = int(round(best_S_val))
        V_next = V_curr

    return s_levels, S_levels


def _sim(sl, Sl, dm, ds, trials=600):
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
            sl, Sl = _dp_solve(demand_mean, demand_sd, 1.2, p_dp, 120.0, 3.5, 0.8, p_T, 30.0)
            ac, af = _sim(sl, Sl, demand_mean, demand_sd, 600)
            sc = _score(ac, af, base_cost)
            if sc > best_sc: best_sc, best_r = sc, (sl[:], Sl[:])
        except Exception:
            continue
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
    if best_r:
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
