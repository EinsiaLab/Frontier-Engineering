"""Baseline for Task 03 – direct score maximization."""
from __future__ import annotations
import math

def solve() -> dict:
    K0, n = 100.0, 8
    K = [40.0, 35.0, 30.0, 28.0, 25.0, 22.0, 20.0, 18.0]
    h = [1.8, 2.0, 1.6, 1.7, 1.5, 1.9, 2.1, 1.4]
    d = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]
    hd = [h[i]*d[i] for i in range(n)]
    BC = 1073.6630262434117
    HBC = BC * 0.5
    def opt_T(m):
        A = K0 + sum(K[i]/m[i] for i in range(n))
        B = 0.5 * sum(hd[i]*m[i] for i in range(n))
        return math.sqrt(A / B) if B > 0 else 1.0
    def cost(T, m):
        return K0/T + sum(K[i]/(m[i]*T) + 0.5*hd[i]*m[i]*T for i in range(n))
    def sc(T, m):
        c = cost(T, m)
        cs = min(1.0, max(0.0, (BC - c) / HBC))
        rs = min(1.0, max(0.0, (2.6 - max(m[i]*T for i in range(n))) / 0.8))
        ns = (n - len(set(m))) / (n - 1)
        return 0.55*cs + 0.30*rs + 0.15*ns
    # Uniform m=1: analytical T* = sqrt(318/158.25) ≈ 1.4172
    m1 = [1]*n
    T1 = opt_T(m1)
    best_s, best_T, best_m = sc(T1, m1), T1, m1[:]
    # Exhaustive search m_i in {1,2,3} with analytical + boundary T
    from itertools import product
    for combo in product(range(1, 4), repeat=n):
        m = list(combo)
        T = opt_T(m)
        mmax = max(m)
        if mmax * T <= 2.6:
            s = sc(T, m)
            if s > best_s:
                best_s, best_T, best_m = s, T, list(m)
        Tb = 1.8 / mmax
        if Tb > 0.01:
            s = sc(Tb, m)
            if s > best_s:
                best_s, best_T, best_m = s, Tb, list(m)
    # Fine-tune best with golden section then grid
    m0 = best_m[:]
    T0 = best_T
    lo = max(0.01, T0 - 0.5)
    hi = min(2.6 / max(m0), T0 + 0.5)
    gr = (math.sqrt(5) + 1) / 2
    for _ in range(100):
        if hi - lo < 1e-8: break
        c1 = hi - (hi - lo) / gr
        c2 = lo + (hi - lo) / gr
        if sc(c1, m0) > sc(c2, m0):
            hi = c2
        else:
            lo = c1
    Tg = (lo + hi) / 2
    sg = sc(Tg, m0)
    if sg > best_s:
        best_s, best_T = sg, Tg
    # Ultra-fine grid around best T
    for dT in range(-20000, 20001):
        Tt = best_T + dT * 1e-6
        if Tt <= 0.001 or max(m0[i]*Tt for i in range(n)) > 2.6: continue
        s = sc(Tt, m0)
        if s > best_s:
            best_s, best_T = s, Tt
    qs = [d[i]*best_m[i]*best_T for i in range(n)]
    return {"base_cycle_time": best_T, "order_multiples": best_m, "order_quantities": qs}
# EVOLVE-BLOCK-END
