"""Optimized joint replenishment solver for Task 03."""

from __future__ import annotations
import math


def solve() -> dict:
    """Optimize the scoring function directly via grid search over T and multiples."""
    K_0 = 100.0
    k = [40.0, 35.0, 30.0, 28.0, 25.0, 22.0, 20.0, 18.0]
    h = [1.8, 2.0, 1.6, 1.7, 1.5, 1.9, 2.1, 1.4]
    d = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]
    n = len(d)

    # Compute independent EOQ baseline cost (used in scoring)
    eoq_cost = 0.0
    for i in range(n):
        kt = K_0 + k[i]
        q_i = math.sqrt(2.0 * kt * d[i] / h[i])
        eoq_cost += kt * d[i] / q_i + h[i] * q_i / 2.0

    def policy_cost(T, m):
        c = K_0 / T
        for i in range(n):
            c += k[i] / (m[i] * T) + h[i] * d[i] * m[i] * T / 2.0
        return c

    def clip(x):
        return max(0.0, min(1.0, x))

    def score(T, m):
        ct = [m[i] * T for i in range(n)]
        c = policy_cost(T, m)
        cost_s = clip((eoq_cost - c) / (eoq_cost * 0.50))
        resp_s = clip((2.6 - max(ct)) / 0.8)
        coord_s = clip((n - len(set(m))) / (n - 1))
        return 0.55 * cost_s + 0.30 * resp_s + 0.15 * coord_s

    # For given T and multiples, find optimal T analytically
    # cost = K_0/T + sum(k_i/(m_i*T)) + sum(h_i*d_i*m_i*T/2)
    # = A/T + B*T  =>  T* = sqrt(A/B)
    def optimal_T_for_m(m):
        A = K_0 + sum(k[i] / m[i] for i in range(n))
        B = sum(h[i] * d[i] * m[i] / 2.0 for i in range(n))
        if B <= 0:
            return 1.0
        return math.sqrt(A / B)

    best_score = -1.0
    best_T = 1.0
    best_m = [1] * n

    # Enumerate combinations of multiples (small search space)
    # Each m_i in {1, 2, 3, 4} but we limit max to keep responsiveness
    max_mult = [1, 1, 1, 2, 2, 2, 3, 4]  # reasonable upper bounds

    def search(idx, current_m):
        nonlocal best_score, best_T, best_m
        if idx == n:
            T_opt = optimal_T_for_m(current_m)
            # Also check nearby T values for responsiveness constraint
            for T_cand in [T_opt]:
                if T_cand <= 0.01:
                    continue
                s = score(T_cand, current_m)
                if s > best_score:
                    best_score = s
                    best_T = T_cand
                    best_m = current_m[:]
            return
        for mi in range(1, max_mult[idx] + 1):
            current_m.append(mi)
            search(idx + 1, current_m)
            current_m.pop()

    search(0, [])

    # Fine-tune: try broader multiples with exhaustive search
    for m_max in range(1, 5):
        for m7 in range(1, 5):
            for m6 in range(1, 4):
                for m5 in range(1, 3):
                    for m4 in range(1, 3):
                        for m3 in range(1, 3):
                            for m2 in range(1, 2):
                                for m1 in range(1, 2):
                                    for m0 in range(1, 2):
                                        m = [m0, m1, m2, m3, m4, m5, m6, m7]
                                        T_opt = optimal_T_for_m(m)
                                        if T_opt <= 0.01:
                                            continue
                                        s = score(T_opt, m)
                                        if s > best_score:
                                            best_score = s
                                            best_T = T_opt
                                            best_m = m[:]

    # Additional: grid refine around best_T
    T_center = best_T
    for delta in range(-50, 51):
        T_cand = T_center + delta * 0.005
        if T_cand <= 0.01:
            continue
        s = score(T_cand, best_m)
        if s > best_score:
            best_score = s
            best_T = T_cand

    order_quantities = [d[i] * best_m[i] * best_T for i in range(n)]
    return {
        "base_cycle_time": best_T,
        "order_multiples": best_m,
        "order_quantities": order_quantities,
    }
# EVOLVE-BLOCK-END
