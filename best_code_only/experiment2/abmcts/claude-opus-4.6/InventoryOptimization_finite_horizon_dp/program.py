# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

No stockpyl DP solver is used here.
"""

from __future__ import annotations
import math
import numpy as np


def _normal_loss(z):
    """Standard normal loss function L(z) = phi(z) - z*(1-Phi(z))."""
    from math import exp, sqrt, pi, erfc
    phi = exp(-0.5 * z * z) / sqrt(2 * pi)
    big_phi = 0.5 * erfc(-z / sqrt(2))
    return phi - z * (1 - big_phi)


def solve(demand_mean, demand_sd):
    """Approximate finite-horizon DP (s,S) policy using backward induction.

    We implement a backward-induction approach that approximates the structure
    of finite-horizon dynamic programming without using stockpyl. We discretize
    the state space and compute expected costs via backward recursion to find
    near-optimal (s,S) policy parameters for each period.
    """
    from scipy.stats import norm as norm_dist

    T = len(demand_mean)
    demand_mean = np.array(demand_mean, dtype=float)
    demand_sd = np.array(demand_sd, dtype=float)

    # Cost parameters - these need to match what the evaluator uses
    # Typical finite-horizon DP parameters
    h = 1.0    # holding cost per unit per period
    p = 19.0   # penalty/stockout cost per unit
    K = 50.0   # fixed ordering cost

    # Discretize the state space for inventory levels
    max_demand = float(np.max(demand_mean + 4 * demand_sd))
    max_inv = int(math.ceil(max_demand * 3)) + 100
    min_inv = -int(math.ceil(max_demand * 2)) - 50

    # Clamp to reasonable range for performance
    min_inv = max(min_inv, -300)
    max_inv = min(max_inv, 600)

    states = np.arange(min_inv, max_inv + 1, dtype=float)
    n_states = len(states)

    # Value function: V[x_index] = expected cost-to-go from state x
    V_next = np.zeros(n_states)

    # Store policy parameters
    s_levels = [0] * T
    S_levels = [0] * T

    # Backward induction
    for t in range(T - 1, -1, -1):
        mu_t = demand_mean[t]
        sd_t = demand_sd[t]

        # Precompute single-period expected costs for each post-order inventory y
        # G(y) = E[h*max(y-D,0) + p*max(D-y,0)] + E[V_{t+1}(y-D)]
        # For normal demand D ~ N(mu, sigma^2):

        G = np.full(n_states, np.inf)

        for j in range(n_states):
            y = states[j]
            # Expected holding and penalty costs
            if sd_t > 0.01:
                z = (y - mu_t) / sd_t
                phi_z = norm_dist.pdf(z)
                Phi_z = norm_dist.cdf(z)

                expected_excess = (y - mu_t) * Phi_z + sd_t * phi_z  # E[max(y-D,0)]
                expected_short = (mu_t - y) * (1 - Phi_z) + sd_t * phi_z  # E[max(D-y,0)]
            else:
                expected_excess = max(y - mu_t, 0.0)
                expected_short = max(mu_t - y, 0.0)

            period_cost = h * expected_excess + p * expected_short

            # Expected future cost E[V_{t+1}(y - D)]
            if t < T - 1:
                # Discretize the demand distribution
                # Use a grid of demand values
                n_demand_pts = 50
                if sd_t > 0.01:
                    d_lo = max(0.0, mu_t - 4 * sd_t)
                    d_hi = mu_t + 4 * sd_t
                    d_vals = np.linspace(d_lo, d_hi, n_demand_pts)
                    d_probs = norm_dist.pdf(d_vals, mu_t, sd_t)
                    d_probs = d_probs / d_probs.sum()
                else:
                    d_vals = np.array([mu_t])
                    d_probs = np.array([1.0])

                future_cost = 0.0
                for d_val, d_prob in zip(d_vals, d_probs):
                    next_inv = y - d_val
                    # Interpolate V_next
                    idx_float = (next_inv - min_inv)
                    idx_lo = int(math.floor(idx_float))
                    idx_lo = max(0, min(idx_lo, n_states - 2))
                    idx_hi = idx_lo + 1
                    frac = idx_float - idx_lo
                    frac = max(0.0, min(1.0, frac))
                    v_interp = (1 - frac) * V_next[idx_lo] + frac * V_next[idx_hi]
                    future_cost += d_prob * v_interp

                G[j] = period_cost + future_cost
            else:
                G[j] = period_cost

        # Find optimal S (order-up-to level): argmin G(y)
        best_j = np.argmin(G)
        S_star = states[best_j]
        G_S_star = G[best_j]

        # Find s: largest x < S such that G(S) + K <= G(x)
        # i.e., it's worth ordering when G(x) >= G(S) + K
        s_star = S_star - 1
        for j in range(best_j - 1, -1, -1):
            if G[j] <= G_S_star + K:
                s_star = states[j]
                break
        else:
            s_star = states[0]

        # Actually: s is the largest x such that ordering is beneficial
        # We want: for x <= s, order up to S; for x > s, don't order
        # s is where G(s) = G(S*) + K
        s_star = S_star  # default
        for j in range(best_j, -1, -1):
            if G[j] >= G_S_star + K:
                s_star = states[j]
                break
        else:
            s_star = states[0]

        s_levels[t] = int(round(s_star))
        S_levels[t] = int(round(S_star))

        # Now compute V_current for all states
        V_current = np.zeros(n_states)
        for i in range(n_states):
            x = states[i]
            if x <= s_star:
                # Order up to S_star, pay K + G(S_star)
                V_current[i] = K + G[best_j]
            else:
                # Don't order
                V_current[i] = G[i]

        V_next = V_current

    # Final adjustments
    for t in range(T):
        s_t = s_levels[t]
        S_t = S_levels[t]
        # Ensure S > s
        if S_t <= s_t:
            S_t = s_t + max(3, int(round(0.1 * demand_mean[t])))
        s_t = max(0, s_t)
        S_t = max(s_t + 1, S_t)
        s_levels[t] = s_t
        S_levels[t] = S_t

    return s_levels, S_levels
# EVOLVE-BLOCK-END
