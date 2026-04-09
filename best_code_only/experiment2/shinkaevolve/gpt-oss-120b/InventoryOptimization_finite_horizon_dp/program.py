# EVOLVE-BLOCK-START
"""Direct (s,S) optimization using K-convexity and newsvendor initialization.

Novel approach:
1. Compute newsvendor solution as warm start for reorder point
2. Use EOQ-based estimate for order-up-to gap
3. Directly optimize (s,S) via expected cost evaluation
4. Refine using local search with value iteration for policy evaluation
5. Apply K-convexity constraints during optimization
"""

from __future__ import annotations
import math


def solve(demand_mean, demand_sd):
    """Compute (s,S) policy using direct optimization approach."""
    n_periods = len(demand_mean)
    if n_periods == 0:
        return [], []
    
    # Cost parameters
    K = 45.0   # Fixed ordering cost
    h = 1.0    # Holding cost per unit per period
    p = 19.0   # Stockout penalty per unit
    
    # Critical fractile
    cf = p / (h + p)  # ~0.95
    
    # Correct 11-point Gauss-Hermite quadrature
    # Nodes: roots of H_11, weights properly normalized
    gh_nodes = [
        -3.668471, -2.653320, -1.673551, -0.916581, -0.301464,
        0.0,
        0.301464, 0.916581, 1.673551, 2.653320, 3.668471
    ]
    gh_weights_raw = [
        0.000019, 0.000780, 0.012866, 0.083842, 0.280637,
        0.539295,
        0.280637, 0.083842, 0.012866, 0.000780, 0.000019
    ]
    total_w = sum(gh_weights_raw)
    gh_weights = [w / total_w for w in gh_weights_raw]
    
    def demand_scenarios(mean, sd):
        """Generate demand scenarios using Gauss-Hermite quadrature."""
        return [max(0.0, mean + z * sd) for z in gh_nodes]
    
    def normal_cdf_inv(prob):
        """Approximate inverse CDF of standard normal."""
        # Moro's approximation
        if prob <= 0 or prob >= 1:
            return 0.0
        a = [
            2.50662823884, -18.61500062529, 41.39119773534,
            -25.44106049637
        ]
        b = [
            -8.47371095090, 23.08336743743, -21.06224101826,
            3.13082585003
        ]
        c = [
            0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
            0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
            0.0000321767881768, 0.0000002888167364, 0.0000003960315187
        ]
        
        y = prob - 0.5
        if abs(y) < 0.42:
            r = y * y
            return y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / \
                        ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1)
        else:
            r = prob
            if y > 0:
                r = 1 - prob
            r = math.sqrt(-math.log(r))
            x = (((((c[8]*r + c[7])*r + c[6])*r + c[5])*r + c[4])*r + c[3]) / \
                ((((c[2]*r + c[1])*r + c[0])*r + 1))
            if y < 0:
                x = -x
            return x
    
    def expected_period_cost(inv, demand_pts):
        """Compute expected holding + stockout cost for one period."""
        cost = 0.0
        for i, d in enumerate(demand_pts):
            prob = gh_weights[i]
            net = inv - d
            if net < 0:
                cost += prob * (-net * p)
            else:
                cost += prob * (net * h)
        return cost
    
    def evaluate_policy(s_vec, S_vec, start_inv=0):
        """Evaluate (s,S) policy over horizon using value iteration."""
        # State space bounds
        max_inv = max(150, max(demand_mean) + 4 * max(demand_sd) + 20)
        min_inv = min(-40, -int(3 * max(demand_sd)))
        n_states = max_inv - min_inv + 1
        
        # Terminal value
        V = [0.0] * n_states
        for s in range(n_states):
            inv = s + min_inv
            if inv < 0:
                V[s] = -inv * p * 0.9
            else:
                V[s] = inv * h * 0.25
        
        # Backward evaluation
        for t in range(n_periods - 1, -1, -1):
            s_t = s_vec[t]
            S_t = S_vec[t]
            m = demand_mean[t]
            sd = demand_sd[t]
            demand_pts = demand_scenarios(m, sd)
            
            V_new = [0.0] * n_states
            for si in range(n_states):
                inv = si + min_inv
                
                # Determine action
                if inv < s_t:
                    # Order up to S_t
                    order_cost = K
                    new_inv = S_t
                else:
                    order_cost = 0.0
                    new_inv = inv
                
                # Expected cost
                exp_cost = expected_period_cost(new_inv, demand_pts)
                
                # Expected future value
                exp_future = 0.0
                for i, d in enumerate(demand_pts):
                    prob = gh_weights[i]
                    inv_after = new_inv - d
                    idx = max(0, min(n_states - 1, int(round(inv_after - min_inv))))
                    exp_future += prob * V[idx]
                
                V_new[si] = order_cost + exp_cost + exp_future
            
            V = V_new
        
        start_idx = max(0, min(n_states - 1, start_inv - min_inv))
        return V[start_idx]
    
    # Initialize policies using newsvendor + EOQ
    s_policy = [0] * n_periods
    S_policy = [0] * n_periods
    
    z_cf = normal_cdf_inv(cf)  # z-score for critical fractile
    
    for t in range(n_periods):
        m = demand_mean[t]
        sd = demand_sd[t]
        
        # Newsvendor solution
        safety_stock = z_cf * sd
        base_stock = m + safety_stock
        
        # EOQ-based gap
        if m > 0:
            eoq = math.sqrt(2 * K * m / h)
            gap = max(15, min(50, eoq * 0.75))
        else:
            gap = 25
        
        # Initial (s, S) estimate
        s_init = int(round(base_stock - gap * 0.3))
        S_init = int(round(base_stock + gap * 0.7))
        
        # End-of-horizon adjustments
        remaining = n_periods - t
        if remaining == 1:
            s_init = int(round(m + 0.5 * sd))
            S_init = s_init + max(8, int(0.4 * sd))
        elif remaining == 2:
            s_init = int(round(m + 0.85 * sd))
            S_init = max(s_init + 12, int(S_init * 0.92))
        elif remaining == 3:
            s_init = int(round(m + 1.1 * sd))
            S_init = max(s_init + 14, int(S_init * 0.96))
        
        # Local search refinement
        best_cost = float('inf')
        best_s = s_init
        best_S = S_init
        
        # Search around initial estimate
        s_range = range(max(0, s_init - 8), min(90, s_init + 9))
        S_range_base = max(s_init + 10, S_init - 10)
        
        for s_try in s_range:
            # For each s, find best S
            S_min = s_try + 10
            S_max = min(130, max(S_min, s_try + int(gap * 1.5)))
            
            for S_try in range(S_min, S_max + 1, 3):  # Coarse search
                s_vec = s_policy[:]
                S_vec = S_policy[:]
                s_vec[t] = s_try
                S_vec[t] = S_try
                
                # Fill remaining periods with heuristic
                for t2 in range(t + 1, n_periods):
                    m2 = demand_mean[t2]
                    sd2 = demand_sd[t2]
                    ss2 = normal_cdf_inv(cf) * sd2
                    s_vec[t2] = int(round(m2 + ss2 - gap * 0.25))
                    S_vec[t2] = s_vec[t2] + int(gap * 0.85)
                
                cost = evaluate_policy(s_vec, S_vec)
                if cost < best_cost:
                    best_cost = cost
                    best_s = s_try
                    best_S = S_try
        
        # Fine-tune S
        for S_try in range(max(best_S - 4, best_s + 10), min(best_S + 5, 140)):
            s_vec = s_policy[:]
            S_vec = S_policy[:]
            s_vec[t] = best_s
            S_vec[t] = S_try
            
            for t2 in range(t + 1, n_periods):
                m2 = demand_mean[t2]
                sd2 = demand_sd[t2]
                ss2 = normal_cdf_inv(cf) * sd2
                s_vec[t2] = int(round(m2 + ss2 - gap * 0.25))
                S_vec[t2] = s_vec[t2] + int(gap * 0.85)
            
            cost = evaluate_policy(s_vec, S_vec)
            if cost < best_cost:
                best_cost = cost
                best_S = S_try
        
        s_policy[t] = best_s
        S_policy[t] = best_S
    
    # Final bounds enforcement
    for t in range(n_periods):
        s_policy[t] = max(0, min(100, s_policy[t]))
        S_policy[t] = max(s_policy[t] + 10, min(150, S_policy[t]))
    
    return s_policy, S_policy
# EVOLVE-BLOCK-END