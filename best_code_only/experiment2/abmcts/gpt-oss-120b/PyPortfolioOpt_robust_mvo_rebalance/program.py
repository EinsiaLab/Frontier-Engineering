# EVOLVE-BLOCK-START
import numpy as np

def solve_instance(instance: dict) -> dict:
    # Extract data
    mu = np.asarray(instance["mu"], dtype=float)
    cov = np.asarray(instance["cov"], dtype=float)
    w_prev = np.asarray(instance["w_prev"], dtype=float)
    lower = np.asarray(instance["lower"], dtype=float)
    upper = np.asarray(instance["upper"], dtype=float)
    sector_ids = np.asarray(instance["sector_ids"], dtype=int)
    sector_lower = instance.get("sector_lower", {})
    sector_upper = instance.get("sector_upper", {})
    risk_aversion = float(instance["risk_aversion"])
    transaction_penalty = float(instance["transaction_penalty"])
    turnover_limit = float(instance["turnover_limit"])

    factor_loadings = np.asarray(instance.get("factor_loadings", []), dtype=float)
    factor_lower = instance.get("factor_lower", {})
    factor_upper = instance.get("factor_upper", {})

    N = mu.shape[0]

    # Helper: enforce bounds and sum to 1
    def _project_bounds_sum(w):
        w = np.clip(w, lower, upper)
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = np.ones_like(w) / w.size
        return w

    # Helper: enforce turnover limit by scaling delta
    def _enforce_turnover(w):
        delta = w - w_prev
        turn = np.abs(delta).sum()
        if turn <= turnover_limit + 1e-12:
            return w
        scale = turnover_limit / max(turn, 1e-12)
        return w_prev + scale * delta

    # Helper: simple sector projection (iterative)
    def _enforce_sector(w):
        w = w.copy()
        sectors = np.unique(sector_ids)
        for _ in range(5):
            changed = False
            for s in sectors:
                idx = np.where(sector_ids == s)[0]
                total = w[idx].sum()
                lo = sector_lower.get(int(s), 0.0)
                hi = sector_upper.get(int(s), 1.0)
                if total > hi + 1e-10:
                    excess = total - hi
                    room = w[idx] - lower[idx]
                    cap = room.sum()
                    if cap > 1e-12:
                        take = np.minimum(room, excess * room / cap)
                        w[idx] -= take
                        changed = True
                elif total < lo - 1e-10:
                    need = lo - total
                    room = upper[idx] - w[idx]
                    cap = room.sum()
                    if cap > 1e-12:
                        add = np.minimum(room, need * room / cap)
                        w[idx] += add
                        changed = True
            w = _project_bounds_sum(w)
            if not changed:
                break
        return w

    # Try exact convex solve with CVXPY
    try:
        import cvxpy as cp

        w = cp.Variable(N)

        # Objective: mean - risk_aversion * variance - transaction_penalty * turnover (L1)
        objective = cp.Maximize(
            mu @ w
            - risk_aversion * cp.quad_form(w, cov)
            - transaction_penalty * cp.norm1(w - w_prev)
        )

        constraints = [
            cp.sum(w) == 1,
            w >= lower,
            w <= upper,
        ]

        if turnover_limit >= 0:
            constraints.append(cp.norm1(w - w_prev) <= turnover_limit)

        # Sector constraints
        if sector_lower or sector_upper:
            for s in np.unique(sector_ids):
                idx = np.where(sector_ids == s)[0]
                expr = cp.sum(w[idx])
                lo = sector_lower.get(int(s), 0.0)
                hi = sector_upper.get(int(s), 1.0)
                constraints.append(expr >= lo)
                constraints.append(expr <= hi)

        # Factor constraints
        if factor_loadings.size > 0 and (factor_lower or factor_upper):
            K = factor_loadings.shape[1]
            factor_exp = factor_loadings.T @ w
            for k in range(K):
                lo = factor_lower.get(int(k), -cp.inf)
                hi = factor_upper.get(int(k), cp.inf)
                if lo != -cp.inf:
                    constraints.append(factor_exp[k] >= lo)
                if hi != cp.inf:
                    constraints.append(factor_exp[k] <= hi)

        prob = cp.Problem(objective, constraints)

        # First try ECOS
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=5000)
        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            # Fall back to SCS
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("CVXPY failed to find a feasible solution")
        weights = np.asarray(w.value, dtype=float)

        # Post‑process to guarantee feasibility (tiny numerical errors)
        weights = np.clip(weights, lower, upper)
        weights = _enforce_turnover(weights)
        weights = _enforce_sector(weights)
        weights = _project_bounds_sum(weights)

        return {"weights": weights}

    except Exception:
        # Heuristic fallback (projected gradient ascent)
        def _project_to_simplex(v):
            n = v.size
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
            if rho.size == 0:
                return np.ones(n) / n
            rho = rho[-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            w = np.maximum(v - theta, 0)
            s = w.sum()
            if s <= 0:
                return np.ones(n) / n
            return w / s

        def _enforce_bounds(w):
            return np.clip(w, lower, upper)

        def _enforce_sum_and_bounds(w):
            w = _enforce_bounds(w)
            for _ in range(20):
                gap = 1.0 - w.sum()
                if abs(gap) < 1e-10:
                    break
                free = (w > lower + 1e-12) & (w < upper - 1e-12)
                if not np.any(free):
                    w = _project_to_simplex(w)
                    w = _enforce_bounds(w)
                    continue
                w[free] += gap / free.sum()
                w = _enforce_bounds(w)
            s = w.sum()
            if s <= 0:
                return np.ones_like(w) / w.size
            return w / s

        # Initialize feasible point
        w = _enforce_sum_and_bounds(np.clip(w_prev, lower, upper))

        eps = 1e-4
        for t in range(250):
            step = 0.08 / np.sqrt(t + 1.0)
            delta = w - w_prev
            smooth_sign = delta / np.sqrt(delta * delta + eps)
            grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

            w = w + step * grad
            w = _enforce_bounds(w)
            w = _enforce_turnover(w)
            w = _enforce_sector(w)
            w = _enforce_sum_and_bounds(w)

        return {"weights": w}
# EVOLVE-BLOCK-END
