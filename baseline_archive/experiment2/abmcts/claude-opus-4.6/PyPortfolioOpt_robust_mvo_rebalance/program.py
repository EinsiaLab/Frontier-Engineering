# EVOLVE-BLOCK-START
import numpy as np

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
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


def _enforce_bounds(w: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(w, lower), upper)


def _enforce_sum_and_bounds(w: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    w = _enforce_bounds(w, lower, upper)
    for _ in range(20):
        gap = 1.0 - w.sum()
        if abs(gap) < 1e-10:
            break
        free = (w > lower + 1e-12) & (w < upper - 1e-12)
        if not np.any(free):
            w = _project_to_simplex(w)
            w = _enforce_bounds(w, lower, upper)
            continue
        w[free] += gap / free.sum()
        w = _enforce_bounds(w, lower, upper)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / w.size
    return w / s


def _enforce_turnover(w: np.ndarray, w_prev: np.ndarray, turnover_limit: float) -> np.ndarray:
    delta = w - w_prev
    turn = np.abs(delta).sum()
    if turn <= turnover_limit + 1e-12:
        return w
    scale = turnover_limit / max(turn, 1e-12)
    return w_prev + scale * delta


def _enforce_sector_bounds(
    w: np.ndarray,
    sector_ids: np.ndarray,
    sector_lower: dict,
    sector_upper: dict,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
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
        w = _enforce_sum_and_bounds(w, lower, upper)
        if not changed:
            break
    return w


def solve_instance_cvxpy(instance: dict) -> dict:
    mu = np.asarray(instance["mu"], dtype=float)
    cov = np.asarray(instance["cov"], dtype=float)
    w_prev = np.asarray(instance["w_prev"], dtype=float)
    lower = np.asarray(instance["lower"], dtype=float)
    upper = np.asarray(instance["upper"], dtype=float)
    sector_ids = np.asarray(instance["sector_ids"], dtype=int)
    sector_lower = instance["sector_lower"]
    sector_upper = instance["sector_upper"]
    risk_aversion = float(instance["risk_aversion"])
    transaction_penalty = float(instance["transaction_penalty"])
    turnover_limit = float(instance["turnover_limit"])

    N = len(mu)
    w = cp.Variable(N)

    # Objective: mu^T w - risk_aversion * w^T cov w - transaction_penalty * ||w - w_prev||_1
    ret = mu @ w
    risk = cp.quad_form(w, cp.psd_wrap(cov))
    tc = transaction_penalty * cp.norm(w - w_prev, 1)
    objective = cp.Maximize(ret - risk_aversion * risk - tc)

    constraints = [
        cp.sum(w) == 1,
        w >= lower,
        w <= upper,
        cp.norm(w - w_prev, 1) <= turnover_limit,
    ]

    # Sector constraints
    unique_sectors = np.unique(sector_ids)
    for s in unique_sectors:
        idx = np.where(sector_ids == s)[0]
        s_int = int(s)
        if s_int in sector_lower:
            constraints.append(cp.sum(w[idx]) >= sector_lower[s_int])
        if s_int in sector_upper:
            constraints.append(cp.sum(w[idx]) <= sector_upper[s_int])

    # Factor exposure constraints
    if "factor_loadings" in instance and instance["factor_loadings"] is not None:
        B = np.asarray(instance["factor_loadings"], dtype=float)
        if "factor_lower" in instance and instance["factor_lower"] is not None:
            fl = np.asarray(instance["factor_lower"], dtype=float)
            constraints.append(B.T @ w >= fl)
        if "factor_upper" in instance and instance["factor_upper"] is not None:
            fu = np.asarray(instance["factor_upper"], dtype=float)
            constraints.append(B.T @ w <= fu)

    prob = cp.Problem(objective, constraints)

    # Try multiple solvers with tuned parameters
    solver_configs = [
        (cp.CLARABEL, {"verbose": False, "tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12}),
        (cp.ECOS, {"verbose": False, "max_iters": 500, "abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10}),
        (cp.SCS, {"verbose": False, "max_iters": 100000, "eps": 1e-12}),
        (cp.OSQP, {"verbose": False, "max_iter": 50000, "eps_abs": 1e-10, "eps_rel": 1e-10}),
        (None, {"verbose": False}),
    ]

    for solver, kwargs in solver_configs:
        try:
            prob.solve(solver=solver, **kwargs)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                result = np.array(w.value).flatten()
                result = np.clip(result, lower, upper)
                s = result.sum()
                if abs(s - 1.0) > 1e-6:
                    result = result / s
                return {"weights": result}
        except Exception:
            continue

    return None


def solve_instance_heuristic(instance: dict) -> dict:
    mu = np.asarray(instance["mu"], dtype=float)
    cov = np.asarray(instance["cov"], dtype=float)
    w_prev = np.asarray(instance["w_prev"], dtype=float)
    lower = np.asarray(instance["lower"], dtype=float)
    upper = np.asarray(instance["upper"], dtype=float)
    sector_ids = np.asarray(instance["sector_ids"], dtype=int)
    sector_lower = instance["sector_lower"]
    sector_upper = instance["sector_upper"]
    risk_aversion = float(instance["risk_aversion"])
    transaction_penalty = float(instance["transaction_penalty"])
    turnover_limit = float(instance["turnover_limit"])

    w = np.clip(w_prev.copy(), lower, upper)
    w = _enforce_sum_and_bounds(w, lower, upper)

    eps = 1e-4
    for t in range(250):
        step = 0.08 / np.sqrt(t + 1.0)
        delta = w - w_prev
        smooth_sign = delta / np.sqrt(delta * delta + eps)
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

        w = w + step * grad
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_sector_bounds(w, sector_ids, sector_lower, sector_upper, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sum_and_bounds(w, lower, upper)

    return {"weights": w}


def solve_instance(instance: dict) -> dict:
    if HAS_CVXPY:
        result = solve_instance_cvxpy(instance)
        if result is not None:
            return result
    return solve_instance_heuristic(instance)
# EVOLVE-BLOCK-END
