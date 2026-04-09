# EVOLVE-BLOCK-START
import numpy as np
import cvxpy as cp


def solve_instance(instance: dict) -> dict:
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

    N = mu.shape[0]

    # Symmetrize covariance
    cov = (cov + cov.T) * 0.5

    # Cholesky factorization for better numerical conditioning
    try:
        L = np.linalg.cholesky(cov + 1e-14 * np.eye(N))
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-14)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    # Decision variables
    w = cp.Variable(N)
    t = cp.Variable(N)  # auxiliary for L1: t_i >= |w_i - w_prev_i|

    # Objective using Cholesky: ||L'w||^2 = w'Cov w
    Lw = L.T @ w
    objective = cp.Maximize(
        mu @ w - risk_aversion * cp.sum_squares(Lw) - transaction_penalty * cp.sum(t)
    )

    # Constraints
    constraints = [
        cp.sum(w) == 1,          # budget
        w >= lower,               # lower bounds
        w <= upper,               # upper bounds
        t >= w - w_prev,          # L1 linearization
        t >= w_prev - w,          # L1 linearization
        t >= 0,                   # non-negativity of abs values
        cp.sum(t) <= turnover_limit,  # turnover
    ]

    # Sector constraints - use vectorized approach
    unique_sectors = np.unique(sector_ids)
    for s in unique_sectors:
        idx = np.where(sector_ids == s)[0]
        s_int = int(s)
        if s_int in sector_lower:
            constraints.append(cp.sum(w[idx]) >= sector_lower[s_int])
        if s_int in sector_upper:
            constraints.append(cp.sum(w[idx]) <= sector_upper[s_int])

    # Factor exposure constraints - vectorized
    if "factor_loadings" in instance and instance["factor_loadings"] is not None:
        factor_loadings = np.asarray(instance["factor_loadings"], dtype=float)
        if factor_loadings.size > 0:
            factor_lower_arr = np.asarray(instance["factor_lower"], dtype=float)
            factor_upper_arr = np.asarray(instance["factor_upper"], dtype=float)
            exposure = factor_loadings.T @ w
            if factor_lower_arr.size == factor_loadings.shape[1]:
                constraints.append(exposure >= factor_lower_arr)
            if factor_upper_arr.size == factor_loadings.shape[1]:
                constraints.append(exposure <= factor_upper_arr)

    prob = cp.Problem(objective, constraints)

    solved = False

    # Phase 1: CLARABEL with moderate tolerances (fast convergence)
    if not solved:
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=500,
                       tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                       tol_feas=1e-10)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                solved = True
        except Exception:
            pass

    # Phase 2: Re-solve CLARABEL with extremely tight tolerances
    # The problem compilation is cached, so this is fast
    if solved:
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=5000,
                       tol_gap_abs=1e-15, tol_gap_rel=1e-15,
                       tol_feas=1e-15)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                solved = True
        except Exception:
            pass

    # Fallback: SCS with high accuracy
    if not solved:
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-12)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                solved = True
        except Exception:
            pass

    # Fallback: ECOS
    if not solved:
        try:
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=500, abstol=1e-10, reltol=1e-10)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                solved = True
        except Exception:
            pass

    if w.value is not None:
        weights = np.array(w.value).flatten()
        # Minimal post-processing for numerical noise
        weights = np.clip(weights, lower, upper)
        # Re-normalize budget constraint
        residual = 1.0 - weights.sum()
        if abs(residual) > 1e-14:
            if residual > 0:
                room = upper - weights
            else:
                room = weights - lower
            room = np.maximum(room, 0.0)
            total_room = room.sum()
            if total_room > 1e-14:
                weights += residual * (room / total_room)
                weights = np.clip(weights, lower, upper)
        return {"weights": weights}

    # Fallback: return clipped w_prev
    weights = np.clip(w_prev.copy(), lower, upper)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    else:
        weights = np.ones(N) / N
    return {"weights": weights}
# EVOLVE-BLOCK-END