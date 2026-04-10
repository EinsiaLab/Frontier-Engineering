# EVOLVE-BLOCK-START
import numpy as np


def solve_instance(instance: dict) -> dict:
    import cvxpy as cp

    mu = np.asarray(instance["mu"], dtype=float)
    cov = np.asarray(instance["cov"], dtype=float)
    w_prev = np.asarray(instance["w_prev"], dtype=float)
    lower = np.asarray(instance["lower"], dtype=float)
    upper = np.asarray(instance["upper"], dtype=float)
    sector_ids = np.asarray(instance["sector_ids"], dtype=int)
    sector_lower = instance["sector_lower"]
    sector_upper = instance["sector_upper"]
    factor_loadings = np.asarray(instance["factor_loadings"], dtype=float)
    factor_lower = np.asarray(instance["factor_lower"], dtype=float)
    factor_upper = np.asarray(instance["factor_upper"], dtype=float)
    risk_aversion = float(instance["risk_aversion"])
    transaction_penalty = float(instance["transaction_penalty"])
    turnover_limit = float(instance["turnover_limit"])

    n = mu.size
    cov = 0.5 * (cov + cov.T)
    w = cp.Variable(n)

    obj = cp.Maximize(
        mu @ w
        - risk_aversion * cp.quad_form(w, cov, assume_PSD=True)
        - transaction_penalty * cp.norm1(w - w_prev)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= lower,
        w <= upper,
        cp.norm1(w - w_prev) <= turnover_limit,
        factor_loadings.T @ w >= factor_lower,
        factor_loadings.T @ w <= factor_upper,
    ]

    for s, lo in sector_lower.items():
        idx = np.where(sector_ids == int(s))[0]
        constraints.append(cp.sum(w[idx]) >= float(lo))

    for s, hi in sector_upper.items():
        idx = np.where(sector_ids == int(s))[0]
        constraints.append(cp.sum(w[idx]) <= float(hi))

    prob = cp.Problem(obj, constraints)

    _eval = lambda wv: float(mu @ wv - risk_aversion * (wv @ cov @ wv) - transaction_penalty * np.abs(wv - w_prev).sum())

    solver_configs = [
        (cp.CLARABEL, {"tol_gap_abs": 1e-15, "tol_gap_rel": 1e-15, "tol_feas": 1e-15, "max_iter": 1000}),
        (cp.ECOS, {"max_iters": 2000, "abstol": 1e-12, "reltol": 1e-12, "feastol": 1e-12}),
        (cp.SCS, {"max_iters": 300000, "eps": 1e-12}),
    ]

    best_w = None
    best_obj = -np.inf
    for solver, kwargs in solver_configs:
        try:
            prob.solve(solver=solver, verbose=False, **kwargs)
            if prob.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
                wv = np.asarray(w.value, dtype=float).reshape(-1)
                ov = _eval(wv)
                if ov > best_obj:
                    best_obj = ov
                    best_w = wv
                if prob.status == "optimal":
                    break
        except Exception:
            continue

    if best_w is not None:
        return {"weights": best_w}

    w_fall = np.clip(w_prev.copy(), lower, upper)
    sv = w_fall.sum()
    return {"weights": w_fall / sv if sv > 0 else np.ones(n) / n}
# EVOLVE-BLOCK-END
