# EVOLVE-BLOCK-START
import numpy as np
import cvxpy as cp


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
    w = np.minimum(np.maximum(w, lower), upper)
    return w


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


# Solve the problem exactly with CVXPY – this respects all constraints
def solve_instance(instance: dict) -> dict:
    # ----- Data preparation -------------------------------------------------
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
    w = cp.Variable(n)

    # ----- Objective --------------------------------------------------------
    objective = cp.Maximize(
        mu @ w
        - risk_aversion * cp.quad_form(w, cov)
        - transaction_penalty * cp.norm1(w - w_prev)
    )

    # ----- Constraints -------------------------------------------------------
    constraints = [
        cp.sum(w) == 1,
        w >= lower,
        w <= upper,
        cp.norm1(w - w_prev) <= turnover_limit,
        factor_loadings.T @ w >= factor_lower,
        factor_loadings.T @ w <= factor_upper,
    ]

    # Sector exposure constraints
    for s, lo in sector_lower.items():
        idx = np.where(sector_ids == int(s))[0]
        constraints.append(cp.sum(w[idx]) >= float(lo))
    for s, hi in sector_upper.items():
        idx = np.where(sector_ids == int(s))[0]
        constraints.append(cp.sum(w[idx]) <= float(hi))

    # ----- Solve ------------------------------------------------------------
    # ----- Solve with tighter tolerances ---------------------------------
    prob = cp.Problem(objective, constraints)
    solved = False

    # Solve with a single high‑precision solver (ECOS). This is sufficient for
    # the problem instances and removes unnecessary complexity.
    try:
        # Solve with tighter tolerances and more iterations.
        # This gives a solution that is numerically closer to the true
        # convex optimum, shrinking the objective gap observed on a few
        # instances.
        prob.solve(
            solver=cp.ECOS,
            verbose=False,
            max_iters=20000,      # allow more interior‑point iterations
            abstol=1e-12,         # tighter absolute tolerance
            reltol=1e-12,         # tighter relative tolerance
            feastol=1e-12,        # tighter feasibility tolerance
        )
        if prob.status in {"optimal", "optimal_inaccurate"}:
            solved = True
    except Exception:
        solved = False

    # ----- Return the solver’s exact solution ------------------------------
    # The CVXPY model already enforces all constraints (bounds, sector,
    # factor exposure, turnover, and budget).  Returning the raw solution
    # preserves the optimal objective value and avoids unnecessary numerical
    # perturbations from the manual post‑processing steps above.
    if solved and w.value is not None:
        w_opt = np.asarray(w.value).reshape(-1)

        # ------------------------------------------------------------------
        # Post‑process the raw CVXPY solution.
        # --------------------------------------------------------------
        # • Enforce per‑asset bounds and the budget (sum==1) exactly.
        # • Re‑apply the turnover limit in case rounding introduced a tiny
        #   violation.  This step is cheap and preserves feasibility while
        #   keeping the objective essentially unchanged.
        # ------------------------------------------------------------------
        w_opt = _enforce_bounds(w_opt, lower, upper)
        w_opt = _enforce_sum_and_bounds(w_opt, lower, upper)
        w_opt = _enforce_turnover(w_opt, w_prev, turnover_limit)

        return {"weights": w_opt}

    # ----- Fallback (should rarely be needed) -------------------------------
    if not solved or w.value is None:
        fallback = np.clip(w_prev, lower, upper)
        fallback = fallback / fallback.sum()
        return {"weights": fallback}

    return {"weights": np.asarray(w.value).reshape(-1)}
# EVOLVE-BLOCK-END
