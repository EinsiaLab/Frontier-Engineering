# EVOLVE-BLOCK-START
import numpy as np


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


def _enforce_sum_and_bounds(w: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    w = np.minimum(np.maximum(w, lower), upper)
    for _ in range(20):  # More iterations to ensure sum constraint is fully satisfied
        gap = 1.0 - w.sum()
        if abs(gap) < 1e-10:
            break
        free = (w > lower + 1e-12) & (w < upper - 1e-12)
        if not np.any(free):
            w = _project_to_simplex(w)
            w = np.minimum(np.maximum(w, lower), upper)
            continue
        w[free] += gap / free.sum()
        w = np.minimum(np.maximum(w, lower), upper)
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
    for _ in range(10):  # Extra iterations to fully enforce sector bounds
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


def _enforce_factor_bounds(
    w: np.ndarray,
    factor_loadings: np.ndarray,
    factor_lower: np.ndarray,
    factor_upper: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    w = w.copy()
    K = factor_loadings.shape[1]
    for _ in range(12):  # More iterations for factor constraints (highest 30x penalty weight)
        changed = False
        for k in range(K):
            loading = factor_loadings[:, k]
            exposure = loading @ w
            lo = factor_lower[k]
            hi = factor_upper[k]
            if exposure > hi + 1e-10:
                excess = exposure - hi
                adj = excess * loading / np.maximum(np.sum(loading ** 2), 1e-12)
                w -= adj
                w = np.minimum(np.maximum(w, lower), upper)
                changed = True
            elif exposure < lo - 1e-10:
                need = lo - exposure
                adj = need * loading / np.maximum(np.sum(loading ** 2), 1e-12)
                w += adj
                w = np.minimum(np.maximum(w, lower), upper)
                changed = True
        w = _enforce_sum_and_bounds(w, lower, upper)
        if not changed:
            break
    return w


def solve_instance(instance: dict) -> dict:
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

    w = np.clip(w_prev.copy(), lower, upper)
    w = _enforce_sum_and_bounds(w, lower, upper)
    vel = np.zeros_like(w)
    beta = 0.9  # Momentum coefficient for faster convergence
    eps = 1e-5  # Smaller epsilon for more accurate L1 penalty gradient approximation
    for t in range(300):  # More iterations for better convergence to optimal objective
        step = 0.11 / np.sqrt(t + 1.0)  # Adjusted step size to avoid overshooting optimum
        delta = w - w_prev
        smooth_sign = delta / np.sqrt(delta * delta + eps)
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

        # Nesterov momentum update improves convergence speed and quality
        vel = beta * vel + (1 - beta) * grad
        w_new = w + step * vel
        
        # Early stopping if weights stabilize to save compute
        if np.max(np.abs(w_new - w)) < 1e-8:
            break
        w = w_new
        w = np.minimum(np.maximum(w, lower), upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = np.minimum(np.maximum(w, lower), upper)
        w = _enforce_sector_bounds(
            w, sector_ids, sector_lower, sector_upper, lower, upper
        )
        w = _enforce_factor_bounds(
            w, factor_loadings, factor_lower, factor_upper, lower, upper
        )
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sum_and_bounds(w, lower, upper)

    # Final pass to eliminate any residual factor/sector/turnover/sum constraint violations
    # Enforce sector first, then factor to fix any drift from sector adjustments
    w = _enforce_sector_bounds(w, sector_ids, sector_lower, sector_upper, lower, upper)
    w = _enforce_factor_bounds(w, factor_loadings, factor_lower, factor_upper, lower, upper)
    w = _enforce_turnover(w, w_prev, turnover_limit)
    w = _enforce_sum_and_bounds(w, lower, upper)
    # Extra factor pass to ensure no residual violations from sum adjustment
    w = _enforce_factor_bounds(w, factor_loadings, factor_lower, factor_upper, lower, upper)
    w = _enforce_sum_and_bounds(w, lower, upper)

    return {"weights": w}
# EVOLVE-BLOCK-END
