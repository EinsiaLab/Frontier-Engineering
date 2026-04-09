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


def _enforce_factor_bounds(
    w: np.ndarray,
    factor_loadings: np.ndarray,
    factor_lower: np.ndarray,
    factor_upper: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Adjust weights to satisfy factor exposure constraints with adaptive damping."""
    w = w.copy()
    n_factors = factor_loadings.shape[1]
    
    for iteration in range(20):
        changed = False
        exposure = factor_loadings.T @ w
        damping = 0.95 ** iteration  # Slower decay for more aggressive adjustments
        
        for k in range(n_factors):
            lo, hi = factor_lower[k], factor_upper[k]
            loadings = factor_loadings[:, k]
            
            if exposure[k] > hi + 1e-9:
                excess = exposure[k] - hi
                pos_mask = loadings > 1e-10
                neg_mask = loadings < -1e-10
                if np.any(pos_mask):
                    room = np.maximum(w[pos_mask] - lower[pos_mask], 0)
                    impact = room * loadings[pos_mask]
                    cap = impact.sum()
                    if cap > 1e-12:
                        scale = min(1.0, excess / cap) * damping
                        adj = scale * impact / (np.abs(loadings[pos_mask]).sum() + 1e-12)
                        w[pos_mask] = np.maximum(lower[pos_mask], w[pos_mask] - adj)
                        changed = True
                if np.any(neg_mask):
                    room = np.maximum(upper[neg_mask] - w[neg_mask], 0)
                    impact = room * (-loadings[neg_mask])
                    cap = impact.sum()
                    if cap > 1e-12:
                        scale = min(1.0, excess / cap) * damping
                        adj = scale * impact / (np.abs(loadings[neg_mask]).sum() + 1e-12)
                        w[neg_mask] = np.minimum(upper[neg_mask], w[neg_mask] + adj)
                        changed = True
                        
            elif exposure[k] < lo - 1e-9:
                need = lo - exposure[k]
                pos_mask = loadings > 1e-10
                neg_mask = loadings < -1e-10
                if np.any(pos_mask):
                    room = np.maximum(upper[pos_mask] - w[pos_mask], 0)
                    impact = room * loadings[pos_mask]
                    cap = impact.sum()
                    if cap > 1e-12:
                        scale = min(1.0, need / cap) * damping
                        adj = scale * impact / (np.abs(loadings[pos_mask]).sum() + 1e-12)
                        w[pos_mask] = np.minimum(upper[pos_mask], w[pos_mask] + adj)
                        changed = True
                if np.any(neg_mask):
                    room = np.maximum(w[neg_mask] - lower[neg_mask], 0)
                    impact = room * (-loadings[neg_mask])
                    cap = impact.sum()
                    if cap > 1e-12:
                        scale = min(1.0, need / cap) * damping
                        adj = scale * impact / (np.abs(loadings[neg_mask]).sum() + 1e-12)
                        w[neg_mask] = np.maximum(lower[neg_mask], w[neg_mask] - adj)
                        changed = True
        
        w = np.clip(w, lower, upper)
        s = w.sum()
        if s > 1e-12:
            w = w / s
        else:
            w = np.ones_like(w) / w.size
        
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
    w = _enforce_factor_bounds(w, factor_loadings, factor_lower, factor_upper, lower, upper)

    velocity = np.zeros_like(w)
    best_w, best_obj = w.copy(), -np.inf
    best_feasible_w, best_feasible_obj = w.copy(), -np.inf

    def compute_penalty(wt):
        p = 0.0
        exposure = factor_loadings.T @ wt
        p += np.maximum(0, factor_lower - exposure).sum() + np.maximum(0, exposure - factor_upper).sum()
        for s in np.unique(sector_ids):
            idx = sector_ids == int(s)
            sec_sum = wt[idx].sum()
            p += max(0, sector_lower.get(int(s), 0) - sec_sum) + max(0, sec_sum - sector_upper.get(int(s), 1))
        p += max(0, np.abs(wt - w_prev).sum() - turnover_limit)
        return p

    # Phase 1: Coarse optimization with adaptive smoothing and Nesterov-like momentum
    for t in range(350):
        eps = max(1e-6, 1e-4 * (1 - t / 400))
        step = 0.12 / (1 + 0.003 * t)
        delta = w - w_prev
        # Huber-like smooth approximation for better gradient behavior
        smooth_sign = np.where(np.abs(delta) < eps, delta / eps, np.sign(delta))
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

        velocity = 0.75 * velocity + step * grad
        w = w + velocity
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sector_bounds(w, sector_ids, sector_lower, sector_upper, lower, upper)
        w = _enforce_factor_bounds(w, factor_loadings, factor_lower, factor_upper, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sum_and_bounds(w, lower, upper)

        obj = mu @ w - risk_aversion * w @ cov @ w - transaction_penalty * np.abs(w - w_prev).sum()
        if obj > best_obj:
            best_obj, best_w = obj, w.copy()
        pen = compute_penalty(w)
        if pen < 1e-6 and obj > best_feasible_obj:
            best_feasible_obj, best_feasible_w = obj, w.copy()

    # Phase 2: Fine-tuning from best solution with adaptive restart
    w = best_w.copy()
    velocity = np.zeros_like(w)
    for t in range(250):
        eps = max(1e-7, 5e-5 * (1 - t / 300))
        step = 0.05 / (1 + 0.008 * t)
        delta = w - w_prev
        smooth_sign = np.where(np.abs(delta) < eps, delta / eps, np.sign(delta))
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

        velocity = 0.6 * velocity + step * grad
        w = w + velocity
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sector_bounds(w, sector_ids, sector_lower, sector_upper, lower, upper)
        w = _enforce_factor_bounds(w, factor_loadings, factor_lower, factor_upper, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sum_and_bounds(w, lower, upper)

        obj = mu @ w - risk_aversion * w @ cov @ w - transaction_penalty * np.abs(w - w_prev).sum()
        pen = compute_penalty(w)
        if obj > best_obj:
            best_obj, best_w = obj, w.copy()
        if pen < 1e-6 and obj > best_feasible_obj:
            best_feasible_obj, best_feasible_w = obj, w.copy()

    # Phase 3: Very fine-tuning with minimal momentum
    w = best_w.copy()
    velocity = np.zeros_like(w)
    for t in range(150):
        eps = 1e-7
        step = 0.02 / (1 + 0.01 * t)
        delta = w - w_prev
        smooth_sign = np.where(np.abs(delta) < eps, delta / eps, np.sign(delta))
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

        velocity = 0.3 * velocity + step * grad
        w = w + velocity
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sector_bounds(w, sector_ids, sector_lower, sector_upper, lower, upper)
        w = _enforce_factor_bounds(w, factor_loadings, factor_lower, factor_upper, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sum_and_bounds(w, lower, upper)

        obj = mu @ w - risk_aversion * w @ cov @ w - transaction_penalty * np.abs(w - w_prev).sum()
        pen = compute_penalty(w)
        if obj > best_obj:
            best_obj, best_w = obj, w.copy()
        if pen < 1e-6 and obj > best_feasible_obj:
            best_feasible_obj, best_feasible_w = obj, w.copy()

    result = best_feasible_w if best_feasible_obj > -np.inf else best_w
    for _ in range(10):
        result = _enforce_bounds(result, lower, upper)
        result = _enforce_sector_bounds(result, sector_ids, sector_lower, sector_upper, lower, upper)
        result = _enforce_factor_bounds(result, factor_loadings, factor_lower, factor_upper, lower, upper)
        result = _enforce_turnover(result, w_prev, turnover_limit)
        result = _enforce_sum_and_bounds(result, lower, upper)

    return {"weights": result}
# EVOLVE-BLOCK-END
