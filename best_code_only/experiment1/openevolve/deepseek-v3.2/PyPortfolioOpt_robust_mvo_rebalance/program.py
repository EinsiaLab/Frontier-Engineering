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
    gap = 1.0 - w.sum()
    if abs(gap) > 1e-10:
        free = (w > lower + 1e-12) & (w < upper - 1e-12)
        if np.any(free):
            w[free] += gap / free.sum()
            w = _enforce_bounds(w, lower, upper)
        else:
            w = _project_to_simplex(w)
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
    """Adjust weights to satisfy factor exposure constraints via iterative repair."""
    w = w.copy()
    K = factor_loadings.shape[1]
    exposure = factor_loadings.T @ w
    
    for k in range(K):
        lo = factor_lower[k]
        hi = factor_upper[k]
        if exposure[k] > hi + 1e-10:
            excess = exposure[k] - hi
            # We need to reduce exposure by excess amount
            # Compute sensitivity of each asset to this factor
            sens = factor_loadings[:, k]
            # Choose assets where we can decrease weight (w > lower) and have positive sensitivity
            # Actually, to reduce exposure, we can decrease weights on assets with positive sensitivity
            # or increase weights on assets with negative sensitivity
            # We'll focus on decreasing positive contributions first
            candidates = (w > lower + 1e-12) & (sens > 0)
            if np.any(candidates):
                # Distribute reduction proportionally to sensitivity and available room
                room = w[candidates] - lower[candidates]
                total_sens = sens[candidates].sum()
                if total_sens > 1e-12:
                    reduction = excess * sens[candidates] / total_sens
                    # Limit reduction to available room
                    w[candidates] -= np.minimum(reduction, room)
            # Check again
            new_exposure = factor_loadings[:, k] @ w
            if new_exposure > hi + 1e-10:
                remaining_excess = new_exposure - hi
                # Try increasing negative contributions
                neg_candidates = (w < upper - 1e-12) & (sens < 0)
                if np.any(neg_candidates):
                    room_up = upper[neg_candidates] - w[neg_candidates]
                    total_neg_sens = -sens[neg_candidates].sum()
                    if total_neg_sens > 1e-12:
                        addition = remaining_excess * (-sens[neg_candidates]) / total_neg_sens
                        w[neg_candidates] += np.minimum(addition, room_up)
        elif exposure[k] < lo - 1e-10:
            deficit = lo - exposure[k]
            sens = factor_loadings[:, k]
            # To increase exposure, increase weights on assets with positive sensitivity
            candidates = (w < upper - 1e-12) & (sens > 0)
            if np.any(candidates):
                room = upper[candidates] - w[candidates]
                total_sens = sens[candidates].sum()
                if total_sens > 1e-12:
                    addition = deficit * sens[candidates] / total_sens
                    w[candidates] += np.minimum(addition, room)
            new_exposure = factor_loadings[:, k] @ w
            if new_exposure < lo - 1e-10:
                remaining_deficit = lo - new_exposure
                # Try decreasing negative contributions
                neg_candidates = (w > lower + 1e-12) & (sens < 0)
                if np.any(neg_candidates):
                    room_down = w[neg_candidates] - lower[neg_candidates]
                    total_neg_sens = -sens[neg_candidates].sum()
                    if total_neg_sens > 1e-12:
                        reduction = remaining_deficit * (-sens[neg_candidates]) / total_neg_sens
                        w[neg_candidates] -= np.minimum(reduction, room_down)
    
    w = _enforce_sum_and_bounds(w, lower, upper)
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

    # Start with previous weights, but ensure they are feasible
    w = np.clip(w_prev.copy(), lower, upper)
    w = _enforce_sum_and_bounds(w, lower, upper)

    eps = 1e-8
    momentum = np.zeros_like(w)
    # Use more iterations with better step size schedule
    for t in range(500):
        # More gradual step size decay: start at 0.15, decay to 0.01
        if t < 50:
            step = 0.15 / (1.0 + 0.02 * t)
        elif t < 200:
            step = 0.05 / (1.0 + 0.01 * (t - 50))
        else:
            step = 0.01 / np.sqrt(t - 199)
        delta = w - w_prev
        # Smooth approximation to sign function using tanh (good for gradient)
        smooth_sign = np.tanh(delta / eps)
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign
        
        # Momentum with fixed decay 0.9 for early iterations, then 0.5 later
        if t < 300:
            momentum_decay = 0.9
        else:
            momentum_decay = 0.5
        momentum = momentum_decay * momentum + (1.0 - momentum_decay) * grad
        w = w + step * momentum
        
        # Apply constraints in a logical order
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_sum_and_bounds(w, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sector_bounds(
            w, sector_ids, sector_lower, sector_upper, lower, upper
        )
        w = _enforce_factor_bounds(
            w, factor_loadings, factor_lower, factor_upper, lower, upper
        )
        w = _enforce_sum_and_bounds(w, lower, upper)

    # Final repair cycles to ensure constraints are fully satisfied
    for _ in range(10):
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_sum_and_bounds(w, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sector_bounds(
            w, sector_ids, sector_lower, sector_upper, lower, upper
        )
        w = _enforce_factor_bounds(
            w, factor_loadings, factor_lower, factor_upper, lower, upper
        )
        w = _enforce_sum_and_bounds(w, lower, upper)
    
    return {"weights": w}
# EVOLVE-BLOCK-END
