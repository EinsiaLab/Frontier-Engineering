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
    # Project to simplex for budget constraint
    w = _project_to_simplex(w)
    # Then respect bounds
    w = np.minimum(np.maximum(w, lower), upper)
    # Final normalization if needed
    s = w.sum()
    if s > 0 and abs(s - 1.0) > 1e-8:
        w = w / s
        w = np.minimum(np.maximum(w, lower), upper)
    return w


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
    for _ in range(8):  # Increased iterations for better convergence
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


def _enforce_factor_exposure(
    w: np.ndarray,
    factor_loadings: np.ndarray,
    factor_lower: np.ndarray,
    factor_upper: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iterations: int = 5,
) -> np.ndarray:
    """Enforce factor exposure constraints."""
    w = w.copy()
    n_factors = factor_loadings.shape[1]
    
    for _ in range(max_iterations):
        changed = False
        exposure = factor_loadings.T @ w
        
        for k in range(n_factors):
            if exposure[k] < factor_lower[k] - 1e-10:
                # Need to increase exposure: shift weight from low-loading to high-loading assets
                diff = factor_lower[k] - exposure[k]
                factor_vec = factor_loadings[:, k]
                
                # Find assets within bounds that can help
                can_increase = (w < upper - 1e-12) & (factor_vec > 0)
                can_decrease = (w > lower + 1e-12) & (factor_vec < 0)
                
                if np.any(can_increase) or np.any(can_decrease):
                    # Calculate optimal shift
                    total_loading = np.abs(factor_vec[can_increase]).sum() + np.abs(factor_vec[can_decrease]).sum()
                    if total_loading > 1e-12:
                        # Shift weight proportionally to factor loading
                        shift = diff * 0.3  # Conservative shift size
                        
                        # Increase weights of high-loading assets
                        inc_assets = can_increase & (factor_vec > 0)
                        if np.any(inc_assets):
                            room = upper[inc_assets] - w[inc_assets]
                            total_room = room.sum()
                            if total_room > 0:
                                adj = np.zeros_like(w)
                                adj[inc_assets] = np.minimum(room, shift * factor_vec[inc_assets] / (np.abs(factor_vec[inc_assets]).sum() + 1e-12))
                                w = w + adj
                                changed = True
                        
                        # Decrease weights of negative-loading assets
                        dec_assets = can_decrease & (factor_vec < 0)
                        if np.any(dec_assets):
                            room = w[dec_assets] - lower[dec_assets]
                            total_room = room.sum()
                            if total_room > 0:
                                adj = np.zeros_like(w)
                                adj[dec_assets] = np.minimum(room, -shift * factor_vec[dec_assets] / (np.abs(factor_vec[dec_assets]).sum() + 1e-12))
                                w = w - adj
                                changed = True
            
            elif exposure[k] > factor_upper[k] + 1e-10:
                # Need to decrease exposure: shift weight from high-loading to low-loading assets
                diff = exposure[k] - factor_upper[k]
                factor_vec = factor_loadings[:, k]
                
                # Find assets within bounds that can help
                can_decrease = (w > lower + 1e-12) & (factor_vec > 0)
                can_increase = (w < upper - 1e-12) & (factor_vec < 0)
                
                if np.any(can_increase) or np.any(can_decrease):
                    # Calculate optimal shift
                    total_loading = np.abs(factor_vec[can_decrease]).sum() + np.abs(factor_vec[can_increase]).sum()
                    if total_loading > 1e-12:
                        # Shift weight proportionally to factor loading
                        shift = diff * 0.3  # Conservative shift size
                        
                        # Decrease weights of high-loading assets
                        dec_assets = can_decrease & (factor_vec > 0)
                        if np.any(dec_assets):
                            room = w[dec_assets] - lower[dec_assets]
                            if room.sum() > 0:
                                adj = np.zeros_like(w)
                                adj[dec_assets] = np.minimum(room, shift * factor_vec[dec_assets] / (np.abs(factor_vec[dec_assets]).sum() + 1e-12))
                                w = w - adj
                                changed = True
                        
                        # Increase weights of negative-loading assets
                        inc_assets = can_increase & (factor_vec < 0)
                        if np.any(inc_assets):
                            room = upper[inc_assets] - w[inc_assets]
                            if room.sum() > 0:
                                adj = np.zeros_like(w)
                                adj[inc_assets] = np.minimum(room, -shift * factor_vec[inc_assets] / (np.abs(factor_vec[inc_assets]).sum() + 1e-12))
                                w = w + adj
                                changed = True
        
        # Clip to bounds
        w = np.minimum(np.maximum(w, lower), upper)
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

    # Initialize with previous weights clipped to bounds and normalized
    w = np.clip(w_prev.copy(), lower, upper)
    w = _enforce_sum_and_bounds(w, lower, upper)
    
    # Use faster convergence with adaptive step size and cosine annealing
    eps = 1e-6  # Slightly larger epsilon for numerical stability
    base_step = 0.12  # Slightly larger base step for faster convergence
    
    for t in range(200):  # More iterations for better convergence
        # Cosine annealing for better convergence
        step = base_step * (1 + np.cos(np.pi * t / 200)) / 2 / np.sqrt(t + 1.0)
        
        # Compute gradient more efficiently with smoother L1 approximation
        w_diff = w - w_prev
        smooth_sign = np.tanh(w_diff / 1e-5)  # Smoother approximation with better gradient
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

        # Gradient ascent step
        w = w + step * grad
        
        # Apply constraints in optimal order: bounds → sector → factor → turnover → budget
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_sector_bounds(w, sector_ids, sector_lower, sector_upper, lower, upper)
        w = _enforce_factor_exposure(w, factor_loadings, factor_lower, factor_upper, lower, upper)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sum_and_bounds(w, lower, upper)

    # Final refinement: more aggressive projection onto all constraints
    for _ in range(3):  # Multiple rounds of constraint enforcement
        w = _enforce_bounds(w, lower, upper)
        w = _enforce_sector_bounds(w, sector_ids, sector_lower, sector_upper, lower, upper)
        w = _enforce_factor_exposure(w, factor_loadings, factor_lower, factor_upper, lower, upper, max_iterations=10)
        w = _enforce_turnover(w, w_prev, turnover_limit)
        w = _enforce_sum_and_bounds(w, lower, upper)

    return {"weights": w}
# EVOLVE-BLOCK-END
