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
    return np.clip(w, lower, upper)


def _enforce_sum_and_bounds(w: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    w = _enforce_bounds(w, lower, upper)
    for _ in range(50):
        gap = 1.0 - w.sum()
        if abs(gap) < 1e-12:
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


def _enforce_turnover(w: np.ndarray, w_prev: np.ndarray, turnover_limit: float,
                      lower: np.ndarray = None, upper: np.ndarray = None) -> np.ndarray:
    delta = w - w_prev
    turn = np.abs(delta).sum()
    if turn <= turnover_limit + 1e-12:
        return w
    scale = turnover_limit / max(turn, 1e-12)
    w_scaled = w_prev + scale * delta
    if lower is not None and upper is not None:
        for _ in range(10):
            violations = (w_scaled < lower - 1e-10) | (w_scaled > upper + 1e-10)
            if not np.any(violations):
                break
            scale *= 0.9
            w_scaled = w_prev + scale * delta
        w_scaled = np.clip(w_scaled, lower, upper)
    return w_scaled


def _enforce_sector_bounds(w: np.ndarray, sector_ids: np.ndarray, sector_lower: dict,
                           sector_upper: dict, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    w = w.copy()
    sectors = np.unique(sector_ids)
    for _ in range(8):
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


def _enforce_factor_bounds(w: np.ndarray, factor_loadings: np.ndarray, factor_lower: np.ndarray,
                           factor_upper: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if factor_loadings is None or factor_loadings.shape[1] == 0:
        return w
    w = w.copy()
    n_factors = factor_loadings.shape[1]
    for _ in range(5):
        changed = False
        for k in range(n_factors):
            exposure = np.dot(factor_loadings[:, k], w)
            lo = factor_lower[k]
            hi = factor_upper[k]
            if exposure > hi + 1e-10:
                excess = exposure - hi
                sens = factor_loadings[:, k]
                room = w - lower
                pos_sens = np.maximum(sens, 0)
                cap = np.dot(pos_sens, room)
                if cap > 1e-12:
                    adjust = np.minimum(room, excess * room * pos_sens / cap)
                    w -= adjust
                    changed = True
            elif exposure < lo - 1e-10:
                need = lo - exposure
                sens = factor_loadings[:, k]
                room = upper - w
                neg_sens = np.maximum(-sens, 0)
                cap = np.dot(neg_sens, room)
                if cap > 1e-12:
                    adjust = np.minimum(room, need * room * neg_sens / cap)
                    w += adjust
                    changed = True
        w = _enforce_sum_and_bounds(w, lower, upper)
        if not changed:
            break
    return w


def _check_feasibility(w: np.ndarray, w_prev: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                       turnover_limit: float, sector_ids: np.ndarray, sector_lower: dict,
                       sector_upper: dict, factor_loadings=None, factor_lower=None,
                       factor_upper=None, tol: float = 1e-6) -> bool:
    if abs(w.sum() - 1.0) > tol:
        return False
    if np.any(w < lower - tol) or np.any(w > upper + tol):
        return False
    if np.abs(w - w_prev).sum() > turnover_limit + tol:
        return False
    for s in np.unique(sector_ids):
        idx = np.where(sector_ids == s)[0]
        total = w[idx].sum()
        lo = sector_lower.get(int(s), 0.0)
        hi = sector_upper.get(int(s), 1.0)
        if total < lo - tol or total > hi + tol:
            return False
    if factor_loadings is not None and factor_lower is not None:
        for k in range(factor_loadings.shape[1]):
            exp = np.dot(factor_loadings[:, k], w)
            if exp < factor_lower[k] - tol or exp > factor_upper[k] + tol:
                return False
    return True


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

    factor_loadings = instance.get("factor_loadings", None)
    if factor_loadings is not None:
        factor_loadings = np.asarray(factor_loadings, dtype=float)
    factor_lower = instance.get("factor_lower", None)
    factor_upper = instance.get("factor_upper", None)
    if factor_lower is not None:
        factor_lower = np.asarray(factor_lower, dtype=float)
    if factor_upper is not None:
        factor_upper = np.asarray(factor_upper, dtype=float)

    n = len(mu)
    sectors = np.unique(sector_ids)
    sector_idx = {s: np.where(sector_ids == s)[0] for s in sectors}
    n_factors = factor_loadings.shape[1] if factor_loadings is not None else 0

    def compute_objective(w):
        delta = w - w_prev
        return np.dot(mu, w) - risk_aversion * np.dot(w, cov @ w) - transaction_penalty * np.abs(delta).sum()

    def project_simplex_bounded(w, lo, hi):
        w = np.clip(w, lo, hi)
        for _ in range(30):
            gap = 1.0 - w.sum()
            if abs(gap) < 1e-12:
                break
            free = (w > lo + 1e-12) & (w < hi - 1e-12)
            if not np.any(free):
                break
            w[free] += gap / free.sum()
            w = np.clip(w, lo, hi)
        s = w.sum()
        return w / s if s > 0 else np.ones(n) / n

    def project_feasible(w, max_iters=15):
        w = w.copy()
        for _ in range(max_iters):
            w = np.clip(w, lower, upper)
            w = project_simplex_bounded(w, lower, upper)

            delta = w - w_prev
            turn = np.abs(delta).sum()
            if turn > turnover_limit + 1e-10:
                scale = turnover_limit / turn
                w = w_prev + scale * delta
                w = np.clip(w, lower, upper)
                w = project_simplex_bounded(w, lower, upper)

            for s in sectors:
                idx = sector_idx[s]
                total = w[idx].sum()
                lo_s = sector_lower.get(int(s), 0.0)
                hi_s = sector_upper.get(int(s), 1.0)
                if total > hi_s + 1e-10:
                    excess = total - hi_s
                    room = w[idx] - lower[idx]
                    cap = room.sum()
                    if cap > 1e-12:
                        w[idx] -= np.minimum(room, excess * room / cap)
                elif total < lo_s - 1e-10:
                    need = lo_s - total
                    room = upper[idx] - w[idx]
                    cap = room.sum()
                    if cap > 1e-12:
                        w[idx] += np.minimum(room, need * room / cap)
            w = project_simplex_bounded(w, lower, upper)

            if factor_loadings is not None:
                for k in range(n_factors):
                    exp = np.dot(factor_loadings[:, k], w)
                    if exp > factor_upper[k] + 1e-10:
                        excess = exp - factor_upper[k]
                        sens = factor_loadings[:, k]
                        room = w - lower
                        pos_sens = np.maximum(sens, 0)
                        cap = np.dot(pos_sens, room)
                        if cap > 1e-12:
                            w -= np.minimum(room, excess * room * pos_sens / cap)
                    elif exp < factor_lower[k] - 1e-10:
                        need = factor_lower[k] - exp
                        sens = factor_loadings[:, k]
                        room = upper - w
                        neg_sens = np.maximum(-sens, 0)
                        cap = np.dot(neg_sens, room)
                        if cap > 1e-12:
                            w += np.minimum(room, need * room * neg_sens / cap)
                w = project_simplex_bounded(w, lower, upper)
        return w

    def compute_gradient(w, lambdas, rho, eps=1e-6):
        delta = w - w_prev
        smooth_sign = delta / np.sqrt(delta * delta + eps)
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * smooth_sign

        lam_idx = 0
        c_sum = w.sum() - 1.0
        grad -= (lambdas[lam_idx] + rho * c_sum) * np.ones(n)
        lam_idx += 1

        turn = np.abs(delta).sum()
        if turn > turnover_limit:
            grad -= (lambdas[lam_idx] + rho * (turn - turnover_limit)) * smooth_sign
        lam_idx += 1

        for s in sectors:
            idx = sector_idx[s]
            total = w[idx].sum()
            lo_s = sector_lower.get(int(s), 0.0)
            hi_s = sector_upper.get(int(s), 1.0)
            if total < lo_s:
                grad[idx] += lambdas[lam_idx] + rho * (lo_s - total)
            lam_idx += 1
            if total > hi_s:
                grad[idx] -= lambdas[lam_idx] + rho * (total - hi_s)
            lam_idx += 1

        if factor_loadings is not None:
            for k in range(n_factors):
                exp = np.dot(factor_loadings[:, k], w)
                sens = factor_loadings[:, k]
                if exp < factor_lower[k]:
                    grad += (lambdas[lam_idx] + rho * (factor_lower[k] - exp)) * sens
                lam_idx += 1
                if exp > factor_upper[k]:
                    grad -= (lambdas[lam_idx] + rho * (exp - factor_upper[k])) * sens
                lam_idx += 1
        return grad

    def count_constraints():
        n_constr = 2 + 2 * len(sectors) + 2 * n_factors
        return n_constr

    def optimize_al(w_init, n_outer=12, n_inner=80, rho_init=1.0):
        w = project_feasible(w_init.copy(), max_iters=5)
        n_constr = count_constraints()
        lambdas = np.zeros(n_constr)
        rho = rho_init

        best_w = w.copy()
        best_obj = compute_objective(w)

        # Nesterov momentum state
        v = np.zeros_like(w)
        beta = 0.75
        stagnation = 0

        for outer in range(n_outer):
            eps = max(1e-8, 1e-4 * (0.92 ** outer))
            step = 0.08 / (1 + 0.03 * outer)

            for inner in range(n_inner):
                # Nesterov lookahead
                w_look = w + beta * v

                grad = compute_gradient(w_look, lambdas, rho, eps)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 8.0:
                    grad = grad * 8.0 / grad_norm

                # Momentum update
                v_new = beta * v + step * grad
                w_new = w + v_new
                w_new = project_feasible(w_new, max_iters=2)

                # Check improvement
                obj_new = compute_objective(w_new)
                obj_old = compute_objective(w)

                if obj_new > obj_old - 1e-10:
                    v = v_new
                    w = w_new
                    stagnation = 0
                else:
                    # Backtrack with reduced momentum
                    for bt in range(4):
                        v_bt = beta * v * 0.5 + step * grad * (0.5 ** bt)
                        w_bt = w + v_bt
                        w_bt = project_feasible(w_bt, max_iters=2)
                        if compute_objective(w_bt) > obj_old - 1e-10:
                            v = v_bt
                            w = w_bt
                            break
                    else:
                        # Reset momentum on repeated failure
                        v = np.zeros_like(w)
                        stagnation += 1

            # Adaptive restart on stagnation
            if stagnation > 15:
                v = np.zeros_like(w)
                beta = max(0.4, beta * 0.7)
                stagnation = 0
                # Random perturbation to escape local optimum
                perturb = np.random.randn(n) * 0.01
                w_pert = project_feasible(w + perturb, max_iters=3)
                if compute_objective(w_pert) > compute_objective(w) - 0.01:
                    w = w_pert

            # Dual updates
            lam_idx = 0
            lambdas[lam_idx] += rho * (w.sum() - 1.0)
            lam_idx += 1
            lambdas[lam_idx] += rho * max(0, np.abs(w - w_prev).sum() - turnover_limit)
            lam_idx += 1

            for s in sectors:
                idx = sector_idx[s]
                total = w[idx].sum()
                lo_s = sector_lower.get(int(s), 0.0)
                hi_s = sector_upper.get(int(s), 1.0)
                lambdas[lam_idx] += rho * max(0, lo_s - total)
                lam_idx += 1
                lambdas[lam_idx] += rho * max(0, total - hi_s)
                lam_idx += 1

            if factor_loadings is not None:
                for k in range(n_factors):
                    exp = np.dot(factor_loadings[:, k], w)
                    lambdas[lam_idx] += rho * max(0, factor_lower[k] - exp)
                    lam_idx += 1
                    lambdas[lam_idx] += rho * max(0, exp - factor_upper[k])
                    lam_idx += 1

            obj = compute_objective(w)
            if obj > best_obj:
                best_obj = obj
                best_w = w.copy()
                beta = min(0.85, beta + 0.02)

            rho = min(rho * 1.4, 80.0)
        return best_w, best_obj

    init_points = []

    # 1. Previous weights (baseline)
    w_prev_clipped = np.clip(w_prev.copy(), lower, upper)
    if w_prev_clipped.sum() > 1e-10:
        w_prev_clipped = w_prev_clipped / w_prev_clipped.sum()
    init_points.append(np.clip(w_prev_clipped, lower, upper))

    # 2. Greedy allocation by expected return
    w_greedy = lower.copy()
    remaining = 1.0 - w_greedy.sum()
    for i in np.argsort(mu)[::-1]:
        add = min(remaining, upper[i] - w_greedy[i])
        w_greedy[i] += add
        remaining -= add
        if remaining < 1e-10:
            break
    init_points.append(w_greedy)

    # 3. Mid-point allocation
    w_mid = (lower + upper) / 2
    w_mid = w_mid / w_mid.sum() if w_mid.sum() > 0 else np.ones(n) / n
    init_points.append(np.clip(w_mid, lower, upper))

    # 4. Inverse variance (minimum variance proxy)
    var_diag = np.diag(cov)
    w_invvar = 1.0 / (var_diag + 1e-10)
    w_invvar = w_invvar / w_invvar.sum()
    w_invvar = np.clip(w_invvar, lower, upper)
    if w_invvar.sum() > 0:
        w_invvar = w_invvar / w_invvar.sum()
    init_points.append(w_invvar)

    # 5. Sharpe-like (risk-adjusted return)
    risk_adj = mu / (np.sqrt(np.maximum(var_diag, 1e-12)) + 1e-10)
    w_sharpe = lower.copy()
    remaining = 1.0 - w_sharpe.sum()
    for i in np.argsort(risk_adj)[::-1]:
        add = min(remaining, upper[i] - w_sharpe[i])
        w_sharpe[i] += add
        remaining -= add
        if remaining < 1e-10:
            break
    init_points.append(w_sharpe)

    # 6. Turnover-constrained greedy (blend of w_prev and greedy)
    w_turn_greedy = w_prev.copy()
    delta_greedy = w_greedy - w_prev
    turn_greedy = np.abs(delta_greedy).sum()
    if turn_greedy > turnover_limit + 1e-10:
        scale = turnover_limit / turn_greedy
        w_turn_greedy = w_prev + scale * delta_greedy
    w_turn_greedy = np.clip(w_turn_greedy, lower, upper)
    if w_turn_greedy.sum() > 1e-10:
        w_turn_greedy = w_turn_greedy / w_turn_greedy.sum()
    init_points.append(np.clip(w_turn_greedy, lower, upper))

    # 7. Turnover-constrained sharpe
    w_turn_sharpe = w_prev.copy()
    delta_sharpe = w_sharpe - w_prev
    turn_sharpe = np.abs(delta_sharpe).sum()
    if turn_sharpe > turnover_limit + 1e-10:
        scale = turnover_limit / turn_sharpe
        w_turn_sharpe = w_prev + scale * delta_sharpe
    w_turn_sharpe = np.clip(w_turn_sharpe, lower, upper)
    if w_turn_sharpe.sum() > 1e-10:
        w_turn_sharpe = w_turn_sharpe / w_turn_sharpe.sum()
    init_points.append(np.clip(w_turn_sharpe, lower, upper))

    # 8. Diversification-focused (inverse correlation)
    corr = np.corrcoef(cov) if n > 1 else np.array([[1.0]])
    avg_corr = np.mean(np.abs(corr - np.eye(n)), axis=1)
    w_divers = 1.0 / (avg_corr + 0.1)
    w_divers = w_divers / w_divers.sum()
    w_divers = np.clip(w_divers, lower, upper)
    if w_divers.sum() > 0:
        w_divers = w_divers / w_divers.sum()
    init_points.append(np.clip(w_divers, lower, upper))

    best_w = None
    best_obj = -np.inf

    for i, w_init in enumerate(init_points):
        rho_init = 0.3 + 0.4 * (i % 4)
        w_opt, obj_opt = optimize_al(w_init, n_outer=15, n_inner=70, rho_init=rho_init)
        if obj_opt > best_obj:
            best_obj = obj_opt
            best_w = w_opt.copy()

    # Second pass: refine the best solution
    if best_w is not None:
        w_refined, obj_refined = optimize_al(best_w, n_outer=10, n_inner=50, rho_init=2.0)
        if obj_refined > best_obj:
            best_obj = obj_refined
            best_w = w_refined.copy()

    w_final = project_feasible(best_w, max_iters=30)

    for repair_attempt in range(3):
        if _check_feasibility(w_final, w_prev, lower, upper, turnover_limit,
                              sector_ids, sector_lower, sector_upper,
                              factor_loadings, factor_lower, factor_upper):
            break
        if repair_attempt == 0:
            w_final = project_feasible(w_final, max_iters=50)
        elif repair_attempt == 1:
            w_final = project_feasible(best_w, max_iters=100)
        else:
            w_final = project_feasible(w_prev.copy(), max_iters=50)

    if not _check_feasibility(w_final, w_prev, lower, upper, turnover_limit,
                               sector_ids, sector_lower, sector_upper,
                               factor_loadings, factor_lower, factor_upper):
        w_safe = np.clip(w_prev.copy(), lower, upper)
        s = w_safe.sum()
        if s > 1e-10:
            w_safe = w_safe / s
        w_final = np.clip(w_safe, lower, upper)
        gap = 1.0 - w_final.sum()
        if abs(gap) > 1e-10:
            free = (w_final > lower + 1e-12) & (w_final < upper - 1e-12)
            if np.any(free):
                w_final[free] += gap / free.sum()
                w_final = np.clip(w_final, lower, upper)

    return {"weights": w_final}
# EVOLVE-BLOCK-END