# EVOLVE-BLOCK-START
import cvxpy as cp
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
    w = np.asarray(w, dtype=float)
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        w = _enforce_bounds(w, lower, upper)
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / w.size
    lo = np.min(w - upper)
    hi = np.max(w - lower)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        x = np.minimum(np.maximum(w - mid, lower), upper)
        if x.sum() > 1.0:
            lo = mid
        else:
            hi = mid
    return np.minimum(np.maximum(w - hi, lower), upper)


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


def solve_instance(instance: dict) -> dict:
    mu = np.asarray(instance["mu"], dtype=float)
    cov = 0.5 * (
        np.asarray(instance["cov"], dtype=float)
        + np.asarray(instance["cov"], dtype=float).T
    )
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
    diag = np.maximum(np.diag(cov), 1e-8)

    def score(x: np.ndarray) -> float:
        d = x - w_prev
        return float(
            mu @ x
            - risk_aversion * (x @ (cov @ x))
            - transaction_penalty * np.abs(d).sum()
        )

    def viol(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        v = abs(float(x.sum()) - 1.0)
        v = max(v, float(np.max(lower - x)), float(np.max(x - upper)))
        v = max(v, float(np.abs(x - w_prev).sum() - turnover_limit))
        if factor_lower.size:
            e = factor_loadings.T @ x
            v = max(v, float(np.max(factor_lower - e)), float(np.max(e - factor_upper)))
        for s, lo in sector_lower.items():
            v = max(v, float(lo - x[sector_ids == int(s)].sum()))
        for s, hi in sector_upper.items():
            v = max(v, float(x[sector_ids == int(s)].sum() - hi))
        return max(0.0, v)

    def tighten(x: np.ndarray) -> np.ndarray:
        x = _enforce_sum_and_bounds(np.asarray(x, dtype=float).reshape(-1), lower, upper)
        if viol(x) <= 1e-8:
            return x
        d = x - w_prev
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            y = w_prev + mid * d
            if viol(y) <= 1e-8:
                lo = mid
            else:
                hi = mid
        return w_prev + lo * d

    try:
        w = cp.Variable(n)
        cons = [
            cp.sum(w) == 1,
            w >= lower,
            w <= upper,
            cp.norm1(w - w_prev) <= turnover_limit,
            factor_loadings.T @ w >= factor_lower,
            factor_loadings.T @ w <= factor_upper,
        ]
        for s, lo in sector_lower.items():
            idx = np.where(sector_ids == int(s))[0]
            cons.append(cp.sum(w[idx]) >= float(lo))
        for s, hi in sector_upper.items():
            idx = np.where(sector_ids == int(s))[0]
            cons.append(cp.sum(w[idx]) <= float(hi))

        prob = cp.Problem(
            cp.Maximize(
                mu @ w
                - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
                - transaction_penalty * cp.norm1(w - w_prev)
            ),
            cons,
        )

        for solver, opts in (
            (cp.ECOS, {"abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10}),
            (cp.OSQP, {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 200000, "polish": True}),
            (cp.SCS, {"eps": 1e-7, "max_iters": 20000}),
        ):
            try:
                prob.solve(solver=solver, warm_start=True, verbose=False, **opts)
                if prob.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
                    x = tighten(np.asarray(w.value, dtype=float).reshape(-1))
                    if np.all(np.isfinite(x)) and viol(x) <= 1e-7:
                        return {"weights": x}
            except Exception:
                pass
    except Exception:
        pass

    def repair(x: np.ndarray) -> np.ndarray:
        x = _enforce_sum_and_bounds(np.asarray(x, dtype=float), lower, upper)
        for _ in range(3):
            old = x.copy()
            x = _enforce_sector_bounds(
                x, sector_ids, sector_lower, sector_upper, lower, upper
            )
            x = _enforce_turnover(x, w_prev, turnover_limit)
            x = _enforce_sum_and_bounds(x, lower, upper)
            if np.max(np.abs(x - old)) < 1e-11:
                break
        return tighten(x)

    w0 = repair(w_prev.copy())
    grad0 = mu - 2.0 * risk_aversion * (cov @ w0)
    scale = 1.0 / (1.0 + 2.0 * risk_aversion * diag)
    rem = max(1.0 - float(lower.sum()), 0.0)

    seeds = [w0, repair(w0 + 0.5 * scale * grad0), repair(w0 + scale * grad0)]

    pos = np.maximum(mu, 0.0)
    if rem > 0 and pos.sum() > 1e-12:
        seeds.append(repair(lower + rem * pos / pos.sum()))

    if rem > 0 and n <= 250:
        try:
            signal = np.linalg.lstsq(cov + 1e-6 * np.eye(n), mu, rcond=None)[0]
            signal = np.maximum(signal, 0.0)
            if signal.sum() > 1e-12:
                seeds.append(repair(lower + rem * signal / signal.sum()))
        except np.linalg.LinAlgError:
            pass

    seeds.append(repair(0.5 * (seeds[0] + seeds[-1])))

    best_w = w0
    best_val = score(best_w)

    for seed in seeds:
        w = repair(seed)
        prev = w.copy()
        val = score(w)
        if val > best_val:
            best_w, best_val = w.copy(), val
        fail = 0

        for t in range(120):
            delta = w - w_prev
            grad = mu - 2.0 * risk_aversion * (cov @ w)
            grad -= transaction_penalty * delta / np.sqrt(delta * delta + 1e-8)
            direction = scale * grad + 0.35 * (w - prev)
            prev = w.copy()
            base = 0.6 / np.sqrt(t + 2.0)

            cand_best = None
            cand_val = val
            for step in (1.5 * base, base, 0.5 * base):
                cand = repair(w + step * direction)
                s = score(cand)
                if s > cand_val + 1e-12:
                    cand_best, cand_val = cand, s

            if cand_best is None:
                fail += 1
                if fail >= 8:
                    break
                w = repair(0.7 * w + 0.3 * w0)
                prev = w.copy()
                val = score(w)
                continue

            fail = 0
            w, val = cand_best, cand_val
            if val > best_val:
                best_w, best_val = w.copy(), val

    return {"weights": repair(best_w)}
# EVOLVE-BLOCK-END
