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
    sector_data,
    sector_lower: dict,
    sector_upper: dict,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    w = w.copy()
    for _ in range(4):
        changed = False
        for s, idx in sector_data:
            total = w[idx].sum()
            lo = sector_lower.get(s, 0.0)
            hi = sector_upper.get(s, 1.0)
            if total > hi + 1e-10:
                room = w[idx] - lower[idx]
                cap = room.sum()
                if cap > 1e-12:
                    w[idx] -= np.minimum(room, (total - hi) * room / cap)
                    changed = True
            elif total < lo - 1e-10:
                room = upper[idx] - w[idx]
                cap = room.sum()
                if cap > 1e-12:
                    w[idx] += np.minimum(room, (lo - total) * room / cap)
                    changed = True
        w = _enforce_sum_and_bounds(w, lower, upper)
        if not changed:
            break
    return w


def _enforce_factor_bounds(w, F, fl, fu, lower, upper):
    if F.size == 0:
        return w
    w = w.copy()
    for _ in range(6):
        e = F.T @ w
        bad = False
        for k, x in enumerate(e):
            a = F[:, k]
            if x < fl[k] - 1e-10:
                need = fl[k] - x
                up = a > 1e-12
                dn = a < -1e-12
                ru = upper[up] - w[up]
                rd = w[dn] - lower[dn]
                zu = a[up] * ru
                zd = (-a[dn]) * rd
                su, sd = zu.sum(), zd.sum()
                tot = su + sd
                if tot > 1e-12:
                    if su > 1e-12:
                        w[up] += np.minimum(ru, 0.8 * need * zu / tot)
                    if sd > 1e-12:
                        w[dn] -= np.minimum(rd, 0.8 * need * zd / tot)
                    bad = True
            elif x > fu[k] + 1e-10:
                need = x - fu[k]
                dn = a > 1e-12
                up = a < -1e-12
                rd = w[dn] - lower[dn]
                ru = upper[up] - w[up]
                zd = a[dn] * rd
                zu = (-a[up]) * ru
                sd, su = zd.sum(), zu.sum()
                tot = su + sd
                if tot > 1e-12:
                    if sd > 1e-12:
                        w[dn] -= np.minimum(rd, 0.8 * need * zd / tot)
                    if su > 1e-12:
                        w[up] += np.minimum(ru, 0.8 * need * zu / tot)
                    bad = True
        w = _enforce_sum_and_bounds(w, lower, upper)
        if not bad:
            break
    return w


def solve_instance(instance: dict) -> dict:
    mu = np.asarray(instance["mu"], float)
    cov = np.asarray(instance["cov"], float)
    w_prev = np.asarray(instance["w_prev"], float)
    lower = np.asarray(instance["lower"], float)
    upper = np.asarray(instance["upper"], float)
    sector_ids = np.asarray(instance["sector_ids"], int)
    sector_lower = instance["sector_lower"]
    sector_upper = instance["sector_upper"]
    F = np.asarray(instance["factor_loadings"], float)
    fl = np.asarray(instance["factor_lower"], float)
    fu = np.asarray(instance["factor_upper"], float)
    risk_aversion = float(instance["risk_aversion"])
    transaction_penalty = float(instance["transaction_penalty"])
    turnover_limit = float(instance["turnover_limit"])

    groups = [(int(s), np.where(sector_ids == s)[0]) for s in np.unique(sector_ids)]

    ivar = 1.0 / (np.diag(cov) + 1e-8)
    tilt = mu * ivar
    tilt = tilt / (np.abs(tilt).sum() + 1e-12)
    base = np.clip(0.7 * w_prev + 0.3 * _enforce_sum_and_bounds(np.maximum(tilt, 0), lower, upper), lower, upper)
    w = _enforce_sector_bounds(base, groups, sector_lower, sector_upper, lower, upper)
    w = _enforce_factor_bounds(w, F, fl, fu, lower, upper)
    w = _enforce_turnover(w, w_prev, turnover_limit)
    w = _enforce_sum_and_bounds(w, lower, upper)
    w = _enforce_turnover(w, w_prev, turnover_limit)

    def obj(x):
        d = x - w_prev
        return mu @ x - risk_aversion * (x @ (cov @ x)) - transaction_penalty * np.abs(d).sum()

    best_w = w.copy()
    best_obj = obj(w)
    eps = 1e-4
    for t in range(120):
        step = 0.22 / np.sqrt(t + 5.0)
        d = w - w_prev
        grad = mu - 2.0 * risk_aversion * (cov @ w) - transaction_penalty * d / np.sqrt(d * d + eps)
        trial_best, trial_obj = w, obj(w)
        for a in (0.4, 0.8, 1.2, 1.8):
            cand = w + a * step * grad
            cand = _enforce_turnover(cand, w_prev, turnover_limit)
            cand = _enforce_sector_bounds(cand, groups, sector_lower, sector_upper, lower, upper)
            cand = _enforce_factor_bounds(cand, F, fl, fu, lower, upper)
            cand = _enforce_sum_and_bounds(cand, lower, upper)
            cand = _enforce_turnover(cand, w_prev, turnover_limit)
            cand = _enforce_sector_bounds(cand, groups, sector_lower, sector_upper, lower, upper)
            cand = _enforce_factor_bounds(cand, F, fl, fu, lower, upper)
            cand = _enforce_sum_and_bounds(cand, lower, upper)
            val = obj(cand)
            if val > trial_obj:
                trial_obj, trial_best = val, cand
        if trial_obj >= best_obj:
            best_obj, best_w, w = trial_obj, trial_best.copy(), trial_best
        else:
            w = 0.7 * w + 0.3 * best_w
            w = _enforce_turnover(w, w_prev, turnover_limit)
            w = _enforce_sector_bounds(w, groups, sector_lower, sector_upper, lower, upper)
            w = _enforce_factor_bounds(w, F, fl, fu, lower, upper)
            w = _enforce_sum_and_bounds(w, lower, upper)

    for _ in range(2):
        best_w = _enforce_sector_bounds(best_w, groups, sector_lower, sector_upper, lower, upper)
        best_w = _enforce_factor_bounds(best_w, F, fl, fu, lower, upper)
        best_w = _enforce_turnover(best_w, w_prev, turnover_limit)
        best_w = _enforce_sum_and_bounds(best_w, lower, upper)
        best_w = _enforce_turnover(best_w, w_prev, turnover_limit)

    if obj(w_prev) > best_obj:
        best_w = w_prev.copy()

    return {"weights": best_w}
# EVOLVE-BLOCK-END
