"""Simple baseline for the Reizman Suzuki emulator task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _is_repo_root(path: Path) -> bool:
    return (path / "benchmarks").is_dir() and (path / "frontier_eval").is_dir()


def _ensure_domain_on_path() -> None:
    env_root = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])

    repo_root = next((path for path in candidates if _is_repo_root(path)), None)
    if repo_root is None:
        raise RuntimeError("Could not locate repository root for ReactionOptimisation.")

    domain_root = repo_root / "benchmarks" / "ReactionOptimisation"
    if not domain_root.is_dir():
        raise RuntimeError(f"ReactionOptimisation directory not found under {repo_root}.")

    domain_root_str = str(domain_root)
    if domain_root_str not in sys.path:
        sys.path.insert(0, domain_root_str)


_ensure_domain_on_path()

from reizman_suzuki_pareto import task
from shared.utils import dump_json, seed_everything


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    # Get task bounds
    catalysts = ["P1-L1", "P1-L2", "P1-L3", "P1-L4", "P1-L5", "P1-L6", "P1-L7"]
    cont_bounds = {
        "t_res": (60.0, 600.0),
        "temperature": (30.0, 110.0),
        "catalyst_loading": (0.498, 2.515),
    }
    X_min = np.array([cont_bounds["t_res"][0], cont_bounds["temperature"][0], cont_bounds["catalyst_loading"][0]])
    X_max = np.array([cont_bounds["t_res"][1], cont_bounds["temperature"][1], cont_bounds["catalyst_loading"][1]])

    def eval_candidate(candidate):
        record = task.evaluate(experiment, candidate)
        history.append(record)
        return record

    def make_candidate(cat, t_res, temperature, catalyst_loading):
        return {
            "catalyst": cat,
            "t_res": float(np.clip(t_res, cont_bounds["t_res"][0], cont_bounds["t_res"][1])),
            "temperature": float(np.clip(temperature, cont_bounds["temperature"][0], cont_bounds["temperature"][1])),
            "catalyst_loading": float(np.clip(catalyst_loading, cont_bounds["catalyst_loading"][0], cont_bounds["catalyst_loading"][1])),
        }

    def scalarize(record, w_yld=0.5):
        yld = record["yld"] / 100.0
        ton = 1.0 - record["ton"] / 200.0
        yld = float(np.clip(yld, 0, 1))
        ton = float(np.clip(ton, 0, 1))
        return w_yld * yld + (1.0 - w_yld) * ton

    # --- Stage 1: Screen all catalysts with diverse conditions ---
    screening_conditions = [
        (60.0, 110.0, 2.515),   # short, hot, high loading -> high yield
        (600.0, 30.0, 0.498),   # long, cold, low loading -> low ton
        (300.0, 70.0, 1.5),     # moderate
        (120.0, 100.0, 2.0),    # short-ish, hot
        (500.0, 50.0, 0.8),     # long, cool, low loading
    ]

    screening_budget = min(len(catalysts) * len(screening_conditions), budget // 3)
    catalyst_scores = {cat: [] for cat in catalysts}

    used = 0
    for cond in screening_conditions:
        for cat in catalysts:
            if used >= screening_budget:
                break
            candidate = make_candidate(cat, cond[0], cond[1], cond[2])
            record = eval_candidate(candidate)
            catalyst_scores[cat].append(record)
            used += 1
        if used >= screening_budget:
            break

    # Rank catalysts by contribution across multiple scalarization weights
    weights_for_ranking = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
    catalyst_rank = {}
    for cat in catalysts:
        if not catalyst_scores[cat]:
            catalyst_rank[cat] = -1.0
            continue
        total = sum(
            max(scalarize(r, w) for r in catalyst_scores[cat])
            for w in weights_for_ranking
        )
        catalyst_rank[cat] = total

    sorted_cats = sorted(catalyst_rank.keys(), key=lambda c: catalyst_rank[c], reverse=True)
    n_top = min(5, max(3, len([c for c in sorted_cats if catalyst_rank[c] > 0])))
    top_cats = sorted_cats[:n_top]

    # --- Stage 2: GP-based Bayesian optimization per catalyst/weight combo ---
    remaining_budget = budget - len(history)

    # Define scalarization weights to cover Pareto front well
    pareto_weights = [0.03, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97]

    # Create subproblems: (catalyst, weight)
    subproblems = []
    for cat in top_cats[:3]:
        for w in pareto_weights:
            subproblems.append((cat, w))
    for cat in top_cats[3:]:
        for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
            subproblems.append((cat, w))

    budget_per_sub = max(3, remaining_budget * 2 // (3 * max(len(subproblems), 1)))

    def gp_optimize(cat, w, n_iters):
        """GP-based Bayesian optimization with Expected Improvement."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            from scipy.optimize import minimize as sp_minimize
            from scipy.stats import norm as sp_norm
        except ImportError:
            local_search(cat, w, n_iters)
            return

        for iteration in range(n_iters):
            if len(history) >= budget:
                break

            cat_records = [r for r in history if r["catalyst"] == cat]

            if len(cat_records) < 3:
                # Random exploration
                t = rng.uniform(*cont_bounds["t_res"])
                temp = rng.uniform(*cont_bounds["temperature"])
                cl = rng.uniform(*cont_bounds["catalyst_loading"])
                candidate = make_candidate(cat, t, temp, cl)
                eval_candidate(candidate)
                continue

            X = np.array([[r["t_res"], r["temperature"], r["catalyst_loading"]] for r in cat_records])
            y = np.array([scalarize(r, w) for r in cat_records])

            X_norm = (X - X_min) / (X_max - X_min + 1e-10)

            kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=3, normalize_y=True)
            try:
                gp.fit(X_norm, y)
            except:
                local_search(cat, w, max(1, n_iters - iteration))
                return

            best_score = y.max()

            def neg_ei(x_norm):
                x_norm = x_norm.reshape(1, -1)
                mu, sigma = gp.predict(x_norm, return_std=True)
                sigma = max(sigma[0], 1e-8)
                z = (mu[0] - best_score - 0.005) / sigma
                ei = sigma * (z * sp_norm.cdf(z) + sp_norm.pdf(z))
                return -ei

            best_ei_x = None
            best_ei_val = float('inf')
            for _ in range(15):
                x0 = rng.uniform(0, 1, size=3)
                try:
                    res = sp_minimize(neg_ei, x0, bounds=[(0, 1)] * 3, method='L-BFGS-B')
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_ei_x = res.x
                except:
                    pass

            if best_ei_x is None:
                best_ei_x = rng.uniform(0, 1, size=3)

            x_orig = best_ei_x * (X_max - X_min) + X_min
            candidate = make_candidate(cat, x_orig[0], x_orig[1], x_orig[2])
            eval_candidate(candidate)

    def local_search(cat, w, n_iters):
        cat_records = [r for r in history if r["catalyst"] == cat]
        if cat_records:
            best_rec = max(cat_records, key=lambda r: scalarize(r, w))
            best_t = best_rec["t_res"]
            best_temp = best_rec["temperature"]
            best_cl = best_rec["catalyst_loading"]
            best_score = scalarize(best_rec, w)
        else:
            best_t, best_temp, best_cl = 300.0, 70.0, 1.5
            best_score = -1.0

        scale = 0.25
        no_improve = 0
        for _ in range(n_iters):
            if len(history) >= budget:
                break
            if no_improve > 3:
                scale *= 0.6
                no_improve = 0
            new_t = best_t + rng.normal(0, (X_max[0]-X_min[0]) * scale)
            new_temp = best_temp + rng.normal(0, (X_max[1]-X_min[1]) * scale)
            new_cl = best_cl + rng.normal(0, (X_max[2]-X_min[2]) * scale)
            candidate = make_candidate(cat, new_t, new_temp, new_cl)
            record = eval_candidate(candidate)
            s = scalarize(record, w)
            if s > best_score:
                best_score = s
                best_t = candidate["t_res"]
                best_temp = candidate["temperature"]
                best_cl = candidate["catalyst_loading"]
                no_improve = 0
            else:
                no_improve += 1

    for cat, w in subproblems:
        if len(history) >= budget:
            break
        gp_optimize(cat, w, budget_per_sub)

    # --- Stage 3: Targeted Pareto refinement ---
    remaining = budget - len(history)
    if remaining > 0:
        fine_weights = np.linspace(0.02, 0.98, 20)
        per_weight = max(1, remaining // (len(fine_weights) * min(3, len(top_cats))))

        for w in fine_weights:
            for cat in top_cats[:3]:
                if len(history) >= budget:
                    break
                cat_records = [r for r in history if r["catalyst"] == cat]
                if cat_records:
                    best_rec = max(cat_records, key=lambda r: scalarize(r, w))
                    bt = best_rec["t_res"]
                    btemp = best_rec["temperature"]
                    bcl = best_rec["catalyst_loading"]
                    for _ in range(per_weight):
                        if len(history) >= budget:
                            break
                        t = bt + rng.normal(0, 20)
                        temp = btemp + rng.normal(0, 4)
                        cl = bcl + rng.normal(0, 0.1)
                        candidate = make_candidate(cat, t, temp, cl)
                        eval_candidate(candidate)

    # Fill remaining budget
    while len(history) < budget:
        cat = rng.choice(top_cats[:3])
        w = rng.uniform(0.02, 0.98)
        cat_records = [r for r in history if r["catalyst"] == cat]
        if cat_records:
            best_rec = max(cat_records, key=lambda r: scalarize(r, w))
            t = best_rec["t_res"] + rng.normal(0, 15)
            temp = best_rec["temperature"] + rng.normal(0, 3)
            cl = best_rec["catalyst_loading"] + rng.normal(0, 0.08)
        else:
            t = rng.uniform(*cont_bounds["t_res"])
            temp = rng.uniform(*cont_bounds["temperature"])
            cl = rng.uniform(*cont_bounds["catalyst_loading"])
        candidate = make_candidate(cat, t, temp, cl)
        eval_candidate(candidate)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "gp_bayesian_pareto_search_v2",
        "seed": seed,
        "budget": budget,
        "history": history,
        "summary": task.summarize(history),
    }
# EVOLVE-BLOCK-END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=task.DEFAULT_BUDGET)
    args = parser.parse_args()
    print(dump_json(solve(seed=args.seed, budget=args.budget)))


if __name__ == "__main__":
    main()
