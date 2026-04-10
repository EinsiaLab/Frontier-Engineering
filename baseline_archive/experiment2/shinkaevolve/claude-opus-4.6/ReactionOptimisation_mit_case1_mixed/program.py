"""Simple baseline for the MIT case 1 task."""

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

from mit_case1_mixed import task
from shared.utils import dump_json, seed_everything


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    from scipy.stats import norm as sp_norm

    # Extract bounds from task
    cont_bounds = {
        "conc_cat": (0.000835, 0.004175),
        "t": (60.0, 600.0),
        "temperature": (30.0, 110.0),
    }
    cont_names = list(cont_bounds.keys())
    n_cont = len(cont_names)
    n_cats = 8

    def make_candidate(cat_idx, cont_values):
        cand = {"cat_index": int(cat_idx)}
        for name in cont_names:
            lo, hi = cont_bounds[name]
            cand[name] = float(np.clip(cont_values[name], lo, hi))
        return cand

    def cont_to_norm(record):
        return np.array([(record[name] - cont_bounds[name][0]) / (cont_bounds[name][1] - cont_bounds[name][0]) for name in cont_names])

    def norm_to_cont(nvec):
        result = {}
        for i, name in enumerate(cont_names):
            lo, hi = cont_bounds[name]
            result[name] = float(np.clip(lo + nvec[i] * (hi - lo), lo, hi))
        return result

    def evaluate_and_record(cat_idx, cont_values):
        candidate = make_candidate(cat_idx, cont_values)
        record = task.evaluate(experiment, candidate)
        history.append(record)
        return record

    # Track observations per category
    cat_records = {i: [] for i in range(n_cats)}

    def record_obs(record):
        ci = int(record["cat_index"])
        cat_records[ci].append(record)

    # ---- Matern 5/2 kernel for better GP ----
    def matern52_kernel(X1, X2, length_scale, signal_var):
        """Matern 5/2 kernel."""
        diff = X1[:, None, :] - X2[None, :, :]
        r = np.sqrt(np.sum(diff ** 2 / (length_scale ** 2), axis=2) + 1e-12)
        sqrt5r = np.sqrt(5.0) * r
        return signal_var * (1.0 + sqrt5r + 5.0 / 3.0 * r ** 2) * np.exp(-sqrt5r)

    def gp_predict(X_train, y_train, X_test, length_scale, signal_var, noise_var):
        """GP prediction with Matern 5/2 kernel."""
        n = X_train.shape[0]
        K = matern52_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(n)
        K_s = matern52_kernel(X_train, X_test, length_scale, signal_var)
        K_ss_diag = signal_var * np.ones(X_test.shape[0])  # diagonal only

        try:
            L = np.linalg.cholesky(K + 1e-7 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            mu = K_s.T @ alpha
            v = np.linalg.solve(L, K_s)
            var = K_ss_diag - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)
        except np.linalg.LinAlgError:
            mu = np.full(X_test.shape[0], np.mean(y_train))
            var = np.full(X_test.shape[0], np.var(y_train) + 0.01)

        return mu, var

    def gp_log_marginal_likelihood(X_train, y_train, length_scale, signal_var, noise_var):
        """Compute log marginal likelihood for hyperparameter selection."""
        n = X_train.shape[0]
        K = matern52_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(n)
        try:
            L = np.linalg.cholesky(K + 1e-7 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            lml = -0.5 * y_train @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return lml
        except np.linalg.LinAlgError:
            return -1e10

    def expected_improvement(mu, var, best_y, xi=0.01):
        """Compute Expected Improvement."""
        sigma = np.sqrt(var)
        z = (mu - best_y - xi) / (sigma + 1e-10)
        ei = (mu - best_y - xi) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z)
        return ei

    # ---- Phase 1: Initial exploration ----
    # Explore 5 categories initially with diverse continuous settings
    # Use a space-filling design
    n_phase1 = min(5, budget)

    # Generate good initial continuous points using LHS
    perms = {name: rng.permutation(n_phase1) for name in cont_names}
    lhs_samples = []
    for i in range(n_phase1):
        cont_vals = {}
        for name in cont_names:
            lo, hi = cont_bounds[name]
            stratum_idx = perms[name][i]
            f = (stratum_idx + rng.uniform(0.15, 0.85)) / n_phase1
            cont_vals[name] = lo + f * (hi - lo)
        lhs_samples.append(cont_vals)

    # Choose which 5 categories to try initially (random subset)
    initial_cats = rng.choice(n_cats, size=n_phase1, replace=False)

    for i in range(n_phase1):
        record = evaluate_and_record(initial_cats[i], lhs_samples[i])
        record_obs(record)

    used = n_phase1
    remaining = budget - used

    if remaining <= 0:
        return {
            "task_name": task.TASK_NAME,
            "algorithm_name": "gp_bo_mixed_v2",
            "seed": seed,
            "budget": budget,
            "history": history,
            "summary": task.summarize(history),
        }

    # ---- Phase 2: Explore remaining categories with promising settings ----
    # Try remaining 3 categories using the best continuous setting found so far
    unexplored_cats = [c for c in range(n_cats) if not cat_records[c]]
    n_phase2 = min(len(unexplored_cats), remaining)

    if n_phase2 > 0:
        # Use the best continuous setting found so far
        best_rec = max(history, key=lambda r: r["y"])
        best_cont = {name: best_rec[name] for name in cont_names}

        # Add small perturbations to avoid exact duplicates
        phase2_cats = unexplored_cats[:n_phase2]
        for ci in phase2_cats:
            perturbed_cont = {}
            for name in cont_names:
                lo, hi = cont_bounds[name]
                val = best_cont[name] + rng.standard_normal() * 0.05 * (hi - lo)
                perturbed_cont[name] = float(np.clip(val, lo, hi))
            record = evaluate_and_record(ci, perturbed_cont)
            record_obs(record)

        used += n_phase2
        remaining = budget - used

    if remaining <= 0:
        return {
            "task_name": task.TASK_NAME,
            "algorithm_name": "gp_bo_mixed_v2",
            "seed": seed,
            "budget": budget,
            "history": history,
            "summary": task.summarize(history),
        }

    # ---- Phase 3: GP-based Bayesian Optimization ----
    for iteration in range(remaining):
        best_y_global = max(r["y"] for r in history)

        best_cat = -1
        best_point = None
        best_acq = -1e10

        # Collect all data
        all_X = []
        all_y = []
        all_cat = []
        for ci in range(n_cats):
            for r in cat_records[ci]:
                all_X.append(cont_to_norm(r))
                all_y.append(r["y"])
                all_cat.append(ci)

        all_X = np.array(all_X)
        all_y = np.array(all_y)
        all_cat = np.array(all_cat)
        n_total = len(all_y)

        # Build feature space with one-hot encoding
        def build_features(cat_idx, cont_norm):
            one_hot = np.zeros(n_cats)
            one_hot[cat_idx] = 1.0
            return np.concatenate([one_hot * 0.5, cont_norm])

        X_train = np.array([build_features(all_cat[i], all_X[i]) for i in range(n_total)])
        y_train = all_y.copy()

        # Normalize y
        y_mean = np.mean(y_train)
        y_std = max(np.std(y_train), 1e-6)
        y_norm = (y_train - y_mean) / y_std
        best_y_norm = (best_y_global - y_mean) / y_std

        # Tune GP hyperparameters via grid search on log marginal likelihood
        best_lml = -1e10
        best_ls = 0.5
        best_sv = 1.0
        best_nv = 0.01
        d = n_cats + n_cont

        for ls_val in [0.3, 0.5, 0.8, 1.2]:
            for sv_val in [0.5, 1.0, 2.0]:
                for nv_val in [0.005, 0.01, 0.05]:
                    ls = np.ones(d) * ls_val
                    lml = gp_log_marginal_likelihood(X_train, y_norm, ls, sv_val, nv_val)
                    if lml > best_lml:
                        best_lml = lml
                        best_ls = ls_val
                        best_sv = sv_val
                        best_nv = nv_val

        length_scale = np.ones(d) * best_ls
        signal_var = best_sv
        noise_var = best_nv

        # Adaptive exploration parameter
        progress = iteration / max(remaining - 1, 1)
        xi = 0.02 * (1.0 - 0.7 * progress)

        # Rank categories by best observed score
        cat_best_scores = []
        for ci in range(n_cats):
            if cat_records[ci]:
                cat_best_scores.append((ci, max(r["y"] for r in cat_records[ci])))
            else:
                cat_best_scores.append((ci, -1.0))
        cat_best_scores.sort(key=lambda x: x[1], reverse=True)

        # Focus on top categories but occasionally explore others
        # More exploitation as we progress
        if progress < 0.3:
            n_consider = min(6, n_cats)
        elif progress < 0.7:
            n_consider = min(4, n_cats)
        else:
            n_consider = min(3, n_cats)

        cats_to_try = [cs[0] for cs in cat_best_scores[:n_consider]]

        # For each category, generate candidate points and evaluate acquisition
        n_candidates_per_cat = 500
        for ci in cats_to_try:
            candidates_norm = []

            # Random candidates (broader exploration)
            n_random = n_candidates_per_cat // 5
            for _ in range(n_random):
                candidates_norm.append(rng.uniform(0, 1, n_cont))

            # Local perturbations around best in this category (multi-scale)
            if cat_records[ci]:
                cat_best_rec = max(cat_records[ci], key=lambda r: r["y"])
                base = cont_to_norm(cat_best_rec)

                # Also consider global best
                global_best_rec = max(history, key=lambda r: r["y"])
                global_base = cont_to_norm(global_best_rec)

                n_local = n_candidates_per_cat - n_random
                for j in range(n_local):
                    # Alternate between cat best, global best, and random restarts
                    if j % 5 == 0:
                        ref = global_base
                    elif j % 5 == 4:
                        ref = rng.uniform(0, 1, n_cont)
                    else:
                        ref = base

                    # Multi-scale perturbation
                    scale = rng.choice([0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25])
                    perturbed = ref + rng.standard_normal(n_cont) * scale
                    candidates_norm.append(np.clip(perturbed, 0, 1))
            else:
                for _ in range(n_candidates_per_cat - n_random):
                    candidates_norm.append(rng.uniform(0, 1, n_cont))

            candidates_norm = np.array(candidates_norm)

            # Build test features
            X_test = np.array([build_features(ci, c) for c in candidates_norm])

            # GP predict
            mu, var = gp_predict(X_train, y_norm, X_test, length_scale, signal_var, noise_var)

            # Expected Improvement
            ei = expected_improvement(mu, var, best_y_norm, xi=xi)

            # Find best candidate for this category
            best_idx = np.argmax(ei)
            if ei[best_idx] > best_acq:
                best_acq = ei[best_idx]
                best_cat = ci
                best_point = candidates_norm[best_idx]

        # Refine the best point found with fine local search
        if best_point is not None and best_cat >= 0:
            n_refine = 300
            refine_candidates = []
            for _ in range(n_refine):
                scale = rng.choice([0.005, 0.01, 0.02, 0.03])
                perturbed = best_point + rng.standard_normal(n_cont) * scale
                refine_candidates.append(np.clip(perturbed, 0, 1))
            refine_candidates = np.array(refine_candidates)

            X_refine = np.array([build_features(best_cat, c) for c in refine_candidates])
            mu_r, var_r = gp_predict(X_train, y_norm, X_refine, length_scale, signal_var, noise_var)
            ei_r = expected_improvement(mu_r, var_r, best_y_norm, xi=xi)

            best_refine_idx = np.argmax(ei_r)
            if ei_r[best_refine_idx] > best_acq:
                best_point = refine_candidates[best_refine_idx]

        # Fallback: perturb global best with UCB-style exploration
        if best_acq < 1e-12 or best_point is None:
            global_best_rec = max(history, key=lambda r: r["y"])
            best_cat = int(global_best_rec["cat_index"])
            base = cont_to_norm(global_best_rec)
            # Shrinking perturbation
            scale = 0.08 * (1.0 - 0.5 * progress)
            best_point = np.clip(base + rng.standard_normal(n_cont) * scale, 0, 1)

        # Evaluate the chosen point
        new_cont = norm_to_cont(best_point)
        record = evaluate_and_record(best_cat, new_cont)
        record_obs(record)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "gp_bo_mixed_v2",
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