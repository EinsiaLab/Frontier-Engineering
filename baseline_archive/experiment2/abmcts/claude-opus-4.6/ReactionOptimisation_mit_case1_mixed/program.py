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
    best_candidate = None
    best_y = -float('inf')

    # Get domain info
    cont_names = [name for name in task.INPUT_NAMES if name != 'cat_index']
    bounds = {}
    for name in cont_names:
        bounds[name] = task.BOUNDS[name]
    
    n_categories = len(task.CATEGORIES)
    
    def eval_candidate(candidate):
        nonlocal best_y, best_candidate
        record = task.evaluate(experiment, candidate)
        history.append(record)
        if record['y'] > best_y:
            best_y = record['y']
            best_candidate = {name: record[name] for name in task.INPUT_NAMES}
        return record
    
    # Track per-category performance
    cat_observations = {c: [] for c in range(n_categories)}
    
    # Phase 1: Latin hypercube-like initial exploration for each category
    initial_per_cat = min(4, max(2, budget // (n_categories * 3)))
    
    for cat_idx in range(n_categories):
        for i in range(initial_per_cat):
            candidate = {}
            candidate['cat_index'] = cat_idx
            for name in cont_names:
                lo, hi = bounds[name]
                # Stratified sampling within category
                segment_lo = lo + (hi - lo) * i / initial_per_cat
                segment_hi = lo + (hi - lo) * (i + 1) / initial_per_cat
                candidate[name] = float(rng.uniform(segment_lo, segment_hi))
            record = eval_candidate(candidate)
            cat_observations[cat_idx].append((record['y'], {name: record[name] for name in task.INPUT_NAMES}))
    
    remaining = budget - len(history)
    
    if remaining <= 0:
        return {
            "task_name": task.TASK_NAME,
            "algorithm_name": "bayesian_category_search",
            "seed": seed,
            "budget": budget,
            "history": history,
            "summary": task.summarize(history),
        }
    
    # Try to use a simple GP-based approach per category
    # First, determine category ranking
    cat_best_y = {}
    cat_best_candidate = {}
    for c in range(n_categories):
        if cat_observations[c]:
            best_obs = max(cat_observations[c], key=lambda x: x[0])
            cat_best_y[c] = best_obs[0]
            cat_best_candidate[c] = best_obs[1]
        else:
            cat_best_y[c] = -float('inf')
    
    sorted_cats = sorted(range(n_categories), key=lambda c: cat_best_y[c], reverse=True)
    
    # Allocate budget with emphasis on top categories
    # Top cat: 55%, second: 25%, rest: split remainder
    allocations = {}
    if n_categories >= 3:
        allocations[sorted_cats[0]] = int(remaining * 0.55)
        allocations[sorted_cats[1]] = int(remaining * 0.28)
        leftover = remaining - allocations[sorted_cats[0]] - allocations[sorted_cats[1]]
        for i in range(2, n_categories):
            share = max(1, leftover // (n_categories - 2))
            allocations[sorted_cats[i]] = share
            leftover -= share
        allocations[sorted_cats[0]] += max(0, leftover)
    elif n_categories == 2:
        allocations[sorted_cats[0]] = int(remaining * 0.65)
        allocations[sorted_cats[1]] = remaining - allocations[sorted_cats[0]]
    else:
        allocations[sorted_cats[0]] = remaining
    
    # Phase 2: For each category, use GP-based optimization if sklearn available, else adaptive local search
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
        use_gp = True
    except ImportError:
        use_gp = False
    
    for cat_idx in sorted_cats:
        cat_budget = allocations.get(cat_idx, 0)
        if cat_budget <= 0 or len(history) >= budget:
            continue
        
        # Collect all observations for this category
        all_X = []
        all_Y = []
        for rec in history:
            if rec['cat_index'] == cat_idx:
                x = [rec[name] for name in cont_names]
                all_X.append(x)
                all_Y.append(rec['y'])
        
        local_best_y = cat_best_y[cat_idx]
        local_best = cat_best_candidate.get(cat_idx, best_candidate).copy() if cat_best_candidate.get(cat_idx) else best_candidate.copy()
        
        for step in range(cat_budget):
            if len(history) >= budget:
                break
            
            candidate = {'cat_index': cat_idx}
            
            if use_gp and len(all_X) >= 4:
                # Fit GP and use EI to suggest next point
                try:
                    X_arr = np.array(all_X)
                    Y_arr = np.array(all_Y)
                    
                    # Normalize inputs
                    lo_arr = np.array([bounds[name][0] for name in cont_names])
                    hi_arr = np.array([bounds[name][1] for name in cont_names])
                    X_norm = (X_arr - lo_arr) / (hi_arr - lo_arr + 1e-10)
                    
                    kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=0.3) + WhiteKernel(noise_level=0.01)
                    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6, normalize_y=True)
                    gp.fit(X_norm, Y_arr)
                    
                    # Generate candidate points and pick best EI
                    n_candidates = 2000
                    X_cand = rng.uniform(0, 1, size=(n_candidates, len(cont_names)))
                    
                    # Also add perturbations around current best
                    n_local = 500
                    progress = step / max(1, cat_budget - 1)
                    scale = 0.15 * (1.0 - 0.6 * progress)
                    best_norm = (np.array([local_best[name] for name in cont_names]) - lo_arr) / (hi_arr - lo_arr + 1e-10)
                    X_local = best_norm + rng.normal(0, scale, size=(n_local, len(cont_names)))
                    X_local = np.clip(X_local, 0, 1)
                    X_cand = np.vstack([X_cand, X_local])
                    
                    mu, sigma = gp.predict(X_cand, return_std=True)
                    sigma = np.maximum(sigma, 1e-8)
                    
                    # Expected Improvement
                    from scipy.stats import norm
                    best_so_far = np.max(Y_arr)
                    z = (mu - best_so_far - 0.01) / sigma
                    ei = (mu - best_so_far - 0.01) * norm.cdf(z) + sigma * norm.pdf(z)
                    
                    best_idx = np.argmax(ei)
                    x_next = X_cand[best_idx] * (hi_arr - lo_arr) + lo_arr
                    
                    for j, name in enumerate(cont_names):
                        lo, hi = bounds[name]
                        candidate[name] = float(np.clip(x_next[j], lo, hi))
                    
                except Exception:
                    # Fallback to local search
                    progress = step / max(1, cat_budget - 1)
                    scale = 0.25 * (1.0 - 0.7 * progress)
                    for name in cont_names:
                        lo, hi = bounds[name]
                        span = hi - lo
                        noise = rng.normal(0, scale * span)
                        candidate[name] = float(np.clip(local_best[name] + noise, lo, hi))
            else:
                # Local search fallback
                progress = step / max(1, cat_budget - 1)
                scale = 0.3 * (1.0 - 0.7 * progress)
                n_mutate = rng.integers(1, len(cont_names) + 1)
                vars_to_mutate = rng.choice(cont_names, size=n_mutate, replace=False)
                for name in cont_names:
                    if name in vars_to_mutate:
                        lo, hi = bounds[name]
                        span = hi - lo
                        noise = rng.normal(0, scale * span)
                        candidate[name] = float(np.clip(local_best[name] + noise, lo, hi))
                    else:
                        candidate[name] = local_best[name]
            
            record = eval_candidate(candidate)
            all_X.append([record[name] for name in cont_names])
            all_Y.append(record['y'])
            
            if record['y'] > local_best_y:
                local_best_y = record['y']
                local_best = {name: record[name] for name in task.INPUT_NAMES}
    
    # Phase 3: Final refinement around global best with remaining budget
    while len(history) < budget:
        candidate = best_candidate.copy()
        scale = 0.05
        for name in cont_names:
            lo, hi = bounds[name]
            span = hi - lo
            noise = rng.normal(0, scale * span)
            candidate[name] = float(np.clip(candidate[name] + noise, lo, hi))
        eval_candidate(candidate)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "bayesian_category_search",
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
