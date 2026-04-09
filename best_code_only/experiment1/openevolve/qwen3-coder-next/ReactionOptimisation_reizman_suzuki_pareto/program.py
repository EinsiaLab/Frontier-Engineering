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
    
    # Adaptive catalyst screening: try each catalyst with varied conditions
    screening_budget = min(len(task.CATEGORIES["catalyst"]), budget // 3)
    remaining_budget = budget - screening_budget
    
    # Phase 1: Adaptive catalyst screening with varied conditions
    screening_results = []
    # More comprehensive screening conditions
    cond_values = [
        (240.0, 80.0, 1.5),  # Low conditions
        (360.0, 100.0, 2.0), # Medium conditions
        (480.0, 120.0, 2.5), # High conditions (clamped)
        (300.0, 90.0, 1.75), # Balanced low
        (420.0, 110.0, 2.25), # Balanced high
    ]
    
    for i, catalyst in enumerate(task.CATEGORIES["catalyst"][:screening_budget]):
        # Use different conditions for different catalysts
        cond_idx = i % len(cond_values)
        t_res, temperature, catalyst_loading = cond_values[cond_idx]
        
        candidate = {
            "catalyst": catalyst,
            "t_res": t_res,
            "temperature": temperature,
            "catalyst_loading": catalyst_loading,
        }
        record = task.evaluate(experiment, candidate)
        screening_results.append(record)
        history.append(record)
    
    # Add some additional screening with random conditions for better coverage
    if screening_budget < len(task.CATEGORIES["catalyst"]):
        for i, catalyst in enumerate(task.CATEGORIES["catalyst"][screening_budget:]):
            candidate = {
                "catalyst": catalyst,
                "t_res": float(rng.uniform(*task.BOUNDS["t_res"])),
                "temperature": float(rng.uniform(*task.BOUNDS["temperature"])),
                "catalyst_loading": float(rng.uniform(*task.BOUNDS["catalyst_loading"])),
            }
            record = task.evaluate(experiment, candidate)
            screening_results.append(record)
            history.append(record)
    
    # Phase 2: Multi-weight catalyst ranking
    if screening_results:
        # Score catalysts with multiple weights to capture Pareto diversity
        all_scored = []
        for weight in [0.2, 0.5, 0.8]:
            for record in screening_results:
                score = task.scalarize(record, weight)
                all_scored.append((record["catalyst"], weight, score))
        
        # Aggregate scores per catalyst (average across weights)
        catalyst_scores = {}
        for catalyst, weight, score in all_scored:
            if catalyst not in catalyst_scores:
                catalyst_scores[catalyst] = []
            catalyst_scores[catalyst].append(score)
        
        avg_scores = [(c, np.mean(s)) for c, s in catalyst_scores.items()]
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        # Select top 3 catalysts with more diversity in performance
        # Weight selection based on all scalarization weights
        catalyst_scores_by_weight = {c: [] for c in task.CATEGORIES["catalyst"]}
        for record in screening_results:
            for weight in [0.2, 0.5, 0.8]:
                score = task.scalarize(record, weight)
                catalyst_scores_by_weight[record["catalyst"]].append(score)
        
        # Calculate performance metrics for each catalyst
        catalyst_metrics = {}
        for catalyst, scores in catalyst_scores_by_weight.items():
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                # Prefer catalysts with high mean and moderate std (reliable performers)
                catalyst_metrics[catalyst] = (mean_score, std_score, mean_score - 0.1 * std_score)
        
        # Sort by robust score (mean - 0.1*std) to balance performance and reliability
        sorted_catalysts = sorted(catalyst_metrics.items(), key=lambda x: x[1][2], reverse=True)
        
        # Select top 3 diverse catalysts
        top_catalysts = []
        for catalyst, metrics in sorted_catalysts:
            if len(top_catalysts) < 3:
                # Check for diversity in performance space
                is_diverse = True
                for selected in top_catalysts:
                    # Check if this catalyst is too similar to selected ones
                    selected_metrics = catalyst_metrics[selected]
                    if abs(metrics[0] - selected_metrics[0]) < 0.1 * max(abs(metrics[0]), abs(selected_metrics[0])):
                        is_diverse = False
                        break
                if is_diverse:
                    top_catalysts.append(catalyst)
            else:
                break
        
        # If we couldn't get 3 diverse catalysts, just take top ones by robust score
        if len(top_catalysts) < 3:
            top_catalysts = [c[0] for c in sorted_catalysts[:3]]
    
    # Try Bayesian optimization for continuous variables
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        from scipy.optimize import minimize
        
        # Use Bayesian optimization with scalarized objective
        def scalarized_objective(params, weight=0.5):
            candidate = {
                "catalyst": top_catalysts[0],  # Focus on best catalyst for BO
                "t_res": params[0],
                "temperature": params[1],
                "catalyst_loading": params[2],
            }
            record = task.evaluate(experiment, candidate)
            history.append(record)
            score = task.scalarize(record, weight)
            return -score  # Minimize negative score
        
        # Prepare bounds for optimization
        bounds = [
            task.BOUNDS["t_res"],
            task.BOUNDS["temperature"],
            task.BOUNDS["catalyst_loading"]
        ]
        
        # Run BO for remaining budget
        bo_budget = min(remaining_budget, 12)  # Limit BO calls
        remaining_random = remaining_budget - bo_budget
        
        # Initial points from Sobol sampling
        if remaining_random > 0:
            try:
                from scipy.stats import qmc
                dim = 3
                sampler = qmc.Sobol(d=dim, seed=seed+100)
                sobol_samples = sampler.random(n=remaining_random)
            except (ImportError, AttributeError):
                sobol_samples = None
        
        # Implement better BO strategy with adaptive exploration
        for i in range(bo_budget):
            # Adaptive catalyst selection based on iteration and diversity needs
            if i < 5 and len(top_catalysts) > 1:
                # Use different top catalysts in early iterations for diversity
                current_catalyst = top_catalysts[i % len(top_catalysts)]
            else:
                # Use best catalyst with probability based on performance spread
                if screening_results:
                    # Calculate performance spread for top catalysts
                    catalyst_scores = []
                    for c in top_catalysts:
                        scores = [task.scalarize(r, 0.5) for r in screening_results if r["catalyst"] == c]
                        if scores:
                            catalyst_scores.append((c, np.mean(scores)))
                    if catalyst_scores:
                        best_score = max(s for _, s in catalyst_scores)
                        # Use best catalyst with probability based on score gap
                        prob_best = min(0.9, 0.7 + 0.1 * (best_score - min(s for _, s in catalyst_scores)))
                        current_catalyst = top_catalysts[0] if rng.random() < prob_best else top_catalysts[1]
                    else:
                        current_catalyst = top_catalysts[0]
                else:
                    current_catalyst = top_catalysts[0]
            
            # Use different weights for exploration with adaptive scheduling
            if i < 3:
                weight = [0.2, 0.5, 0.8][i]
            else:
                # Gradually shift toward higher yield focus as budget progresses
                weight = min(0.8, 0.2 + 0.2 * (i // 3))
            
            # Initialize with good points from screening, but more intelligently
            if screening_results:
                # Find best points from screening for this catalyst
                catalyst_screening = [r for r in screening_results if r["catalyst"] == current_catalyst]
                if catalyst_screening:
                    best_screening = min(catalyst_screening, 
                                        key=lambda x: -task.scalarize(x, weight))
                    x0 = [best_screening["t_res"], 
                         best_screening["temperature"], 
                         best_screening["catalyst_loading"]]
                else:
                    # Use best from overall screening
                    best_screening = min(screening_results, 
                                        key=lambda x: -task.scalarize(x, weight))
                    x0 = [best_screening["t_res"], 
                         best_screening["temperature"], 
                         best_screening["catalyst_loading"]]
            else:
                x0 = [rng.uniform(*b) for b in bounds]
            
            # More sophisticated BO with multiple restarts and better initialization
            best_val = float('inf')
            best_x = x0
            
            # Try different starting points based on screening results and Sobol samples
            starting_points = [x0]
            if screening_results:
                # Sort screening results by current weight and try top points
                sorted_screening = sorted(screening_results, 
                                        key=lambda x: -task.scalarize(x, weight))
                for j in range(min(3, len(sorted_screening))):
                    point = sorted_screening[j]
                    starting_points.append([point["t_res"], point["temperature"], point["catalyst_loading"]])
            
            # Add Sobol-based starting points for better global exploration
            try:
                if remaining_random > 0 and sobol_samples is not None:
                    for j in range(min(2, remaining_random)):
                        t_res, temperature, catalyst_loading = sobol_samples[j]
                        t_res = task.BOUNDS["t_res"][0] + t_res * (task.BOUNDS["t_res"][1] - task.BOUNDS["t_res"][0])
                        temperature = task.BOUNDS["temperature"][0] + temperature * (task.BOUNDS["temperature"][1] - task.BOUNDS["temperature"][0])
                        catalyst_loading = task.BOUNDS["catalyst_loading"][0] + catalyst_loading * (task.BOUNDS["catalyst_loading"][1] - task.BOUNDS["catalyst_loading"][0])
                        starting_points.append([t_res, temperature, catalyst_loading])
            except:
                pass
            
            for start_x in starting_points:
                for _ in range(2):  # Multiple local optimizations
                    result = minimize(scalarized_objective, start_x, method='L-BFGS-B', 
                                    bounds=bounds, options={'maxiter': 15})
                    if result.fun < best_val:
                        best_val = result.fun
                        best_x = result.x
            
            # Add adaptive perturbation for exploration (more aggressive early, less late)
            perturbation_scale = max(0.03, 0.15 * (1 - i / bo_budget)) * (bounds[0][1] - bounds[0][0])
            if rng.random() < 0.4 + 0.1 * (1 - i / bo_budget):  # Higher early exploration probability
                best_x = [best_x[j] + rng.normal(0, perturbation_scale) 
                         for j in range(3)]
                best_x = [min(max(best_x[j], bounds[j][0]), bounds[j][1]) 
                         for j in range(3)]
            
            # Evaluate the best point
            candidate = {
                "catalyst": current_catalyst,
                "t_res": best_x[0],
                "temperature": best_x[1],
                "catalyst_loading": best_x[2],
            }
            record = task.evaluate(experiment, candidate)
            history.append(record)
        
        # Fill remaining budget with smarter random sampling
        if remaining_random > 0:
            # Use weighted selection for catalysts based on their performance
            if top_catalysts and screening_results:
                # Calculate selection weights based on screening performance
                catalyst_weights = []
                for catalyst in top_catalysts:
                    scores = [task.scalarize(r, 0.5) for r in screening_results if r["catalyst"] == catalyst]
                    if scores:
                        catalyst_weights.append((catalyst, np.mean(scores)))
                    else:
                        catalyst_weights.append((catalyst, 0.0))
                
                # Normalize weights for sampling
                total_weight = sum(w for _, w in catalyst_weights)
                if total_weight > 0:
                    probs = [w/total_weight for _, w in catalyst_weights]
                else:
                    probs = [1.0/len(catalyst_weights)] * len(catalyst_weights)
            else:
                probs = [1.0/len(top_catalysts)] * len(top_catalysts) if top_catalysts else None
            
            if sobol_samples is not None:
                for i in range(remaining_random):
                    t_res, temperature, catalyst_loading = sobol_samples[i]
                    t_res = task.BOUNDS["t_res"][0] + t_res * (task.BOUNDS["t_res"][1] - task.BOUNDS["t_res"][0])
                    temperature = task.BOUNDS["temperature"][0] + temperature * (task.BOUNDS["temperature"][1] - task.BOUNDS["temperature"][0])
                    catalyst_loading = task.BOUNDS["catalyst_loading"][0] + catalyst_loading * (task.BOUNDS["catalyst_loading"][1] - task.BOUNDS["catalyst_loading"][0])
                    
                    # Use weighted catalyst selection
                    if probs:
                        catalyst = rng.choice(top_catalysts, p=probs)
                    else:
                        catalyst = rng.choice(top_catalysts) if top_catalysts else rng.choice(task.CATEGORIES["catalyst"])
                    
                    candidate = {
                        "catalyst": catalyst,
                        "t_res": t_res,
                        "temperature": temperature,
                        "catalyst_loading": catalyst_loading,
                    }
                    record = task.evaluate(experiment, candidate)
                    history.append(record)
            else:
                for i in range(remaining_random):
                    t_res = rng.uniform(*task.BOUNDS["t_res"])
                    temperature = rng.uniform(*task.BOUNDS["temperature"])
                    catalyst_loading = rng.uniform(*task.BOUNDS["catalyst_loading"])
                    
                    # Use weighted catalyst selection
                    if probs:
                        catalyst = rng.choice(top_catalysts, p=probs)
                    else:
                        catalyst = rng.choice(top_catalysts) if top_catalysts else rng.choice(task.CATEGORIES["catalyst"])
                    
                    candidate = {
                        "catalyst": catalyst,
                        "t_res": t_res,
                        "temperature": temperature,
                        "catalyst_loading": catalyst_loading,
                    }
                    record = task.evaluate(experiment, candidate)
                    history.append(record)
    except Exception:
        # Fallback to original Sobol approach if BO fails
        try:
            from scipy.stats import qmc
            dim = 3
            sampler = qmc.Sobol(d=dim, seed=seed+100)
            sobol_samples = sampler.random(n=remaining_budget)
        except (ImportError, AttributeError):
            sobol_samples = None
        
        if sobol_samples is not None:
            for i in range(remaining_budget):
                t_res, temperature, catalyst_loading = sobol_samples[i]
                t_res = task.BOUNDS["t_res"][0] + t_res * (task.BOUNDS["t_res"][1] - task.BOUNDS["t_res"][0])
                temperature = task.BOUNDS["temperature"][0] + temperature * (task.BOUNDS["temperature"][1] - task.BOUNDS["temperature"][0])
                catalyst_loading = task.BOUNDS["catalyst_loading"][0] + catalyst_loading * (task.BOUNDS["catalyst_loading"][1] - task.BOUNDS["catalyst_loading"][0])
                
                candidate = {
                    "catalyst": rng.choice(top_catalysts),
                    "t_res": t_res,
                    "temperature": temperature,
                    "catalyst_loading": catalyst_loading,
                }
                record = task.evaluate(experiment, candidate)
                history.append(record)
        else:
            for i in range(remaining_budget):
                t_res = rng.uniform(*task.BOUNDS["t_res"])
                temperature = rng.uniform(*task.BOUNDS["temperature"])
                catalyst_loading = rng.uniform(*task.BOUNDS["catalyst_loading"])
                catalyst = rng.choice(top_catalysts)
                
                candidate = {
                    "catalyst": catalyst,
                    "t_res": t_res,
                    "temperature": temperature,
                    "catalyst_loading": catalyst_loading,
                }
                record = task.evaluate(experiment, candidate)
                history.append(record)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "adaptive_screened_sobol",
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
