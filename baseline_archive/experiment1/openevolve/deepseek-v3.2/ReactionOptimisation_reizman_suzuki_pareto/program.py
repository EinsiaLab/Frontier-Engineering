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
from shared.utils import dump_json, seed_everything, clamp


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    # Stage 1: Screen all catalysts
    screening_history = []
    for catalyst in task.CATEGORIES["catalyst"]:
        candidate = {
            "catalyst": catalyst,
            "t_res": 360.0,
            "temperature": 100.0,
            "catalyst_loading": 2.0,
        }
        record = task.evaluate(experiment, candidate)
        screening_history.append(record)
    history.extend(screening_history)

    # Stage 2: Select best catalyst for each weight and perform improved local search
    # Use fewer weights like reference to allocate more budget per weight
    weights = [0.2, 0.5, 0.8]
    remaining_budget = budget - len(screening_history)
    if remaining_budget <= 0:
        # If budget is too small, just return screening results
        return {
            "task_name": task.TASK_NAME,
            "algorithm_name": "screen_and_local_search",
            "seed": seed,
            "budget": budget,
            "history": history[:budget],
            "summary": task.summarize(history[:budget]),
        }

    # For each weight, pick the best catalyst from screening
    best_per_weight = []
    for weight in weights:
        best_record = max(screening_history, key=lambda row: task.scalarize(row, weight))
        best_per_weight.append((weight, best_record))

    # Distribute remaining budget evenly across weights
    per_weight_budget = remaining_budget // len(weights)
    extra = remaining_budget % len(weights)
    budgets = [per_weight_budget + (1 if i < extra else 0) for i in range(len(weights))]

    # Improved local search for each selected catalyst with adaptive exploration
    for (weight, seed_record), sub_budget in zip(best_per_weight, budgets):
        catalyst = seed_record["catalyst"]
        # Start from screening point
        current_candidate = {
            "catalyst": catalyst,
            "t_res": seed_record["t_res"],
            "temperature": seed_record["temperature"],
            "catalyst_loading": seed_record["catalyst_loading"],
        }
        current_score = task.scalarize(seed_record, weight)
        
        for iteration in range(sub_budget):
            # Adaptive number of proposals: start with more, reduce over time
            # At least 2 proposals, maximum 10, decreasing linearly
            num_proposals = max(2, min(10, 10 - iteration // (sub_budget // 5)))
            best_proposal_record = None
            best_proposal_score = -float('inf')
            best_proposal_candidate = None
            
            for _ in range(num_proposals):
                proposal = task.mutate_candidate(current_candidate, rng)
                record = task.evaluate(experiment, proposal)
                score = task.scalarize(record, weight)
                
                if score > best_proposal_score:
                    best_proposal_score = score
                    best_proposal_record = record
                    best_proposal_candidate = proposal
            
            history.append(best_proposal_record)
            
            # Accept if better than current, or occasionally accept worse to explore
            if best_proposal_score > current_score:
                current_candidate = best_proposal_candidate
                current_score = best_proposal_score
            else:
                # Small chance to accept worse solution to maintain exploration
                # Probability decays over iterations
                accept_probability = 0.1 * (1.0 - iteration / sub_budget)
                if rng.random() < accept_probability:
                    current_candidate = best_proposal_candidate
                    current_score = best_proposal_score

    # Ensure we use exactly budget (should be exact, but just in case)
    if len(history) < budget:
        remaining = budget - len(history)
        # Distribute remaining budget across weights to improve Pareto frontier
        # Calculate current best scores for each weight
        best_scores = {}
        for weight in weights:
            best_record = max(history, key=lambda row: task.scalarize(row, weight))
            best_scores[weight] = task.scalarize(best_record, weight)
        
        # Sort weights by their best scores (lower scores need more improvement)
        sorted_weights = sorted(weights, key=lambda w: best_scores[w])
        
        # Distribute remaining budget proportionally to weights with lower scores
        weight_budgets = {}
        total_remaining = remaining
        # Assign more budget to weights with lower scores
        for i, weight in enumerate(sorted_weights):
            if i == 0:
                weight_budgets[weight] = total_remaining // 2
            elif i == 1:
                weight_budgets[weight] = total_remaining // 3
            else:
                weight_budgets[weight] = total_remaining - (total_remaining // 2 + total_remaining // 3)
        
        # For each weight, find the best candidate and continue optimization
        for weight, sub_budget in weight_budgets.items():
            if sub_budget <= 0:
                continue
            # Find best candidate for this weight
            best_record = max(history, key=lambda row: task.scalarize(row, weight))
            catalyst = best_record["catalyst"]
            current_candidate = {
                "catalyst": catalyst,
                "t_res": best_record["t_res"],
                "temperature": best_record["temperature"],
                "catalyst_loading": best_record["catalyst_loading"],
            }
            current_score = task.scalarize(best_record, weight)
            
            for iteration in range(sub_budget):
                # Adaptive proposals similar to above
                num_proposals = max(2, min(10, 10 - iteration // (sub_budget // 3)))
                best_proposal_record = None
                best_proposal_score = -float('inf')
                best_proposal_candidate = None
                
                for _ in range(num_proposals):
                    proposal = task.mutate_candidate(current_candidate, rng)
                    record = task.evaluate(experiment, proposal)
                    score = task.scalarize(record, weight)
                    
                    if score > best_proposal_score:
                        best_proposal_score = score
                        best_proposal_record = record
                        best_proposal_candidate = proposal
                
                history.append(best_proposal_record)
                
                if best_proposal_score > current_score:
                    current_candidate = best_proposal_candidate
                    current_score = best_proposal_score
                else:
                    accept_probability = 0.1 * (1.0 - iteration / sub_budget)
                    if rng.random() < accept_probability:
                        current_candidate = best_proposal_candidate
                        current_score = best_proposal_score
    
    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "screen_and_local_search",
        "seed": seed,
        "budget": budget,
        "history": history[:budget],
        "summary": task.summarize(history[:budget]),
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
