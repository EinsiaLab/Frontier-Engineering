"""Simple baseline for the SnAr multi-objective task."""

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

from shared.utils import dump_json, seed_everything
from snar_multiobjective import task


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []
    
    # Use a diverse set of weights
    weights = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    # Track best candidate and score for each weight
    best_candidates = {w: None for w in weights}
    best_scores = {w: -float('inf') for w in weights}
    
    # Pareto archive to maintain diverse high-performing solutions
    pareto_archive = []
    
    # Adaptive weight probabilities: start uniform, then adjust based on improvement
    weight_probs = np.array([1.0 / len(weights)] * len(weights))
    
    for step in range(budget):
        # Phase-based strategy: initial exploration (first 8 steps)
        if step < 8:
            # Pure random sampling to build diverse initial data
            candidate = task.sample_candidate(rng)
        else:
            # Select weight using adaptive probabilities
            weight = rng.choice(weights, p=weight_probs)
            
            # Choose candidate source: best for selected weight or Pareto archive
            if best_candidates[weight] is None:
                candidate = task.sample_candidate(rng)
            else:
                # 70% mutate best for weight, 20% mutate from Pareto archive, 10% random
                choice = rng.random()
                if choice < 0.7:
                    candidate = task.mutate_candidate(best_candidates[weight], rng)
                elif choice < 0.9:
                    # Select from Pareto archive if available
                    if pareto_archive:
                        parent_idx = rng.choice(len(pareto_archive))
                        parent = {name: pareto_archive[parent_idx][name] for name in task.INPUT_NAMES}
                        candidate = task.mutate_candidate(parent, rng)
                    else:
                        candidate = task.mutate_candidate(best_candidates[weight], rng)
                else:
                    candidate = task.sample_candidate(rng)
        
        record = task.evaluate(experiment, candidate)
        history.append(record)
        
        # Update best scores and candidates for each weight
        for w in weights:
            score = task.scalarize(record, w)
            if score > best_scores[w]:
                best_scores[w] = score
                best_candidates[w] = {name: record[name] for name in task.INPUT_NAMES}
        
        # Update Pareto archive with non-dominated solutions
        # First, check if the new record is dominated by any existing archive member
        is_dominated = False
        for arch in pareto_archive:
            if (arch['sty'] >= record['sty'] and arch['e_factor'] <= record['e_factor']):
                is_dominated = True
                break
        
        # If not dominated, add to archive
        if not is_dominated:
            pareto_archive.append(record)
            # Remove any solutions in the archive that are dominated by the new record
            new_archive = []
            for arch in pareto_archive:
                # Check if arch is dominated by record
                if (record['sty'] >= arch['sty'] and record['e_factor'] <= arch['e_factor']):
                    # arch is dominated by record, skip it
                    continue
                # Also check if arch is dominated by any other in the archive
                dominated_by_other = False
                for other in pareto_archive:
                    if other != arch and (other['sty'] >= arch['sty'] and other['e_factor'] <= arch['e_factor']):
                        dominated_by_other = True
                        break
                if not dominated_by_other:
                    new_archive.append(arch)
            pareto_archive = new_archive
        
        # Limit archive size to maintain diversity (max 6)
        if len(pareto_archive) > 6:
            # Sort by sty and compute crowding distance
            sorted_by_sty = sorted(pareto_archive, key=lambda x: x['sty'])
            distances = []
            for i, sol in enumerate(sorted_by_sty):
                if i == 0 or i == len(sorted_by_sty) - 1:
                    distance = float('inf')
                else:
                    sty_dist = sorted_by_sty[i+1]['sty'] - sorted_by_sty[i-1]['sty']
                    # Sort by e_factor to get neighbors in that dimension
                    sorted_by_ef = sorted(pareto_archive, key=lambda x: x['e_factor'])
                    idx_ef = sorted_by_ef.index(sol)
                    if idx_ef == 0 or idx_ef == len(sorted_by_ef) - 1:
                        ef_dist = float('inf')
                    else:
                        ef_dist = sorted_by_ef[idx_ef+1]['e_factor'] - sorted_by_ef[idx_ef-1]['e_factor']
                    distance = sty_dist + ef_dist
                distances.append(distance)
            # Find solution with smallest crowding distance
            min_idx = 0
            min_distance = float('inf')
            for i, arch in enumerate(pareto_archive):
                pos = sorted_by_sty.index(arch)
                if distances[pos] < min_distance:
                    min_distance = distances[pos]
                    min_idx = i
            pareto_archive.pop(min_idx)
        
        # Update weight probabilities based on recent improvements and archive diversity
        if step >= 8:
            # Compute improvement ratio for each weight
            improvements = []
            for w in weights:
                # Find best score for this weight in recent steps (last 5)
                recent_history = history[-5:] if len(history) >= 5 else history
                recent_scores = [task.scalarize(rec, w) for rec in recent_history]
                if recent_scores:
                    max_recent = max(recent_scores)
                    improvements.append(max_recent - best_scores[w])
                else:
                    improvements.append(0.0)
            
            # Compute diversity contribution for each weight
            diversity_scores = []
            for w in weights:
                if pareto_archive:
                    scores = [task.scalarize(arch, w) for arch in pareto_archive]
                    max_archive_score = max(scores)
                    diversity_scores.append(1.0 / (max_archive_score + 1e-6))
                else:
                    diversity_scores.append(1.0)
            
            # Combine improvements and diversity (weighted 0.6 and 0.4)
            if sum(improvements) > 0:
                imp_probs = np.array(improvements) / sum(improvements)
            else:
                imp_probs = np.array([1.0 / len(weights)] * len(weights))
            
            div_probs = np.array(diversity_scores) / sum(diversity_scores)
            
            # Blend: 0.6 * improvement + 0.4 * diversity + 0.1 * uniform
            weight_probs = 0.6 * imp_probs + 0.4 * div_probs + 0.1 * (1.0 / len(weights))
            weight_probs = weight_probs / sum(weight_probs)  # Normalize
    
    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "improved_pareto_weight_tracking",
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
