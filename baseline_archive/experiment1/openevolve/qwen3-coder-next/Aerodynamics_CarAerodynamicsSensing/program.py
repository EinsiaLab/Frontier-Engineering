# EVOLVE-BLOCK-START
"""
CarAerodynamicsSensing (Aerodynamics)

Goal: choose 30 sensor locations (as indices into a fixed reference point set on a 3D car surface)
to minimize reconstruction error of the full pressure field.

Contract:
- Writes `submission.json` in the current working directory.
- JSON format: {"indices": [int, ...]} with exactly 30 unique indices.
"""

import json
import os
from pathlib import Path

import numpy as np

SENSOR_NUM = 30
NUM_RESTARTS = 5  # Multiple random restarts for better exploration
MAX_FPS_ITERATIONS = 100  # Limit FPS iterations for efficiency


def _find_repo_root() -> Path:
    env = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Fallback for manual runs from a repo checkout.
    return Path(__file__).resolve().parents[4]


def _load_reference_points(repo_root: Path) -> np.ndarray:
    ref_path = (
        repo_root
        / "benchmarks"
        / "Aerodynamics"
        / "CarAerodynamicsSensing"
        / "references"
        / "car_surface_points.npy"
    )
    points = np.load(ref_path)
    return points


def _farthest_point_sampling(points: np.ndarray, k: int, *, rng: np.random.Generator, seed_strategy: str = "random") -> list[int]:
    m = int(points.shape[0])
    
    # Different initialization strategies for better exploration
    if seed_strategy == "centroid":
        centroid = np.mean(points, axis=0)
        first = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
    elif seed_strategy == "corner":
        # Find point farthest from centroid (approximate corner)
        centroid = np.mean(points, axis=0)
        first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    else:  # random
        first = int(rng.integers(0, m))

    selected: list[int] = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)
    
    # Mark first selected point to prevent re-selection
    dist2[first] = float('inf')

    # Use set for O(1) lookup instead of list operations
    selected_set = set(selected)
    
    # Limit iterations for efficiency with large point clouds
    iterations = min(k - 1, MAX_FPS_ITERATIONS)
    for _ in range(iterations):
        idx = int(np.argmax(dist2))
        
        # Ensure uniqueness - if we somehow selected an already-chosen index, skip
        if idx in selected_set:
            dist2[idx] = float('inf')
            continue
            
        selected.append(idx)
        selected_set.add(idx)
        
        # Mark this index as unavailable immediately
        dist2[idx] = float('inf')
        
        d2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)

    # If we couldn't select enough points due to iteration limit, fill remaining
    if len(selected) < k:
        # Find unselected indices efficiently
        all_indices = np.arange(m)
        mask = np.ones(m, dtype=bool)
        mask[list(selected_set)] = False
        available_indices = all_indices[mask]
        
        if len(available_indices) > 0:
            num_needed = k - len(selected)
            chosen_indices = rng.choice(available_indices, size=min(num_needed, len(available_indices)), replace=False)
            selected.extend(chosen_indices.tolist())
    
    return selected[:k]  # Ensure exactly k indices


def _compute_coverage_score(ref_points: np.ndarray, indices: list[int]) -> float:
    """Compute comprehensive coverage score optimized for efficiency."""
    ref_subset = ref_points[indices]
    
    # 1. Coverage spread (larger is better) - dominant factor
    centroid = np.mean(ref_subset, axis=0)
    dists_to_centroid = np.linalg.norm(ref_subset - centroid, axis=1)
    spread_score = float(np.mean(dists_to_centroid))
    
    # 2. Uniformity (lower std of pairwise distances is better)
    # Use sampling for efficiency with large k
    n_indices = len(indices)
    if n_indices > 10:
        rng = np.random.default_rng(42)
        sample_size = min(100, n_indices * (n_indices - 1) // 2)
        # Generate all pairs efficiently
        all_pairs = [(i, j) for i in range(n_indices) for j in range(i+1, n_indices)]
        if len(all_pairs) > sample_size:
            sampled_indices = rng.choice(len(all_pairs), size=sample_size, replace=False)
            sampled_pairs = [all_pairs[i] for i in sampled_indices]
        else:
            sampled_pairs = all_pairs
        
        distances = [
            np.linalg.norm(ref_subset[i] - ref_subset[j])
            for i, j in sampled_pairs
        ]
    else:
        pairwise_dists = np.linalg.norm(ref_subset[:, np.newaxis] - ref_subset, axis=2)
        triu_indices = np.triu_indices_from(pairwise_dists, k=1)
        distances = pairwise_dists[triu_indices]
    
    uniformity_score = float(np.std(distances)) if len(distances) > 1 else 0.0
    
    # 3. Minimum spacing (larger is better)
    min_dist = float(np.min(distances)) if len(distances) > 0 else 0.0
    
    # Combined score (to be minimized) - weighted combination
    # Emphasize minimum spacing more to ensure good coverage distribution
    return spread_score * 0.3 + uniformity_score * 0.3 - min_dist * 0.4


def main() -> None:
    repo_root = _find_repo_root()
    ref_points = _load_reference_points(repo_root)
    
    best_indices = None
    best_score = float("inf")
    
    # Try multiple strategies with different seed initialization methods
    strategies = ["random", "centroid", "corner"]
    
    for strategy in strategies:
        for restart_seed in range(NUM_RESTARTS):
            rng = np.random.default_rng(restart_seed)
            indices = _farthest_point_sampling(ref_points, SENSOR_NUM, rng=rng, seed_strategy=strategy)
            
            # Use more comprehensive coverage score
            score = _compute_coverage_score(ref_points, indices)
            
            if score < best_score:
                best_score = score
                best_indices = indices
    
    indices = best_indices
    
    Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
    print(f"Wrote {SENSOR_NUM} indices to submission.json")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END

