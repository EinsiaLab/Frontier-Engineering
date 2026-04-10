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
SEED = 0


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
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing reference points: {ref_path}. "
            "Run benchmarks/Aerodynamics/CarAerodynamicsSensing/references/extract_car_mesh.py first."
        )
    points = np.load(ref_path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reference points must have shape (M, 3).")
    return points


def _kmeans_sampling(points: np.ndarray, k: int, *, seed: int) -> list[int]:
    m = int(points.shape[0])
    if m < k:
        raise ValueError(f"Not enough reference points: M={m} < k={k}")

    rng = np.random.default_rng(int(seed))
    
    # Initialize centroids using random points
    centroids = points[rng.choice(m, k, replace=False)]
    
    # Run a few iterations of K-means (since we don't need perfect convergence)
    for iteration in range(10):
        # Assign each point to the closest centroid
        distances = np.linalg.norm(points - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.zeros((k, 3))
        for i in range(k):
            cluster_points = points[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
        centroids = new_centroids
    
    # For each centroid, find the closest point in the original set
    indices = []
    for i in range(k):
        distances = np.linalg.norm(points - centroids[i], axis=1)
        closest_idx = np.argmin(distances)
        indices.append(closest_idx)
    
    # Ensure unique indices (in case two centroids pick the same point)
    unique_indices = list(set(indices))
    if len(unique_indices) < k:
        # Add additional points that are farthest from existing selected points
        remaining = k - len(unique_indices)
        available = list(set(range(m)) - set(unique_indices))
        if len(available) < remaining:
            raise RuntimeError("Cannot find enough unique points.")
        # Compute distances from each available point to selected points
        selected_points = points[unique_indices]
        farthest_scores = np.zeros(len(available))
        for idx, avail_idx in enumerate(available):
            # Minimum distance to any selected point
            min_dist = np.min(np.sum((selected_points - points[avail_idx]) ** 2, axis=1))
            farthest_scores[idx] = min_dist
        # Choose the points with largest minimum distance
        sorted_indices = np.argsort(farthest_scores)[::-1]
        extra = [available[i] for i in sorted_indices[:remaining]]
        unique_indices.extend(extra)
    
    return [int(idx) for idx in unique_indices]


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

indices = _kmeans_sampling(ref_points, SENSOR_NUM, seed=SEED)

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

