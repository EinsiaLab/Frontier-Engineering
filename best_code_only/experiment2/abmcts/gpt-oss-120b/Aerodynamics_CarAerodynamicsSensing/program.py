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
    """Select k points using a lightweight K‑means clustering heuristic.

    The algorithm runs a small number of Lloyd iterations and then picks the
    nearest reference point to each centroid.  It guarantees exactly k unique
    indices.
    """
    rng = np.random.default_rng(int(seed))
    m = points.shape[0]
    if k > m:
        raise ValueError(f"Requested {k} samples but only {m} points are available.")

    # Initialise centroids with k distinct random points.
    init_idxs = rng.choice(m, size=k, replace=False)
    centroids = points[init_idxs].astype(np.float64)

    for _ in range(10):  # fixed small number of iterations for speed
        # Compute squared Euclidean distance from each point to each centroid.
        # Shape: (m, k)
        dists = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)

        new_centroids = np.empty_like(centroids)
        for ki in range(k):
            mask = assignments == ki
            if np.any(mask):
                new_centroids[ki] = points[mask].mean(axis=0)
            else:
                # Empty cluster – re‑initialise to a random point.
                new_centroids[ki] = points[rng.integers(0, m)]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Map each centroid to its nearest reference point.
    nearest_idxs = np.argmin(
        np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=0
    ).tolist()

    # Ensure uniqueness; replace duplicates with the next best unused points.
    unique_idxs = []
    used = set()
    for idx in nearest_idxs:
        if idx not in used:
            unique_idxs.append(int(idx))
            used.add(idx)
        else:
            # Find the closest unused point to the duplicated centroid.
            dists_to_dup = np.sum((points - points[idx]) ** 2, axis=1)
            sorted_candidates = np.argsort(dists_to_dup)
            for cand in sorted_candidates:
                if cand not in used:
                    unique_idxs.append(int(cand))
                    used.add(cand)
                    break

    # Fill any remaining slots (unlikely) with random unused points.
    if len(unique_idxs) < k:
        remaining = list(set(range(m)) - used)
        extra = rng.choice(remaining, size=k - len(unique_idxs), replace=False)
        unique_idxs.extend(map(int, extra))

    return unique_idxs


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

indices = _kmeans_sampling(ref_points, SENSOR_NUM, seed=SEED)

Path("submission.json").write_text(
    json.dumps({"indices": indices}, indent=2), encoding="utf-8"
)
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

