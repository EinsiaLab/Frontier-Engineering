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


def _farthest_point_sampling(points: np.ndarray, k: int) -> list[int]:
    m = int(points.shape[0])
    if m < k:
        raise ValueError(f"Not enough reference points: M={m} < k={k}")

    # Normalize points per axis to balance distance metric across all spatial dimensions
    # Prevents longer axes (e.g. car length) from dominating shorter axes (height/width) in sampling
    norm_points = (points - points.mean(axis=0)) / points.std(axis=0)
    
    # Start with point farthest from point cloud centroid for better initial global coverage
    centroid = np.mean(points, axis=0)
    dist_to_centroid = np.sum((points - centroid) ** 2, axis=1)
    first = int(np.argmax(dist_to_centroid))
    
    # Add small weight to prioritize high-gradient regions (front/rear, side edges) for better reconstruction
    # These regions have the highest pressure variance, so sensors here reduce error significantly
    x_center = np.mean(points[:, 0])
    y_abs = np.abs(points[:, 1])
    region_weight = (np.abs(points[:, 0] - x_center) * 0.15 + y_abs * 0.1 + 1.0)

    selected: list[int] = [first]
    dist2 = np.sum((norm_points - norm_points[first]) ** 2, axis=1)

    for _ in range(k - 1):
        idx = int(np.argmax(dist2 * region_weight))
        selected.append(idx)
        d2 = np.sum((norm_points - norm_points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)

    if len(set(selected)) != k:
        raise RuntimeError("Internal error: duplicate indices selected.")
    return selected


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

indices = _farthest_point_sampling(ref_points, SENSOR_NUM)

Path("submission.json").write_text(json.dumps({"indices": indices}), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

