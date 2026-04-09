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


def _aero_weighted_fps(points: np.ndarray, k: int, *, seed: int) -> list[int]:
    """FPS with aerodynamic importance weighting for pressure reconstruction.
    
    Prioritizes front/rear (stagnation/wake zones) and edges (flow separation).
    """
    m = len(points)
    if m < k:
        raise ValueError(f"Not enough points: {m} < {k}")
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Front (min x) and rear (max x) are aerodynamically critical
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    x_imp = np.maximum(x_norm, 1 - x_norm)  # High at both ends
    
    # Edges (y and z extremes) - flow separation zones
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
    edge_imp = np.maximum(y_norm, 1 - y_norm) + np.maximum(z_norm, 1 - z_norm)
    
    # Combined weights: front/rear stagnation + edges
    weights = x_imp + 0.3 * edge_imp
    weights = weights / np.max(weights) + 0.2
    
    rng = np.random.default_rng(seed)
    first = int(rng.choice(m, p=weights / np.sum(weights)))
    
    selected = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)
    
    for _ in range(k - 1):
        idx = int(np.argmax(dist2 * weights))
        selected.append(idx)
        dist2 = np.minimum(dist2, np.sum((points - points[idx]) ** 2, axis=1))
    
    if len(set(selected)) != k:
        raise RuntimeError("Duplicate indices selected.")
    return selected


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

indices = _aero_weighted_fps(ref_points, SENSOR_NUM, seed=SEED)

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

