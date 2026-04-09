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
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[4]


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


def _farthest_point_sampling(points: np.ndarray, k: int, *, seed: int) -> list[int]:
    m = len(points)
    if m < k:
        raise ValueError(f"Not enough reference points: M={m} < k={k}")
    z = (points - points.mean(0)) / (points.std(0) + 1e-12)
    r2 = np.sum(z * z, 1)
    first = int(np.argmin(r2))
    selected = [first]
    dist2 = np.sum((z - z[first]) ** 2, 1)
    weight = 1.0 / (1.0 + 0.8 * r2 / (r2.mean() + 1e-12))
    for _ in range(k - 1):
        idx = int(np.argmax(dist2 * weight))
        selected.append(idx)
        dist2 = np.minimum(dist2, np.sum((z - z[idx]) ** 2, 1))
    return selected


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

indices = _farthest_point_sampling(ref_points, SENSOR_NUM, seed=SEED)

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

