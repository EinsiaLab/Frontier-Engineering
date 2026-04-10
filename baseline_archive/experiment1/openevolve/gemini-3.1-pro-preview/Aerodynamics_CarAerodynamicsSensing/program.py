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
    best_m, best_i = [], np.inf
    starts = [0] + [int(f(points[:, i])) for i in range(3) for f in (np.argmin, np.argmax)]
    
    for s in starts:
        m = [s]
        d2 = np.sum((points - points[s])**2, axis=1)
        for _ in range(k - 1):
            m.append(int(np.argmax(d2)))
            d2 = np.minimum(d2, np.sum((points - points[m[-1]])**2, axis=1))
            
        for _ in range(30):
            dists = np.sum((points[:, None] - points[m])**2, axis=-1)
            labels = np.argmin(dists, axis=1)
            new_m = [
                int(idx[np.argmin(np.sum((points[idx] - points[idx].mean(0))**2, axis=1))])
                if len(idx := np.where(labels == i)[0]) else m[i]
                for i in range(k)
            ]
            if m == new_m:
                break
            m = new_m
        else:
            dists = np.sum((points[:, None] - points[m])**2, axis=-1)
            
        inertia = float(np.sum(np.min(dists, axis=1)))
        if inertia < best_i and len(set(m)) == k:
            best_i, best_m = inertia, m

    return best_m if best_m else m


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

indices = _kmeans_sampling(ref_points, SENSOR_NUM, seed=SEED)

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

