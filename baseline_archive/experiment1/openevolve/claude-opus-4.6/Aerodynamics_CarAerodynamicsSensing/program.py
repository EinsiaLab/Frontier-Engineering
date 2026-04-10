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


def _farthest_point_sampling_from(points: np.ndarray, k: int, first: int) -> list[int]:
    """FPS starting from a specific index."""
    m = int(points.shape[0])
    selected: list[int] = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(k - 1):
        idx = int(np.argmax(dist2))
        selected.append(idx)
        d2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)
    return selected


def _farthest_point_sampling(points: np.ndarray, k: int, *, seed: int) -> list[int]:
    m = int(points.shape[0])
    if m < k:
        raise ValueError(f"Not enough reference points: M={m} < k={k}")

    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, m))

    selected: list[int] = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)

    for _ in range(k - 1):
        idx = int(np.argmax(dist2))
        selected.append(idx)
        d2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)

    if len(set(selected)) != k:
        raise RuntimeError("Internal error: duplicate indices selected.")
    return selected


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

# Try multiple strategic starting points for FPS and pick best coverage
m = ref_points.shape[0]
candidates = []

# Strategy 1: centroid-seeded (scored 0.9618 previously)
centroid = np.mean(ref_points, axis=0)
centroid_idx = int(np.argmin(np.sum((ref_points - centroid) ** 2, axis=1)))
candidates.append(("centroid", _farthest_point_sampling_from(ref_points, SENSOR_NUM, centroid_idx)))

# Strategy 2: extremal points along each axis
for ax in range(3):
    for fn in [np.argmin, np.argmax]:
        idx = int(fn(ref_points[:, ax]))
        candidates.append((f"axis{ax}_{fn.__name__}", _farthest_point_sampling_from(ref_points, SENSOR_NUM, idx)))

# Strategy 3: random seeds 0-4
for s in range(5):
    rng = np.random.default_rng(s)
    first = int(rng.integers(0, m))
    candidates.append((f"seed{s}", _farthest_point_sampling_from(ref_points, SENSOR_NUM, first)))

# Pick the candidate with the best minimum pairwise distance (best spatial coverage)
best_indices = None
best_min_dist = -1.0
best_name = ""
for name, sel in candidates:
    if len(set(sel)) != SENSOR_NUM:
        continue
    pts = ref_points[sel]
    # Efficient min pairwise distance
    diffs = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_d = np.sqrt(np.min(d2))
    if min_d > best_min_dist:
        best_min_dist = min_d
        best_indices = sel
        best_name = name

indices = best_indices if best_indices is not None else _farthest_point_sampling(ref_points, SENSOR_NUM, seed=SEED)
print(f"Used FPS with start={best_name}, min_pairwise_dist={best_min_dist:.4f}")

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

