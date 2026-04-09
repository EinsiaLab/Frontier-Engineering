# EVOLVE-BLOCK-START
import json
import os
from pathlib import Path

import numpy as np


def _find_repo_root():
    env = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _load_reference_points(repo_root):
    ref_path = (
        repo_root
        / "benchmarks"
        / "Aerodynamics"
        / "CarAerodynamicsSensing"
        / "references"
        / "car_surface_points.npy"
    )
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference points: {ref_path}")
    points = np.load(ref_path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reference points must have shape (M, 3).")
    return points


def _farthest_point_sampling(points, k):
    m = int(points.shape[0])
    if m < k:
        raise ValueError(f"Not enough reference points: M={m} < k={k}")

    # Start FPS from highest point (max z). Roof/windshield pressures are
    # critical for lift and wake reconstruction in car aerodynamics; this
    # explores a different spatial prior than rear-bias while still spreading
    # sensors via FPS.
    first = int(np.argmax(points[:, 2]))

    selected = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)

    for _ in range(k - 1):
        idx = int(np.argmax(dist2))
        selected.append(idx)
        d2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)

    return selected


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

indices = _farthest_point_sampling(ref_points, 30)

Path("submission.json").write_text(json.dumps({"indices": indices}), encoding="utf-8")
print("Wrote 30 indices to submission.json")
# EVOLVE-BLOCK-END

