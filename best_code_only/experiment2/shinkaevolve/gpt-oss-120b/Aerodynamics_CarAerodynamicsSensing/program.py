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


def _farthest_point_sampling(
    points: np.ndarray, k: int, *, seed: int
) -> tuple[list[int], np.ndarray]:
    """
    Standard farthest‑point‑sampling (FPS) that also returns the final
    minimum‑squared‑distance array.

    Parameters
    ----------
    points: np.ndarray
        Array of shape (M, 3) containing the reference points.
    k: int
        Number of points to sample.
    seed: int
        Random seed for the initial point selection.

    Returns
    -------
    tuple[list[int], np.ndarray]
        - List of selected point indices (length k, all unique).
        - ``dist2``: array of shape (M,) containing the squared distance
          from each reference point to its nearest selected point after the
          algorithm finishes.  The sum of this array is the coverage metric
          used for quality evaluation.
    """
    m = int(points.shape[0])
    if m < k:
        raise ValueError(f"Not enough reference points: M={m} < k={k}")

    # Deterministic initialization: choose the point farthest from the centroid.
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))

    selected: list[int] = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)

    for _ in range(k - 1):
        idx = int(np.argmax(dist2))
        selected.append(idx)
        d2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)

    if len(set(selected)) != k:
        raise RuntimeError("Internal error: duplicate indices selected.")
    return selected, dist2


def _select_best_fps(points: np.ndarray, k: int, seeds: list[int]) -> list[int]:
    """
    Run FPS with several seeds and keep the set that gives the smallest
    sum of squared distances from every point to its nearest selected point.
    This cheap proxy favours a more uniform coverage of the reference set.
    """
    best_indices: list[int] | None = None
    best_metric: float | None = None

    for s in seeds:
        # FPS now returns both the selected indices and the final dist2 array.
        cand, dist2 = _farthest_point_sampling(points, k, seed=s)
        # The coverage metric is simply the sum of the final minimum distances.
        metric = float(dist2.sum())

        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_indices = cand

    # seeds is guaranteed non‑empty, so best_indices is set.
    return best_indices  # type: ignore[return-value]


def _coverage_metric(points: np.ndarray, sel: list[int]) -> float:
    """
    Compute the sum of squared distances from every point to its nearest selected point.
    Used as a cheap proxy for sensor set quality.
    """
    sel_pts = points[sel]                     # (k, 3)
    d2 = np.sum((points[:, None, :] - sel_pts[None, :, :]) ** 2, axis=2)  # (M, k)
    min_d2 = np.min(d2, axis=1)               # (M,)
    return float(np.sum(min_d2))


def _refine_fps(
    points: np.ndarray,
    init_sel: list[int],
    *,
    rng: np.random.Generator,
    trials: int = 1000,
) -> list[int]:
    """
    Simple greedy refinement: repeatedly propose random swaps of one selected index
    with a non‑selected index and keep the swap if it improves the coverage metric.
    The number of random attempts is limited by ``trials`` to keep runtime modest.
    """
    sel = list(init_sel)
    best_metric = _coverage_metric(points, sel)
    m = points.shape[0]

    for _ in range(trials):
        # Choose a random element to remove and a random new candidate to add.
        out_idx = rng.choice(sel)
        in_idx = int(rng.integers(0, m))
        if in_idx in sel:
            continue

        # Perform the swap.
        new_sel = sel.copy()
        new_sel[sel.index(out_idx)] = in_idx
        metric = _coverage_metric(points, new_sel)

        # Accept if improvement.
        if metric < best_metric:
            sel = new_sel
            best_metric = metric

    return sel


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

# Try a few different seeds for FPS and keep the best coverage.
_candidate_seeds = [
    0,
    42,
    123,
    2025,
]
indices = _select_best_fps(ref_points, SENSOR_NUM, _candidate_seeds)

# Optional refinement to further improve uniformity.
_rng = np.random.default_rng(SEED)
indices = _refine_fps(ref_points, indices, rng=_rng, trials=1000)

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END