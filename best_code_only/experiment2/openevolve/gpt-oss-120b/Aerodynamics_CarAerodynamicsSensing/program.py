# EVOLVE-BLOCK-START
# CarAerodynamicsSensing – select 30 sensor points on the car surface.

import json
import os
from pathlib import Path

import numpy as np

SENSOR_NUM = 30
SEED = 0
# Number of candidate sensor sets to generate and evaluate.
# A handful of trials gives a better‑spread solution at negligible cost.
NUM_TRIALS = 5


def _find_repo_root():
    env = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Fallback for manual runs from a repo checkout.
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
        raise FileNotFoundError(
            f"Missing reference points: {ref_path}. "
            "Run benchmarks/Aerodynamics/CarAerodynamicsSensing/references/extract_car_mesh.py first."
        )
    points = np.load(ref_path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reference points must have shape (M, 3).")
    return points


def _farthest_point_sampling(points, k, *, seed):
    """
    Farthest‑point sampling with a *random* start point controlled by ``seed``.
    The first point is chosen randomly (but reproducibly) and the remaining
    points are added greedily by farthest‑point sampling.
    """
    m = points.shape[0]
    if m < k:
        raise ValueError(f"Not enough reference points: M={m} < k={k}")

    # --- Random first point (deterministic via seed) ---
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, m))       # plain Python int, reproducible

    selected = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)

    # --- Remaining points (standard farthest‑point sampling) ---
    for _ in range(k - 1):
        idx = np.argmax(dist2)
        selected.append(int(idx))
        d2 = ((points - points[idx]) ** 2).sum(axis=1)
        dist2 = np.minimum(dist2, d2)

    # safety check – should never happen
    if len(set(selected)) != k:
        raise RuntimeError("Internal error: duplicate indices selected.")
    return selected


repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

# ----------------------------------------------------------------------
# Choose the best candidate among a few farthest‑point‑sampling runs.
# ----------------------------------------------------------------------
def _min_pairwise_dist(idx_list: list[int], pts: np.ndarray) -> float:
    """Return the minimal Euclidean distance between any two points in idx_list."""
    sel = pts[idx_list]                     # shape (k, 3)
    diff = sel[:, None, :] - sel[None, :, :]  # (k, k, 3)
    d2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)            # ignore self‑distances
    return float(np.sqrt(d2.min()))


def _refine_selected(
    points: np.ndarray,
    selected: list[int],
    iterations: int = 2000,
    *,
    seed: int = 0,
) -> list[int]:
    """
    Greedy hill‑climbing refinement: repeatedly propose swapping one selected
    index with one unselected index and keep the swap only if it *increases*
    the minimal pairwise distance among the selected points.
    """
    rng = np.random.default_rng(seed)
    sel_set = set(selected)

    # current quality
    best_dist = _min_pairwise_dist(selected, points)

    all_idx = np.arange(points.shape[0])
    not_selected = np.setdiff1d(all_idx, list(sel_set), assume_unique=True)

    for _ in range(iterations):
        out_idx = int(rng.choice(not_selected))
        in_pos = int(rng.integers(0, len(selected)))

        trial = selected.copy()
        trial[in_pos] = out_idx
        trial_dist = _min_pairwise_dist(trial, points)

        if trial_dist > best_dist:
            selected = trial
            sel_set = set(selected)
            best_dist = trial_dist
            not_selected = np.setdiff1d(all_idx, list(sel_set), assume_unique=True)

    return selected


best_idxs: list[int] = []
best_min_dist = -1.0
rng = np.random.default_rng(SEED)

for trial in range(NUM_TRIALS):
    # Vary the seed so the first point can differ (random start).
    trial_seed = int(rng.integers(0, 2**31 - 1))
    cand = _farthest_point_sampling(ref_points, SENSOR_NUM, seed=trial_seed)
    cand_min = _min_pairwise_dist(cand, ref_points)
    if cand_min > best_min_dist:
        best_min_dist = cand_min
        best_idxs = cand

# optional greedy refinement to further increase spread
indices = _refine_selected(ref_points, best_idxs, iterations=2000, seed=SEED)

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

