# EVOLVE-BLOCK-START
"""
CarAerodynamicsSensing (Aerodynamics)

Goal: choose 30 sensor locations (as indices into a fixed reference point set on a 3D car surface)
to minimize reconstruction error of the full pressure field.

Uses data-driven sensor placement via QR with column pivoting on training pressure data,
falling back to FPS on geometry if data is unavailable.
"""

import json
import os
import glob
from pathlib import Path

import numpy as np

SENSOR_NUM = 30
SEED = 0


def _find_repo_root() -> Path:
    env = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _load_reference_points(repo_root: Path) -> np.ndarray:
    ref_path = (
        repo_root / "benchmarks" / "Aerodynamics" / "CarAerodynamicsSensing"
        / "references" / "car_surface_points.npy"
    )
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference points: {ref_path}.")
    points = np.load(ref_path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reference points must have shape (M, 3).")
    return points


def _load_pressure_data(repo_root: Path) -> np.ndarray | None:
    """Try to load training pressure fields. Returns (N_samples, M_points) or None."""
    base = repo_root / "benchmarks" / "Aerodynamics" / "CarAerodynamicsSensing"
    
    # Try multiple possible data locations
    candidates = [
        base / "references" / "car_surface_pressures.npy",
        base / "data" / "car_surface_pressures.npy",
        base / "training_data" / "pressures.npy",
    ]
    
    for p in candidates:
        if p.exists():
            data = np.load(p)
            print(f"Loaded pressure data from {p}, shape={data.shape}")
            return data
    
    # Try loading from individual VTK/npy files in a data directory
    for data_dir in [base / "data", base / "references", base / "training_data"]:
        if data_dir.is_dir():
            npy_files = sorted(glob.glob(str(data_dir / "*.npy")))
            # Filter out the points file
            npy_files = [f for f in npy_files if "points" not in os.path.basename(f).lower()]
            if len(npy_files) > 1:
                try:
                    arrays = [np.load(f) for f in npy_files[:500]]
                    if all(a.ndim == 1 and a.shape == arrays[0].shape for a in arrays):
                        data = np.stack(arrays, axis=0)
                        print(f"Loaded {len(arrays)} pressure fields from {data_dir}")
                        return data
                except Exception:
                    pass
    
    return None


def _qr_pivot_selection(data: np.ndarray, k: int) -> list[int]:
    """
    Select k sensor locations using QR with column pivoting on the data matrix.
    data: (N_samples, M_points) pressure matrix
    Returns list of k indices into M_points.
    """
    # Center the data
    mean = data.mean(axis=0)
    centered = data - mean
    
    # Compute SVD to get dominant modes, then use QR pivoting on modes
    n_modes = min(k * 3, min(centered.shape) - 1, 100)
    if n_modes < k:
        n_modes = min(k, min(centered.shape) - 1)
    
    # U, S, Vt where Vt has shape (n_modes, M)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Take top n_modes
    Vt_trunc = Vt[:n_modes, :]  # (n_modes, M)
    
    # QR with column pivoting on Vt_trunc.T would give us pivot columns
    # But we want to pivot on columns of Vt_trunc (i.e., spatial locations)
    # scipy.linalg.qr with pivoting
    try:
        from scipy.linalg import qr
        _, _, piv = qr(Vt_trunc, pivoting=True)
        indices = [int(piv[i]) for i in range(k)]
    except ImportError:
        # Manual greedy pivoted QR
        indices = _greedy_qr_pivot(Vt_trunc, k)
    
    return indices


def _greedy_qr_pivot(V: np.ndarray, k: int) -> list[int]:
    """Greedy column pivoting: pick columns of V that maximize volume."""
    n_modes, m = V.shape
    norms2 = np.sum(V ** 2, axis=0)  # (M,)
    selected = []
    R = V.copy()
    
    for _ in range(k):
        col_norms = np.sum(R ** 2, axis=0)
        idx = int(np.argmax(col_norms))
        selected.append(idx)
        
        # Orthogonalize: project out the selected column direction
        col = R[:, idx].copy()
        col_norm = np.sqrt(np.sum(col ** 2))
        if col_norm > 1e-12:
            col /= col_norm
            # R = R - col * (col^T @ R)
            proj = col @ R  # (M,)
            R -= np.outer(col, proj)
        
        # Zero out selected column to avoid re-selection
        R[:, idx] = 0.0
    
    return selected


def _farthest_point_sampling(points: np.ndarray, k: int, *, seed: int) -> list[int]:
    m = int(points.shape[0])
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, m))
    selected = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(k - 1):
        idx = int(np.argmax(dist2))
        selected.append(idx)
        d2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)
    return selected


def _hybrid_selection(points: np.ndarray, k: int, seed: int) -> list[int]:
    """
    Try multiple FPS seeds and pick the best geometric spread.
    Also try starting from extremal points (min/max along each axis).
    """
    best_indices = None
    best_min_dist = -1.0
    
    # Try different seeds
    for s in range(20):
        indices = _farthest_point_sampling(points, k, seed=s)
        # Evaluate: minimum pairwise distance (higher is better for coverage)
        pts = points[indices]
        dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_indices = indices
    
    # Try starting from extremal points along each axis
    for axis in range(3):
        for func in [np.argmin, np.argmax]:
            start = int(func(points[:, axis]))
            selected = [start]
            dist2 = np.sum((points - points[start]) ** 2, axis=1)
            for _ in range(k - 1):
                idx = int(np.argmax(dist2))
                selected.append(idx)
                d2 = np.sum((points - points[idx]) ** 2, axis=1)
                dist2 = np.minimum(dist2, d2)
            pts = points[selected]
            dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = dists.min()
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_indices = selected
    
    return best_indices


# Main execution
repo_root = _find_repo_root()
ref_points = _load_reference_points(repo_root)

# Try data-driven approach first
pressure_data = _load_pressure_data(repo_root)

if pressure_data is not None and pressure_data.shape[0] >= SENSOR_NUM:
    print("Using data-driven QR pivot sensor selection")
    indices = _qr_pivot_selection(pressure_data, SENSOR_NUM)
    # Validate
    if len(set(indices)) != SENSOR_NUM:
        print("QR pivot produced duplicates, falling back to hybrid FPS")
        indices = _hybrid_selection(ref_points, SENSOR_NUM, SEED)
else:
    print("No pressure data found, using hybrid FPS with multiple seeds")
    indices = _hybrid_selection(ref_points, SENSOR_NUM, SEED)

# Ensure exactly SENSOR_NUM unique indices
indices = list(dict.fromkeys(indices))[:SENSOR_NUM]
assert len(indices) == SENSOR_NUM, f"Expected {SENSOR_NUM} indices, got {len(indices)}"

Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json")
# EVOLVE-BLOCK-END

