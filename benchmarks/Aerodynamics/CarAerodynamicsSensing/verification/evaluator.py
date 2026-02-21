#!/usr/bin/env python3
"""Evaluator for CarAerodynamicsSensing."""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

DATA_DIR = Path("/data/physense_car_data")
CKPT_PATH = Path("/data/physense_car_ckpt/physense_transolver_car_base.pth")

SENSOR_NUM = 30
CASE_START = 76
CASE_END = 100
SAMPLE_K = 10
SAMPLE_SEED = 2025

SIGMA_CLIP = 3.0
P_MIN = -844.3360
P_MAX = 602.6890


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "PhySense").is_dir() and (parent / "Frontier-Engineering").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root containing PhySense and Frontier-Engineering.")


def load_case(case_id: int, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pressure_path = data_dir / "pressure_files" / f"case_{case_id}_p_car_patch.raw"
    if not pressure_path.exists():
        raise FileNotFoundError(f"Missing pressure file: {pressure_path}")

    arr = np.loadtxt(pressure_path, comments="#", dtype=np.float32)
    coords = arr[:, :3]
    pressures = arr[:, 3]

    mean = pressures.mean()
    std = pressures.std()
    lower = mean - SIGMA_CLIP * std
    upper = mean + SIGMA_CLIP * std
    mask = (pressures >= lower) & (pressures <= upper)

    coords = coords[mask]
    pressures = pressures[mask]
    pressures = (pressures - P_MIN) / (P_MAX - P_MIN)

    return coords, pressures


def load_reference_points(ref_path: Path, data_dir: Path) -> np.ndarray:
    if ref_path.exists():
        points = np.load(ref_path)
    else:
        points, _ = load_case(1, data_dir)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(ref_path, points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reference points must have shape (M, 3).")
    return points


def parse_submission(path: Path, max_index: int) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission file: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        indices = data
    elif isinstance(data, dict) and "indices" in data:
        indices = data["indices"]
    else:
        raise ValueError("Submission must be a list or a dict with key 'indices'.")

    if len(indices) != SENSOR_NUM:
        raise ValueError(f"Expected {SENSOR_NUM} indices, got {len(indices)}")
    if len(set(indices)) != SENSOR_NUM:
        raise ValueError("Indices must be unique.")
    for idx in indices:
        if not isinstance(idx, int):
            raise ValueError("All indices must be integers.")
        if idx < 0 or idx >= max_index:
            raise ValueError(f"Index out of range: {idx} (max {max_index - 1})")
    return indices


def select_cases() -> list[int]:
    cases = list(range(CASE_START, CASE_END + 1))
    rng = random.Random(SAMPLE_SEED)
    selected = rng.sample(cases, SAMPLE_K)
    selected.sort()
    return selected


def snap_to_case(ref_points: np.ndarray, case_pos: torch.Tensor) -> torch.Tensor:
    ref_t = torch.tensor(ref_points, device=case_pos.device, dtype=case_pos.dtype)
    dist = torch.cdist(ref_t, case_pos)
    idx = torch.argmin(dist, dim=1)
    return case_pos[idx]


def load_model(device: torch.device):
    repo_root = find_repo_root(Path(__file__).resolve())
    physense_car = repo_root / "PhySense" / "Car-Aerodynamics"
    sys.path.insert(0, str(physense_car))

    from models import physense_transolver_car_walk

    model = physense_transolver_car_walk.Model(
        n_hidden=374,
        n_layers=12,
        space_dim=3,
        fun_dim=0,
        n_head=8,
        mlp_ratio=2,
        out_dim=1,
        slice_num=32,
        unified_pos=1,
    ).to(device)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(submission_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluator.")
    device = torch.device("cuda")

    task_root = Path(__file__).resolve().parents[1]
    ref_path = task_root / "references" / "car_surface_points.npy"

    ref_points = load_reference_points(ref_path, DATA_DIR)
    indices = parse_submission(submission_path, ref_points.shape[0])
    selected_ref = ref_points[np.array(indices, dtype=np.int64)]

    cases = select_cases()
    model = load_model(device)

    rel_losses = []
    with torch.no_grad():
        for case_id in cases:
            torch.manual_seed(SAMPLE_SEED + case_id)

            coords, pressures = load_case(case_id, DATA_DIR)
            pos = torch.from_numpy(coords).to(device)
            y = torch.from_numpy(pressures).to(device).unsqueeze(-1)

            sensor_pos = snap_to_case(selected_ref, pos)
            model.xyz_sens = torch.nn.Parameter(sensor_pos, requires_grad=False)

            data = SimpleNamespace(
                pos=pos,
                y=y,
                v=torch.tensor(0.0, device=device),
                angle=torch.tensor(0.0, device=device),
            )

            rel = model.sample(data).item()
            rel_losses.append(rel)
            print(f"case_{case_id}: rel_l2={rel:.6f}")

    mean_rel = float(np.mean(rel_losses))
    score = 1.0 - mean_rel

    print(f"mean_rel_l2: {mean_rel:.6f}")
    print(f"score: {score:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("submission.json"))
    args = parser.parse_args()

    evaluate(args.submission)


if __name__ == "__main__":
    main()
