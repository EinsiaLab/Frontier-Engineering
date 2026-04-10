#!/usr/bin/env python
# EVOLVE-BLOCK-START
"""Baseline solver for Task 03: Dammann-like 1D binary phase grating transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from diffractio import um, mm
from diffractio.scalar_masks_X import Scalar_mask_X


DEFAULT_CONFIG: Dict[str, Any] = {
    "period_size": 40 * um,
    "wavelength": 0.6328 * um,
    "period_pixels": 256,
    "num_transitions": 14,
    "num_repetitions": 10,
    "focal": 1 * mm,
    "lens_radius": 1 * mm,
    "order_min": -3,
    "order_max": 3,
    "order_window_halfwidth_px": 3,
}


def build_problem(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    x_period = np.linspace(-cfg["period_size"] / 2, cfg["period_size"] / 2, cfg["period_pixels"])

    return {
        "cfg": cfg,
        "x_period": x_period,
    }


def baseline_transitions(problem: Dict[str, Any]) -> np.ndarray:
    cfg = problem["cfg"]
    n_trans = cfg["num_transitions"]
    n_half = n_trans // 2
    # Symmetric transition parameterization: generate positive half, mirror to negative side
    # This reduces search space by half and improves order balance between +/- orders
    # Use power-law spacing to match typical Dammann grating transition distribution (better starting point)
    pos = np.power(np.linspace(0.0, 1.0, n_half), 1.2) * 0.48 * cfg["period_size"] + 0.01 * cfg["period_size"]
    pos += np.random.normal(0, cfg["period_size"] * 0.004, size=pos.shape)
    pos = np.clip(pos, 0.005 * cfg["period_size"], 0.495 * cfg["period_size"])
    pos.sort()
    transitions = np.concatenate([-pos[::-1], pos])
    return transitions


def build_incident_field(problem: Dict[str, Any], transitions: np.ndarray) -> Scalar_mask_X:
    cfg = problem["cfg"]
    x_period = problem["x_period"]

    period = Scalar_mask_X(x=x_period, wavelength=cfg["wavelength"])
    period.binary_code_positions(x_transitions=transitions, start="down", has_draw=False)
    period.u = np.exp(1j * np.pi * period.u)

    dammann = period.repeat_structure(
        num_repetitions=cfg["num_repetitions"],
        position="center",
        new_field=True,
    )

    lens = Scalar_mask_X(x=dammann.x, wavelength=cfg["wavelength"])
    lens.lens(x0=0.0, focal=cfg["focal"], radius=cfg["lens_radius"])

    return dammann * lens


def evaluate_orders(problem: Dict[str, Any], intensity_x: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
    cfg = problem["cfg"]
    spacing = cfg["focal"] * cfg["wavelength"] / cfg["period_size"]

    orders = np.arange(cfg["order_min"], cfg["order_max"] + 1, dtype=int)
    energies = []
    positions = []

    hw = int(cfg["order_window_halfwidth_px"])
    for m in orders:
        x_m = m * spacing
        ix = int(np.argmin(np.abs(x - x_m)))
        i0 = max(0, ix - hw)
        i1 = min(len(x), ix + hw + 1)
        energies.append(float(intensity_x[i0:i1].sum()))
        positions.append(float(x_m))

    energies = np.asarray(energies, dtype=float)
    cv = float(energies.std() / (energies.mean() + 1e-12))
    norm = energies / (energies.max() + 1e-12)
    efficiency = float(energies.sum() / (intensity_x.sum() + 1e-12))

    return {
        "orders": orders.tolist(),
        "order_positions": positions,
        "order_energies": energies.tolist(),
        "order_energies_norm": norm.tolist(),
        "cv_orders": cv,
        "efficiency": efficiency,
        "min_to_max": float(norm.min()),
    }


def solve_baseline(problem: Dict[str, Any]) -> Dict[str, Any]:
    cfg = problem["cfg"]
    np.random.seed(42)  # Enforce deterministic results per constraint requirements
    best_transitions = baseline_transitions(problem)
    best_score = -1.0
    best_solution = None
    step = cfg["period_size"] * 0.004  # Initial step size for perturbation
    min_spacing = cfg["period_size"] * 0.015  # Minimum allowed spacing between transitions (improves manufacturability + performance)
    
    # Hill-climbing optimization to find optimal transition positions
    for iter_idx in range(50):  # Increased iterations for better search coverage
        field = build_incident_field(problem, best_transitions)
        focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
        intensity = np.abs(focus_field.u) ** 2
        metrics = evaluate_orders(problem, intensity, focus_field.x)
        
        # Calculate score exactly matching official fitness formula to optimize target objective
        uniform_score = np.clip(1 - metrics["cv_orders"] / 0.9, 0.0, 1.0)
        efficiency_score = np.clip((metrics["efficiency"] - 0.003) / (0.18 - 0.003), 0.0, 1.0)
        balance_score = np.clip((metrics["min_to_max"] - 0.15) / (0.90 - 0.15), 0.0, 1.0)
        current_score = 0.6 * uniform_score + 0.3 * efficiency_score + 0.1 * balance_score
        
        if current_score > best_score:
            best_score = current_score
            best_solution = {
                "transitions": best_transitions.copy(),
                "x_focus": focus_field.x,
                "intensity_focus": intensity,
                "metrics": metrics,
            }
        
        # Generate symmetric candidate: only perturb positive half, mirror to maintain symmetry
        n_trans = len(best_transitions)
        n_half = n_trans // 2
        best_pos = best_transitions[n_half:]
        candidate_pos = best_pos + np.random.normal(0, step, size=best_pos.shape)
        candidate_pos = np.clip(candidate_pos, 0.005 * cfg["period_size"], 0.495 * cfg["period_size"])
        candidate_pos.sort()
        
        # Enforce minimum spacing between adjacent transitions
        while np.any(np.diff(candidate_pos) < min_spacing):
            for i in range(1, len(candidate_pos)):
                if candidate_pos[i] - candidate_pos[i-1] < min_spacing:
                    candidate_pos[i] = candidate_pos[i-1] + min_spacing
            candidate_pos = np.clip(candidate_pos, 0.005 * cfg["period_size"], 0.495 * cfg["period_size"])
            candidate_pos.sort()
        
        candidate = np.concatenate([-candidate_pos[::-1], candidate_pos])
        best_transitions = candidate
        step *= 0.95  # Decay step size over iterations for finer convergence
    
    return best_solution


def save_solution(path: Path, solution: Dict[str, Any], problem: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        transitions=solution["transitions"].astype(np.float32),
        x_focus=solution["x_focus"].astype(np.float32),
        intensity_focus=solution["intensity_focus"].astype(np.float32),
        period_size=np.float32(problem["cfg"]["period_size"]),
        wavelength=np.float32(problem["cfg"]["wavelength"]),
        focal=np.float32(problem["cfg"]["focal"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task03 baseline Dammann transition solver")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_solution.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON config overriding defaults",
    )
    args = parser.parse_args()

    config = None
    if args.config_json is not None:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))

    problem = build_problem(config)
    solution = solve_baseline(problem)
    save_solution(args.output, solution, problem)

    print("[Task03/Baseline] solution saved:", args.output)
    print("[Task03/Baseline] cv_orders={:.6f}, efficiency={:.6f}".format(
        solution["metrics"]["cv_orders"], solution["metrics"]["efficiency"]
    ))


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
