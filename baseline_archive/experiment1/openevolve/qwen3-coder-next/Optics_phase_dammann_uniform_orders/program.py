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
    period_size = cfg["period_size"]
    n_transitions = cfg["num_transitions"]
    
    # Use literature-inspired transitions as a starting point
    # These are normalized positions in [0, 1] from known good Dammann gratings
    literature_norm_positions = np.array([
        0.0645, 0.1290, 0.1935, 0.2581, 0.3226, 0.3871, 0.4516,
        0.5484, 0.6129, 0.6774, 0.7419, 0.8065, 0.8710, 0.9355
    ])
    
    # Trim or extend to match required number of transitions
    if n_transitions <= len(literature_norm_positions):
        norm_positions = literature_norm_positions[:n_transitions]
    else:
        # For more transitions, interpolate
        x = np.linspace(0, 1, n_transitions)
        norm_positions = np.interp(x, np.linspace(0, 1, len(literature_norm_positions)), literature_norm_positions)
    
    # Convert to physical positions with appropriate margin
    margin = 0.42  # Slightly reduced from 0.45 for better control
    transitions = (norm_positions - 0.5) * period_size * margin
    
    # Enforce symmetry for manufacturability and better performance
    half_n = n_transitions // 2
    if n_transitions % 2 == 1:  # Odd number: keep center transition at 0
        symmetric_transitions = np.zeros(n_transitions)
        symmetric_transitions[:half_n] = -transitions[half_n+1:][::-1]
        symmetric_transitions[half_n] = 0.0
        symmetric_transitions[half_n+1:] = transitions[half_n+1:]
        transitions = symmetric_transitions
    
    # Generate literature transitions for comparison
    lit_norm_positions = np.array([
        0.0, 0.201181, 0.250978, 0.326167, 0.370555, 0.372996, 0.396478,
        0.453128, 0.594731, 0.670591, 0.717718, 0.890632, 0.919921, 0.935546
    ])
    
    # Trim literature transitions to match required number
    if n_transitions <= len(lit_norm_positions):
        lit_norm = lit_norm_positions[:n_transitions]
    else:
        # For more transitions, interpolate
        x = np.linspace(0, 1, n_transitions)
        lit_norm = np.interp(x, np.linspace(0, 1, len(lit_norm_positions)), lit_norm_positions)
    
    # Convert literature transitions to physical positions
    lit_transitions = (lit_norm - 0.5) * period_size * 1.0  # Full period margin
    
    # Try both initial solutions and pick the better one
    field_lit = build_incident_field(problem, lit_transitions)
    focus_lit = field_lit.RS(z=cfg["focal"], new_field=True, verbose=False)
    intensity_lit = np.abs(focus_lit.u) ** 2
    metrics_lit = evaluate_orders(problem, intensity_lit, focus_lit.x)
    score_lit = calculate_composite_score(metrics_lit)
    
    field_init = build_incident_field(problem, transitions)
    focus_init = field_init.RS(z=cfg["focal"], new_field=True, verbose=False)
    intensity_init = np.abs(focus_init.u) ** 2
    metrics_init = evaluate_orders(problem, intensity_init, focus_init.x)
    score_init = calculate_composite_score(metrics_init)
    
    # Start with the better solution
    if score_lit > score_init:
        best_transitions = lit_transitions.copy()
        best_score = score_lit
    else:
        best_transitions = transitions.copy()
        best_score = score_init
    
    # Iterative refinement with decreasing step size and adaptive threshold
    step_sizes = [0.015, 0.008, 0.004]  # More conservative step sizes
    for step_frac in step_sizes:
        step = step_frac * period_size
        improved = True
        improvement_count = 0
        
        while improved and improvement_count < 3:  # Limit consecutive improvements
            improved = False
            for i in range(half_n):  # Only optimize half due to symmetry
                # Try positive shift
                test_transitions = best_transitions.copy()
                test_transitions[i] += step
                test_transitions[n_transitions-1-i] = -test_transitions[i]  # Maintain symmetry
                
                # Enforce boundary constraints with slight relaxation
                if abs(test_transitions[i]) < 0.45 * period_size:
                    field = build_incident_field(problem, test_transitions)
                    focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
                    intensity = np.abs(focus_field.u) ** 2
                    metrics = evaluate_orders(problem, intensity, focus_field.x)
                    
                    # Calculate composite score
                    test_score = calculate_composite_score(metrics)
                    
                    # Accept if score improved (relaxed threshold)
                    if test_score > best_score * 1.001:
                        best_transitions = test_transitions.copy()
                        best_score = test_score
                        improved = True
                        improvement_count += 1
                        continue
                
                # Try negative shift
                test_transitions = best_transitions.copy()
                test_transitions[i] -= step
                test_transitions[n_transitions-1-i] = -test_transitions[i]  # Maintain symmetry
                
                # Enforce boundary constraints with slight relaxation
                if abs(test_transitions[i]) < 0.45 * period_size:
                    field = build_incident_field(problem, test_transitions)
                    focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
                    intensity = np.abs(focus_field.u) ** 2
                    metrics = evaluate_orders(problem, intensity, focus_field.x)
                    
                    # Calculate composite score
                    test_score = calculate_composite_score(metrics)
                    
                    # Accept if score improved (relaxed threshold)
                    if test_score > best_score * 1.001:
                        best_transitions = test_transitions.copy()
                        best_score = test_score
                        improved = True
                        improvement_count += 1
    
    # Final fine-tuning with smaller steps and relaxed boundaries
    fine_step = 0.002 * period_size
    for _ in range(5):  # More refinement passes
        for i in range(n_transitions):
            for direction in [1, -1]:
                test_transitions = best_transitions.copy()
                test_transitions[i] += direction * fine_step
                
                # Slightly relaxed boundary for final refinement
                if abs(test_transitions[i]) < 0.45 * period_size:
                    try:
                        field = build_incident_field(problem, test_transitions)
                        focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
                        intensity = np.abs(focus_field.u) ** 2
                        metrics = evaluate_orders(problem, intensity, focus_field.x)
                        
                        test_score = calculate_composite_score(metrics)
                        
                        if test_score > best_score * 1.0005:  # Very small improvement threshold
                            best_transitions = test_transitions.copy()
                            best_score = test_score
                    except Exception:
                        pass
    
    return best_transitions


def calculate_composite_score(metrics: Dict[str, Any]) -> float:
    """Calculate composite score used in evaluation (higher is better)."""
    uniform_score = np.clip(1.0 - metrics["cv_orders"] / 0.9, 0.0, 1.0)
    efficiency_score = np.clip((metrics["efficiency"] - 0.003) / (0.18 - 0.003), 0.0, 1.0)
    balance_score = np.clip((metrics["min_to_max"] - 0.15) / (0.90 - 0.15), 0.0, 1.0)
    return float(0.60 * uniform_score + 0.30 * efficiency_score + 0.10 * balance_score)


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
    transitions = baseline_transitions(problem)
    field = build_incident_field(problem, transitions)
    focus_field = field.RS(z=problem["cfg"]["focal"], new_field=True, verbose=False)

    intensity = np.abs(focus_field.u) ** 2
    metrics = evaluate_orders(problem, intensity, focus_field.x)

    return {
        "transitions": transitions,
        "x_focus": focus_field.x,
        "intensity_focus": intensity,
        "metrics": metrics,
    }


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
