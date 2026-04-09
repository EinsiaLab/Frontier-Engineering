#!/usr/bin/env python
# EVOLVE-BLOCK-START
"""Baseline solver for Task 03: Dammann-like 1D binary phase grating transitions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from diffractio import um, mm
from diffractio.scalar_masks_X import Scalar_mask_X

# ----------------------------------------------------------------------
# Default configuration (can be overridden via JSON)
# ----------------------------------------------------------------------
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
    # optimization settings – modestly increased for higher quality
    "de_maxiter": 45,
    "de_popsize": 15,
    "de_seed": 0,
    # number of independent DE runs (deterministic seeds)
    "de_runs": 3,
    # local refinement
    "refine_iters": 150,
    "refine_step_frac": 0.02,  # fraction of period_size
}

# ----------------------------------------------------------------------
# Problem construction
# ----------------------------------------------------------------------
def build_problem(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    x_period = np.linspace(-cfg["period_size"] / 2, cfg["period_size"] / 2, cfg["period_pixels"])

    return {
        "cfg": cfg,
        "x_period": x_period,
    }

# ----------------------------------------------------------------------
# Simple evenly‑spaced baseline (kept for reference)
# ----------------------------------------------------------------------
def baseline_transitions(problem: Dict[str, Any]) -> np.ndarray:
    cfg = problem["cfg"]
    transitions = np.linspace(
        -0.45 * cfg["period_size"], 0.45 * cfg["period_size"], cfg["num_transitions"]
    )
    return transitions

# ----------------------------------------------------------------------
# Field construction
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Order evaluation
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Scoring function (higher is better)
# ----------------------------------------------------------------------
def _score_from_metrics(metrics: Dict[str, Any]) -> float:
    uniform_score = np.clip(1 - metrics["cv_orders"] / 0.9, 0, 1)
    efficiency_score = np.clip(
        (metrics["efficiency"] - 0.003) / (0.18 - 0.003), 0, 1
    )
    balance_score = np.clip(
        (metrics["min_to_max"] - 0.15) / (0.90 - 0.15), 0, 1
    )
    return 0.60 * uniform_score + 0.30 * efficiency_score + 0.10 * balance_score

# ----------------------------------------------------------------------
# Objective for DE (negative score, deterministic)
# ----------------------------------------------------------------------
def _objective(raw_vec: np.ndarray, problem: Dict[str, Any]) -> float:
    cfg = problem["cfg"]
    transitions = np.sort(raw_vec)
    field = build_incident_field(problem, transitions)
    focus = field.RS(z=cfg["focal"], new_field=True, verbose=False)
    intensity = np.abs(focus.u) ** 2
    metrics = evaluate_orders(problem, intensity, focus.x)
    return -_score_from_metrics(metrics)

# ----------------------------------------------------------------------
# Local refinement (deterministic hill‑climb)
# ----------------------------------------------------------------------
def _refine_transitions(
    transitions: np.ndarray,
    problem: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    cfg = problem["cfg"]
    n = cfg["num_transitions"]
    half_range = 0.45 * cfg["period_size"]
    step = cfg["refine_step_frac"] * cfg["period_size"]
    best_trans = np.copy(transitions)
    best_score = -_objective(best_trans, problem)

    for _ in range(cfg["refine_iters"]):
        idx = rng.integers(n)
        delta = (rng.random() - 0.5) * 2 * step
        candidate = np.copy(best_trans)
        candidate[idx] = np.clip(candidate[idx] + delta, -half_range, half_range)
        candidate = np.sort(candidate)
        cand_score = -_objective(candidate, problem)
        if cand_score > best_score:
            best_score = cand_score
            best_trans = candidate

    return best_trans

# ----------------------------------------------------------------------
# Run a single DE instance with a given seed
# ----------------------------------------------------------------------
def _run_de(problem: Dict[str, Any], seed: int) -> Tuple[np.ndarray, Dict[str, Any], float]:
    cfg = problem["cfg"]
    n = cfg["num_transitions"]
    half_range = 0.45 * cfg["period_size"]
    bounds = [(-half_range, half_range) for _ in range(n)]

    de_kwargs = {
        "bounds": bounds,
        "maxiter": cfg.get("de_maxiter", 30),
        "popsize": cfg.get("de_popsize", 12),
        "seed": seed,
        "polish": True,
        "disp": False,
        "strategy": "best1bin",
        "tol": 1e-6,
    }

    result = differential_evolution(_objective, args=(problem,), **de_kwargs)
    best_trans = np.sort(result.x)
    # deterministic refinement using a seed derived from the DE seed
    rng = np.random.default_rng(seed + 1000)
    best_trans = _refine_transitions(best_trans, problem, rng)

    field = build_incident_field(problem, best_trans)
    focus = field.RS(z=cfg["focal"], new_field=True, verbose=False)
    intensity = np.abs(focus.u) ** 2
    metrics = evaluate_orders(problem, intensity, focus.x)
    score = _score_from_metrics(metrics)
    return best_trans, metrics, score

# ----------------------------------------------------------------------
# Optimized solver (multiple deterministic DE runs)
# ----------------------------------------------------------------------
def solve_baseline(problem: Dict[str, Any]) -> Dict[str, Any]:
    cfg = problem["cfg"]
    runs = cfg.get("de_runs", 1)
    seeds = [cfg.get("de_seed", 0) + i for i in range(runs)]

    best_overall_score = -np.inf
    best_overall_trans = None
    best_overall_metrics = None
    timings: List[float] = []

    for seed in seeds:
        start = time.time()
        trans, metrics, score = _run_de(problem, seed)
        elapsed = time.time() - start
        timings.append(elapsed)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_trans = trans
            best_overall_metrics = metrics

    # Final field for the selected best solution
    field = build_incident_field(problem, best_overall_trans)  # type: ignore[arg-type]
    focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
    intensity = np.abs(focus_field.u) ** 2

    total_time = sum(timings)
    print(
        f"[Task03/DE] Ran {runs} deterministic DE runs (total {total_time:.2f}s), "
        f"best score={best_overall_score:.4f}"
    )

    return {
        "transitions": best_overall_trans,
        "x_focus": focus_field.x,
        "intensity_focus": intensity,
        "metrics": best_overall_metrics,
    }

# ----------------------------------------------------------------------
# Persistence utilities
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
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
    print(
        "[Task03/Baseline] cv_orders={:.6f}, efficiency={:.6f}, score={:.4f}".format(
            solution["metrics"]["cv_orders"],
            solution["metrics"]["efficiency"],
            _score_from_metrics(solution["metrics"]),
        )
    )


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
