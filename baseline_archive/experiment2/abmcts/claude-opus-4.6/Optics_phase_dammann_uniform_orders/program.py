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


def _compute_score(metrics):
    cv = metrics["cv_orders"]
    eff = metrics["efficiency"]
    mtm = metrics["min_to_max"]
    uniform_score = np.clip(1.0 - cv / 0.9, 0.0, 1.0)
    efficiency_score = np.clip((eff - 0.003) / (0.18 - 0.003), 0.0, 1.0)
    balance_score = np.clip((mtm - 0.15) / (0.90 - 0.15), 0.0, 1.0)
    return 0.60 * uniform_score + 0.30 * efficiency_score + 0.10 * balance_score


def _try_transitions(problem, transitions):
    """Helper: evaluate a set of transitions and return (score, transitions, metrics, intensity, x_focus)."""
    cfg = problem["cfg"]
    field = build_incident_field(problem, transitions)
    focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
    intensity = np.abs(focus_field.u) ** 2
    metrics = evaluate_orders(problem, intensity, focus_field.x)
    score = _compute_score(metrics)
    return score, transitions, metrics, intensity, focus_field.x


def _fast_evaluate(problem, transitions):
    """Fast evaluation returning just the score."""
    cfg = problem["cfg"]
    field = build_incident_field(problem, transitions)
    focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
    intensity = np.abs(focus_field.u) ** 2
    metrics = evaluate_orders(problem, intensity, focus_field.x)
    return _compute_score(metrics)


def _optimize_nelder_mead(problem, transitions_init, n_iter=200):
    """Nelder-Mead optimization on transition positions."""
    cfg = problem["cfg"]
    period = cfg["period_size"]
    n = len(transitions_init)
    
    def objective(x):
        trans = np.sort(x)
        trans = np.clip(trans, -period/2 * 0.99, period/2 * 0.99)
        return -_fast_evaluate(problem, trans)
    
    dim = n
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = transitions_init.copy()
    scores = np.zeros(dim + 1)
    scores[0] = objective(simplex[0])
    
    step = period * 0.015
    for i in range(dim):
        simplex[i + 1] = transitions_init.copy()
        simplex[i + 1][i] += step
        scores[i + 1] = objective(simplex[i + 1])
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for iteration in range(n_iter):
        order = np.argsort(scores)
        simplex = simplex[order]
        scores = scores[order]
        
        centroid = simplex[:-1].mean(axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < scores[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                scores[-1] = fe
            else:
                simplex[-1] = xr
                scores[-1] = fr
        elif fr < scores[-2]:
            simplex[-1] = xr
            scores[-1] = fr
        else:
            if fr < scores[-1]:
                xc = centroid + rho * (xr - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        scores[i] = objective(simplex[i])
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc < scores[-1]:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        scores[i] = objective(simplex[i])
        
        if np.std(scores) < 1e-10:
            break
    
    best_idx = np.argmin(scores)
    best_trans = np.sort(simplex[best_idx])
    best_trans = np.clip(best_trans, -period/2 * 0.99, period/2 * 0.99)
    return _try_transitions(problem, best_trans)


def _optimize_local(problem, transitions_init, n_iter=80):
    """Simple local optimization using coordinate-wise perturbation."""
    cfg = problem["cfg"]
    period = cfg["period_size"]
    
    best_score, best_trans, best_metrics, best_intensity, best_x = _try_transitions(problem, transitions_init)
    
    step = period * 0.02
    
    for iteration in range(n_iter):
        improved = False
        for i in range(len(best_trans)):
            for delta_mult in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
                trial = best_trans.copy()
                trial[i] += delta_mult * step
                trial = np.sort(trial)
                trial = np.clip(trial, -period/2 * 0.99, period/2 * 0.99)
                
                sc, tr, met, inten, xf = _try_transitions(problem, trial)
                if sc > best_score:
                    best_score = sc
                    best_trans = tr
                    best_metrics = met
                    best_intensity = inten
                    best_x = xf
                    improved = True
        
        step *= 0.82
        if step < period * 0.00005:
            break
    
    return best_score, best_trans, best_metrics, best_intensity, best_x


def _symmetric_transitions(half_norms, period):
    """Build symmetric transitions from normalized half-period positions."""
    half_pos = np.array(half_norms) * period
    transitions = np.concatenate([-half_pos[::-1], half_pos])
    return np.sort(transitions)


def _optimize_scipy_de(problem, n_half=5, maxiter=80, popsize=12):
    """Use scipy differential evolution for optimization."""
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return None
    
    cfg = problem["cfg"]
    period = cfg["period_size"]
    
    def objective(x):
        # x is normalized half-period positions in [0, 0.5]
        half_pos = np.sort(x) * period
        transitions = np.concatenate([-half_pos[::-1], half_pos])
        transitions = np.sort(transitions)
        try:
            score = _fast_evaluate(problem, transitions)
            return -score
        except Exception:
            return 0.0
    
    bounds = [(0.01, 0.49)] * n_half
    
    result = differential_evolution(
        objective, bounds, maxiter=maxiter, popsize=popsize,
        tol=1e-8, seed=42, mutation=(0.5, 1.5), recombination=0.9,
        polish=False
    )
    
    half_pos = np.sort(result.x) * period
    transitions = np.concatenate([-half_pos[::-1], half_pos])
    transitions = np.sort(transitions)
    return _try_transitions(problem, transitions)


def _optimize_scipy_de_asym(problem, n_trans=10, maxiter=60, popsize=10):
    """Use scipy DE for asymmetric transitions."""
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return None
    
    cfg = problem["cfg"]
    period = cfg["period_size"]
    
    def objective(x):
        trans = np.sort(x) * period - period/2
        trans = np.clip(trans, -period/2 * 0.99, period/2 * 0.99)
        try:
            score = _fast_evaluate(problem, trans)
            return -score
        except Exception:
            return 0.0
    
    bounds = [(0.01, 0.99)] * n_trans
    
    result = differential_evolution(
        objective, bounds, maxiter=maxiter, popsize=popsize,
        tol=1e-8, seed=42, mutation=(0.5, 1.5), recombination=0.9,
        polish=False
    )
    
    trans = np.sort(result.x) * period - period/2
    trans = np.clip(trans, -period/2 * 0.99, period/2 * 0.99)
    return _try_transitions(problem, trans)


def baseline_transitions(problem: Dict[str, Any]) -> np.ndarray:
    cfg = problem["cfg"]
    transitions = np.linspace(-0.45 * cfg["period_size"], 0.45 * cfg["period_size"], cfg["num_transitions"])
    return transitions


def solve_baseline(problem: Dict[str, Any]) -> Dict[str, Any]:
    cfg = problem["cfg"]
    period = cfg["period_size"]
    
    best_score = -1.0
    best_result = None
    
    def update_best(sc, tr, met, inten, xf):
        nonlocal best_score, best_result
        if sc > best_score:
            best_score = sc
            best_result = (tr, met, inten, xf)
    
    # Literature-based Dammann transitions for 7 orders (symmetric)
    lit_half_norms = [
        [0.0336, 0.2580, 0.3376, 0.4530, 0.4724],
        [0.0371, 0.2316, 0.3445, 0.4587, 0.4802],
        [0.0300, 0.2500, 0.3300, 0.4500, 0.4700],
        [0.0400, 0.2650, 0.3400, 0.4600, 0.4750],
        [0.0350, 0.2200, 0.3500, 0.4550, 0.4780],
        [0.0320, 0.2450, 0.3350, 0.4480, 0.4710],
        [0.0380, 0.2400, 0.3420, 0.4560, 0.4790],
        [0.0310, 0.2550, 0.3360, 0.4510, 0.4715],
        [0.0345, 0.2480, 0.3390, 0.4540, 0.4730],
        [0.0360, 0.2350, 0.3460, 0.4570, 0.4760],
    ]
    
    candidates = []
    for norms in lit_half_norms:
        candidates.append(_symmetric_transitions(norms, period))
    
    # Different numbers of half-transitions
    for n_half in [3, 4, 5, 6, 7]:
        for offset in np.linspace(0.02, 0.08, 5):
            half_pos = np.linspace(offset * period, 0.48 * period, n_half)
            transitions = np.concatenate([-half_pos[::-1], half_pos])
            candidates.append(np.sort(transitions))
    
    # Evaluate all candidates quickly
    scored_candidates = []
    for trans in candidates:
        try:
            sc, tr, met, inten, xf = _try_transitions(problem, trans)
            scored_candidates.append((sc, tr, met, inten, xf))
            update_best(sc, tr, met, inten, xf)
        except Exception:
            continue
    
    # Sort by score, take top candidates for optimization
    scored_candidates.sort(key=lambda x: -x[0])
    top_n = min(3, len(scored_candidates))
    
    # Nelder-Mead on top candidates
    for i in range(top_n):
        try:
            sc, tr, met, inten, xf = _optimize_nelder_mead(problem, scored_candidates[i][1], n_iter=150)
            update_best(sc, tr, met, inten, xf)
            sc2, tr2, met2, inten2, xf2 = _optimize_local(problem, tr, n_iter=30)
            update_best(sc2, tr2, met2, inten2, xf2)
        except Exception:
            continue
    
    # Scipy DE optimization (symmetric)
    for n_half in [5, 6, 7, 4]:
        try:
            result = _optimize_scipy_de(problem, n_half=n_half, maxiter=60, popsize=10)
            if result is not None:
                sc, tr, met, inten, xf = result
                update_best(sc, tr, met, inten, xf)
                # Refine best DE result
                sc2, tr2, met2, inten2, xf2 = _optimize_local(problem, tr, n_iter=20)
                update_best(sc2, tr2, met2, inten2, xf2)
        except Exception:
            continue
    
    # Scipy DE asymmetric
    for n_trans in [10, 12, 14]:
        try:
            result = _optimize_scipy_de_asym(problem, n_trans=n_trans, maxiter=50, popsize=8)
            if result is not None:
                sc, tr, met, inten, xf = result
                update_best(sc, tr, met, inten, xf)
                sc2, tr2, met2, inten2, xf2 = _optimize_local(problem, tr, n_iter=20)
                update_best(sc2, tr2, met2, inten2, xf2)
        except Exception:
            continue
    
    if best_result is not None:
        transitions, metrics, intensity, x_focus = best_result
    else:
        transitions = baseline_transitions(problem)
        field = build_incident_field(problem, transitions)
        focus_field = field.RS(z=cfg["focal"], new_field=True, verbose=False)
        intensity = np.abs(focus_field.u) ** 2
        metrics = evaluate_orders(problem, intensity, focus_field.x)
        x_focus = focus_field.x

    return {
        "transitions": transitions,
        "x_focus": x_focus,
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
