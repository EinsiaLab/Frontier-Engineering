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


def _fast_order_energies(transitions_sorted, period_size, orders):
    """
    Compute Fourier coefficients of a binary phase grating analytically.

    For a binary phase grating with phase values 0 and pi, the transmission is
    t(x) = exp(i*pi*u(x)) where u(x) switches between 0 and 1.
    This means t(x) = +1 or -1.

    The Fourier coefficient for order m is:
        c_m = (1/T) * integral_0^T t(x) * exp(-i*2*pi*m*x/T) dx

    For segments where t=+1 and t=-1, we can compute this analytically.
    """
    T = period_size
    half_T = T / 2.0

    # Build segment boundaries from -T/2 to T/2
    # transitions are positions where the phase flips
    # Start with phase "down" (= 0, so t = exp(0) = +1... wait,
    # the code does exp(i*pi*u), and start="down" means u starts at 0)
    # Actually: start="down" -> u starts at 0 -> t = exp(0) = 1
    # At first transition, u -> 1 -> t = exp(i*pi) = -1
    # etc.

    # Segments: from -T/2 to trans[0], trans[0] to trans[1], ..., trans[-1] to T/2
    edges = np.concatenate([[-half_T], transitions_sorted, [half_T]])

    n_orders = len(orders)
    energies = np.zeros(n_orders)

    for oi, m in enumerate(orders):
        c_m = 0.0 + 0.0j
        sign = 1.0  # start="down" -> phase=0 -> t=+1

        if m == 0:
            # c_0 = (1/T) * sum of sign_k * (edge_{k+1} - edge_k)
            for k in range(len(edges) - 1):
                c_m += sign * (edges[k+1] - edges[k])
                sign *= -1
            c_m /= T
        else:
            # c_m = (1/T) * sum_k sign_k * integral_{edges[k]}^{edges[k+1]} exp(-i*2*pi*m*x/T) dx
            # = (1/T) * sum_k sign_k * [T/(-i*2*pi*m)] * [exp(-i*2*pi*m*edges[k+1]/T) - exp(-i*2*pi*m*edges[k]/T)]
            coeff = 1.0 / (-1j * 2 * np.pi * m)
            for k in range(len(edges) - 1):
                phase_end = np.exp(-1j * 2 * np.pi * m * edges[k+1] / T)
                phase_start = np.exp(-1j * 2 * np.pi * m * edges[k] / T)
                c_m += sign * coeff * (phase_end - phase_start)
                sign *= -1

        energies[oi] = np.abs(c_m) ** 2

    return energies


def _fast_order_energies_vectorized(transitions_sorted, period_size, orders):
    """Fully vectorized: compute all orders at once."""
    T = period_size
    half_T = T / 2.0

    edges = np.concatenate([[-half_T], transitions_sorted, [half_T]])
    n_seg = len(edges) - 1

    # signs: +1, -1, +1, -1, ... (start="down")
    signs = np.ones(n_seg)
    signs[1::2] = -1.0

    seg_lengths = np.diff(edges)

    orders_arr = np.asarray(orders, dtype=float)
    energies = np.zeros(len(orders_arr))

    # Vectorize over all orders at once
    # For m=0
    c0 = np.sum(signs * seg_lengths) / T

    for oi, m in enumerate(orders_arr):
        if m == 0:
            energies[oi] = np.abs(c0) ** 2
        else:
            coeff = 1.0 / (-1j * 2 * np.pi * m)
            phase_ends = np.exp(-1j * 2 * np.pi * m * edges[1:] / T)
            phase_starts = np.exp(-1j * 2 * np.pi * m * edges[:-1] / T)
            c_m = np.sum(signs * coeff * (phase_ends - phase_starts))
            energies[oi] = np.abs(c_m) ** 2

    return energies


def _objective(params, period_size, orders, n_transitions):
    """Objective for DE: minimize to get uniform, efficient orders."""
    half_T = period_size / 2.0
    transitions = np.sort(params) * period_size - half_T

    energies = _fast_order_energies_vectorized(transitions, period_size, orders)

    mean_e = energies.mean()
    if mean_e < 1e-15:
        return 1e6

    cv = energies.std() / (mean_e + 1e-15)
    efficiency = energies.sum()
    min_to_max = energies.min() / (energies.max() + 1e-15)

    # Score components (matching the evaluation formula)
    uniform_score = max(0.0, min(1.0, 1.0 - cv / 0.9))
    efficiency_score = max(0.0, min(1.0, (efficiency - 0.003) / (0.18 - 0.003)))
    balance_score = max(0.0, min(1.0, (min_to_max - 0.15) / (0.90 - 0.15)))

    score = 0.60 * uniform_score + 0.30 * efficiency_score + 0.10 * balance_score

    return -score


def _objective_symmetric(half_params, period_size, orders, n_half):
    """Objective using anti-symmetric parameterization.

    For a Dammann grating producing symmetric orders, we can use
    anti-symmetric transitions: if t is a transition in [0, T/2],
    then -t is also a transition (with opposite phase sense).
    This halves the number of free parameters.
    """
    half_T = period_size / 2.0
    # half_params are in (0, 0.5) normalized, representing transitions in [0, T/2)
    half_sorted = np.sort(half_params) * period_size  # in [0, T]
    # Create symmetric transitions: -t and +t for each
    # But we need them relative to center, so map to [-T/2, T/2]
    pos_trans = half_sorted - half_T  # in [-T/2, T/2] but only positive half
    # Actually let's keep it simple: half_params in [0, 0.5] -> positions in [0, T/2]
    pos_positions = np.sort(half_params) * period_size  # [0, T]
    # Map to [-T/2, T/2]
    pos_positions = pos_positions - half_T

    # For anti-symmetric: transitions at +x and -x
    neg_positions = -pos_positions[::-1]
    all_transitions = np.sort(np.concatenate([neg_positions, pos_positions]))

    energies = _fast_order_energies_vectorized(all_transitions, period_size, orders)

    mean_e = energies.mean()
    if mean_e < 1e-15:
        return 1e6

    cv = energies.std() / (mean_e + 1e-15)
    efficiency = energies.sum()
    min_to_max = energies.min() / (energies.max() + 1e-15)

    uniform_score = max(0.0, min(1.0, 1.0 - cv / 0.9))
    efficiency_score = max(0.0, min(1.0, (efficiency - 0.003) / (0.18 - 0.003)))
    balance_score = max(0.0, min(1.0, (min_to_max - 0.15) / (0.90 - 0.15)))

    score = 0.60 * uniform_score + 0.30 * efficiency_score + 0.10 * balance_score

    return -score


def _simulate_and_score(problem, transitions):
    """Run actual simulation and return negative score."""
    try:
        field = build_incident_field(problem, transitions)
        focus_field = field.RS(z=problem["cfg"]["focal"], new_field=True, verbose=False)
        intensity = np.abs(focus_field.u) ** 2
        metrics = evaluate_orders(problem, intensity, focus_field.x)

        cv = metrics["cv_orders"]
        efficiency = metrics["efficiency"]
        min_to_max = metrics["min_to_max"]

        uniform_score = max(0.0, min(1.0, 1.0 - cv / 0.9))
        efficiency_score = max(0.0, min(1.0, (efficiency - 0.003) / (0.18 - 0.003)))
        balance_score = max(0.0, min(1.0, (min_to_max - 0.15) / (0.90 - 0.15)))

        score = 0.60 * uniform_score + 0.30 * efficiency_score + 0.10 * balance_score
        return -score, metrics
    except Exception:
        return 0.0, None


def _sim_objective(params_normalized, problem, period_size):
    """Objective using actual simulation. Expensive but accurate."""
    half_T = period_size / 2.0
    transitions = np.sort(params_normalized) * period_size - half_T
    score, _ = _simulate_and_score(problem, transitions)
    return score


def _sim_objective_symmetric(half_params, problem, period_size, n_half):
    """Symmetric simulation-based objective."""
    half_T = period_size / 2.0
    half_sorted = np.sort(half_params) * period_size - half_T
    neg_positions = -half_sorted[::-1]
    transitions = np.sort(np.concatenate([neg_positions, half_sorted]))
    score, _ = _simulate_and_score(problem, transitions)
    return score


def baseline_transitions(problem: Dict[str, Any]) -> np.ndarray:
    cfg = problem["cfg"]
    period_size = cfg["period_size"]
    n_transitions = cfg["num_transitions"]
    orders = np.arange(cfg["order_min"], cfg["order_max"] + 1, dtype=int)
    half_T = period_size / 2.0
    n_half = n_transitions // 2

    # Known Dammann grating literature transition positions for 7 equal orders
    lit_half_transitions_7 = np.array([
        0.0576, 0.1601, 0.2356, 0.3392, 0.4053, 0.4389, 0.4658
    ])
    pos_trans = lit_half_transitions_7 * period_size
    neg_trans = -pos_trans[::-1]
    lit_transitions_full = np.sort(np.concatenate([neg_trans, pos_trans]))

    lit_half_2 = np.array([0.0380, 0.1180, 0.2130, 0.3260, 0.3960, 0.4340, 0.4720])
    pos_trans_2 = lit_half_2 * period_size
    neg_trans_2 = -pos_trans_2[::-1]
    lit_transitions_2 = np.sort(np.concatenate([neg_trans_2, pos_trans_2]))

    lit_half_3 = np.array([0.0469, 0.1406, 0.2344, 0.3281, 0.4063, 0.4375, 0.4688])
    pos_trans_3 = lit_half_3 * period_size
    neg_trans_3 = -pos_trans_3[::-1]
    lit_transitions_3 = np.sort(np.concatenate([neg_trans_3, pos_trans_3]))

    candidates = []

    try:
        from scipy.optimize import differential_evolution, minimize

        # --- Phase 1: Fast analytical optimization to find good starting points ---

        # Strategy 1: One focused full DE run with larger population
        bounds_full = [(0.01, 0.99)] * n_transitions

        # Create initial population seeded with literature values
        rng = np.random.RandomState(42)
        init_pop_size = 30
        init_pop = rng.uniform(0.01, 0.99, size=(init_pop_size, n_transitions))
        # Inject literature solutions into initial population
        for idx, lit_trans in enumerate([lit_transitions_full, lit_transitions_2, lit_transitions_3]):
            if len(lit_trans) == n_transitions and idx < init_pop_size:
                init_pop[idx] = np.clip((lit_trans + half_T) / period_size, 0.01, 0.99)

        result = differential_evolution(
            _objective,
            bounds=bounds_full,
            args=(period_size, orders, n_transitions),
            seed=42,
            maxiter=400,
            popsize=30,
            init=init_pop,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,
            workers=1,
        )
        trans = np.sort(result.x) * period_size - half_T
        candidates.append((result.fun, trans))

        # Strategy 2: Symmetric DE optimization (fewer params, more focused)
        bounds_half = [(0.01, 0.49)] * n_half

        # Create initial population with literature half-transitions
        init_pop_half_size = 30
        init_pop_half = rng.uniform(0.01, 0.49, size=(init_pop_half_size, n_half))
        for idx, lit_half in enumerate([lit_half_transitions_7, lit_half_2, lit_half_3]):
            if len(lit_half) == n_half and idx < init_pop_half_size:
                init_pop_half[idx] = np.clip(lit_half, 0.01, 0.49)

        result2 = differential_evolution(
            _objective_symmetric,
            bounds=bounds_half,
            args=(period_size, orders, n_half),
            seed=42,
            maxiter=600,
            popsize=30,
            init=init_pop_half,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,
            workers=1,
        )
        half_sorted = np.sort(result2.x) * period_size - half_T
        neg_positions = -half_sorted[::-1]
        trans2 = np.sort(np.concatenate([neg_positions, half_sorted]))
        candidates.append((result2.fun, trans2))

        # Strategy 3: Local optimization from literature starting points
        for lit_trans in [lit_transitions_full, lit_transitions_2, lit_transitions_3]:
            if len(lit_trans) == n_transitions:
                x0 = (lit_trans + half_T) / period_size
                x0 = np.clip(x0, 0.01, 0.99)

                res = minimize(
                    _objective,
                    x0,
                    args=(period_size, orders, n_transitions),
                    method='Nelder-Mead',
                    options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-13},
                )
                trans_nm = np.sort(res.x) * period_size - half_T
                candidates.append((res.fun, trans_nm))

        # Add raw literature candidates
        for lit_trans in [lit_transitions_full, lit_transitions_2, lit_transitions_3]:
            if len(lit_trans) == n_transitions:
                score = _objective((lit_trans + half_T) / period_size, period_size, orders, n_transitions)
                candidates.append((score, lit_trans))

    except ImportError:
        pass

    if not candidates:
        return np.linspace(-0.45 * period_size, 0.45 * period_size, n_transitions)

    # --- Phase 2: Evaluate top analytical candidates with actual simulation ---
    candidates.sort(key=lambda x: x[0])
    top_k = min(8, len(candidates))

    sim_candidates = []  # (real_score, transitions, metrics)
    for i in range(top_k):
        _, trans = candidates[i]
        real_score, metrics = _simulate_and_score(problem, trans)
        sim_candidates.append((real_score, trans, metrics))

    # --- Phase 3: Simulation-based local refinement on best candidates ---
    sim_candidates.sort(key=lambda x: x[0])

    try:
        from scipy.optimize import minimize as sp_minimize

        # Refine top 2 candidates using actual simulation
        n_refine = min(2, len(sim_candidates))
        for i in range(n_refine):
            _, trans, _ = sim_candidates[i]
            x0 = (trans + half_T) / period_size
            x0 = np.clip(x0, 0.01, 0.99)

            # Use Nelder-Mead with simulation objective (expensive but accurate)
            res = sp_minimize(
                _sim_objective,
                x0,
                args=(problem, period_size),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True,
                },
            )
            refined_trans = np.sort(res.x) * period_size - half_T
            real_score, metrics = _simulate_and_score(problem, refined_trans)
            sim_candidates.append((real_score, refined_trans, metrics))

        # Also try symmetric refinement on best
        sim_candidates.sort(key=lambda x: x[0])
        best_trans = sim_candidates[0][1]
        pos_half = best_trans[n_half:]
        half_x0 = (pos_half + half_T) / period_size
        half_x0 = np.clip(half_x0, 0.01, 0.49)

        res_sym = sp_minimize(
            _sim_objective_symmetric,
            half_x0,
            args=(problem, period_size, n_half),
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True,
            },
        )
        half_sorted = np.sort(res_sym.x) * period_size - half_T
        neg_positions = -half_sorted[::-1]
        refined_sym = np.sort(np.concatenate([neg_positions, half_sorted]))
        real_score, metrics = _simulate_and_score(problem, refined_sym)
        sim_candidates.append((real_score, refined_sym, metrics))

        # Multiple rounds of alternating Powell and Nelder-Mead refinement
        for refine_round in range(8):
            sim_candidates.sort(key=lambda x: x[0])
            best_cur = sim_candidates[0][1]
            x0_cur = (best_cur + half_T) / period_size
            x0_cur = np.clip(x0_cur, 0.01, 0.99)

            if refine_round % 3 == 0:
                # Powell round
                res_r = sp_minimize(
                    _sim_objective,
                    x0_cur,
                    args=(problem, period_size),
                    method='Powell',
                    options={
                        'maxiter': 800,
                        'ftol': 1e-15,
                        'xtol': 1e-15,
                    },
                )
                refined_r = np.sort(res_r.x) * period_size - half_T
            elif refine_round % 3 == 1:
                # Nelder-Mead round
                res_r = sp_minimize(
                    _sim_objective,
                    x0_cur,
                    args=(problem, period_size),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 800,
                        'xatol': 1e-11,
                        'fatol': 1e-11,
                        'adaptive': True,
                    },
                )
                refined_r = np.sort(res_r.x) * period_size - half_T
            else:
                # Symmetric Nelder-Mead round
                pos_half_r = best_cur[n_half:]
                half_x0_r = (pos_half_r + half_T) / period_size
                half_x0_r = np.clip(half_x0_r, 0.01, 0.49)
                res_r = sp_minimize(
                    _sim_objective_symmetric,
                    half_x0_r,
                    args=(problem, period_size, n_half),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 600,
                        'xatol': 1e-11,
                        'fatol': 1e-11,
                        'adaptive': True,
                    },
                )
                half_sorted_r = np.sort(res_r.x) * period_size - half_T
                neg_positions_r = -half_sorted_r[::-1]
                refined_r = np.sort(np.concatenate([neg_positions_r, half_sorted_r]))

            real_score_r, metrics_r = _simulate_and_score(problem, refined_r)
            sim_candidates.append((real_score_r, refined_r, metrics_r))

        # Perturbation search around best solution
        sim_candidates.sort(key=lambda x: x[0])
        best_perturb = sim_candidates[0][1]
        best_score_so_far = sim_candidates[0][0]
        rng_perturb = np.random.RandomState(123)
        for trial in range(8):
            scale = 0.003 * period_size * (0.5 ** (trial // 3))
            perturb = rng_perturb.normal(0, scale, size=n_half)
            pos_half_p = best_perturb[n_half:] + perturb
            pos_half_p = np.sort(np.clip(pos_half_p, -half_T + 0.001 * period_size, half_T - 0.001 * period_size))
            neg_half_p = -pos_half_p[::-1]
            trans_p = np.sort(np.concatenate([neg_half_p, pos_half_p]))
            score_p, metrics_p = _simulate_and_score(problem, trans_p)
            sim_candidates.append((score_p, trans_p, metrics_p))
            if score_p < best_score_so_far:
                best_score_so_far = score_p
                best_perturb = trans_p
                # Refine this perturbation with Powell
                x0_p = (trans_p + half_T) / period_size
                x0_p = np.clip(x0_p, 0.01, 0.99)
                res_p = sp_minimize(
                    _sim_objective,
                    x0_p,
                    args=(problem, period_size),
                    method='Powell',
                    options={'maxiter': 600, 'ftol': 1e-15, 'xtol': 1e-15},
                )
                refined_p = np.sort(res_p.x) * period_size - half_T
                score_rp, metrics_rp = _simulate_and_score(problem, refined_p)
                sim_candidates.append((score_rp, refined_p, metrics_rp))
                if score_rp < best_score_so_far:
                    best_score_so_far = score_rp
                    best_perturb = refined_p

        # Final deep polish on absolute best
        sim_candidates.sort(key=lambda x: x[0])
        best_final = sim_candidates[0][1]
        x0_final = (best_final + half_T) / period_size
        x0_final = np.clip(x0_final, 0.01, 0.99)
        res_final = sp_minimize(
            _sim_objective,
            x0_final,
            args=(problem, period_size),
            method='Powell',
            options={'maxiter': 1500, 'ftol': 1e-16, 'xtol': 1e-16},
        )
        refined_final = np.sort(res_final.x) * period_size - half_T
        real_score_f, metrics_f = _simulate_and_score(problem, refined_final)
        sim_candidates.append((real_score_f, refined_final, metrics_f))

        # One last Nelder-Mead with very tight tolerance
        sim_candidates.sort(key=lambda x: x[0])
        best_last = sim_candidates[0][1]
        x0_last = (best_last + half_T) / period_size
        x0_last = np.clip(x0_last, 0.01, 0.99)
        res_last = sp_minimize(
            _sim_objective,
            x0_last,
            args=(problem, period_size),
            method='Nelder-Mead',
            options={
                'maxiter': 1000,
                'xatol': 1e-12,
                'fatol': 1e-12,
                'adaptive': True,
            },
        )
        refined_last = np.sort(res_last.x) * period_size - half_T
        real_score_l, metrics_l = _simulate_and_score(problem, refined_last)
        sim_candidates.append((real_score_l, refined_last, metrics_l))

    except Exception:
        pass

    # Select the best based on actual simulation score
    sim_candidates.sort(key=lambda x: x[0])
    best_transitions = sim_candidates[0][1]

    return best_transitions


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