#!/usr/bin/env python
# EVOLVE-BLOCK-START
"""Baseline solver for Task 02: hard Fourier pattern holography."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np


DEFAULT_CONFIG: Dict[str, Any] = {
    "slm_pixels": 128,
    "aperture_radius_px": 56,
    "seed": 0,
}


def circular_aperture(n: int, radius_px: float) -> np.ndarray:
    y, x = np.indices((n, n))
    c = (n - 1) / 2.0
    return (((x - c) ** 2 + (y - c) ** 2) <= radius_px**2).astype(float)


def build_target_pattern(n: int) -> np.ndarray:
    y, x = np.indices((n, n))
    c = (n - 1) / 2.0

    target = np.zeros((n, n), dtype=float)

    xs = np.linspace(18, 110, 8)
    ys = np.linspace(18, 110, 8)
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            amp = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(0.7 * i + 0.9 * j))
            if (i + j) % 2 == 0:
                amp *= 0.4
            target += amp * np.exp(-((x - xx) ** 2 + (y - yy) ** 2) / (2.0 * 0.9**2))

    for xx in range(20, 108):
        yy = int(64 + 18 * np.sin((xx - 20) / 13.0))
        target[max(0, yy - 1):min(n, yy + 2), max(0, xx - 1):min(n, xx + 2)] += 0.35

    dark_zone = (np.abs(x - c) < 4) & (np.abs(y - c) < 45)
    target[dark_zone] = 0.0

    target = np.clip(target, 0.0, None)
    target = target / (target.max() + 1e-12)
    return target


def build_problem(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    n = int(cfg["slm_pixels"])
    x = np.arange(n, dtype=float)
    y = np.arange(n, dtype=float)

    aperture_amp = circular_aperture(n, float(cfg["aperture_radius_px"]))
    target_amp = build_target_pattern(n)

    return {
        "cfg": cfg,
        "x": x,
        "y": y,
        "aperture_amp": aperture_amp,
        "target_amp": target_amp,
    }


def _score_phase_fast(aperture, phase, bright_mask, dark_mask, T, T_sum, T_mean):
    """Fast score computation with precomputed constants."""
    near = aperture * np.exp(1j * phase)
    far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
    I = np.abs(far) ** 2

    I_sum = I.sum() + 1e-30
    I_norm = I / I_sum * T_sum

    nmse = np.sqrt(np.mean((I_norm - T) ** 2)) / (T_mean + 1e-12)

    energy_in_target = I[bright_mask].sum() / I_sum
    leak = I[dark_mask].sum() / I_sum
    dark_suppression = 1.0 - leak

    pattern_score = np.clip(1.0 - nmse / 4.0, 0.0, 1.0)
    energy_score = np.clip((energy_in_target - 0.10) / 0.60, 0.0, 1.0)
    dark_score = np.clip((dark_suppression - 0.35) / 0.55, 0.0, 1.0)

    return 100.0 * (0.55 * pattern_score + 0.30 * energy_score + 0.15 * dark_score)


def _compute_loss_and_gradient(aperture, phase, target_amp, T, T_sum, T_mean,
                                bright_mask, dark_mask, aperture_mask,
                                w_pattern=0.55, w_energy=0.30, w_dark=0.15):
    """Compute a differentiable loss and its gradient w.r.t. phase."""
    near = aperture * np.exp(1j * phase)
    far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
    I = np.abs(far) ** 2

    I_sum = I.sum() + 1e-30

    # Pattern loss: MSE between normalized intensities
    scale_norm = T_sum / I_sum
    I_norm = I * scale_norm
    diff = I_norm - T
    pattern_loss = np.mean(diff ** 2)

    # Use RMSE-aligned gradient: d/dI (sqrt(mean(diff^2))) = diff * scale / (N2 * rmse)
    rmse = np.sqrt(pattern_loss) + 1e-12

    # Energy loss: want to maximize energy in bright regions
    energy_ratio = I[bright_mask].sum() / I_sum

    # Dark loss: want to minimize energy in dark regions
    dark_ratio = I[dark_mask].sum() / I_sum

    N2 = float(I.size)

    dL_dI = np.zeros_like(I)

    # Pattern gradient - aligned with RMSE/4 scoring
    # d(nmse)/dI = d(rmse/T_mean)/dI = (1/T_mean) * diff*scale_norm / (N2*rmse)
    # d(pattern_score)/dI = -1/4 * d(nmse)/dI
    # We want to minimize loss = -pattern_score, so gradient is +1/4 * d(nmse)/dI
    dL_dI += w_pattern * (1.0 / (4.0 * T_mean)) * diff * scale_norm / (N2 * rmse)

    # Energy gradient - aligned with (energy - 0.10) / 0.60 scoring
    energy_grad = np.zeros_like(I)
    energy_grad[bright_mask] = -1.0 / I_sum
    energy_grad += energy_ratio / I_sum
    dL_dI += w_energy * (1.0 / 0.60) * energy_grad

    # Dark gradient - aligned with (dark_sup - 0.35) / 0.55 scoring
    dark_grad = np.zeros_like(I)
    dark_grad[dark_mask] = 1.0 / I_sum
    dark_grad -= dark_ratio / I_sum
    dL_dI += w_dark * (1.0 / 0.55) * dark_grad

    # Backprop through I = |far|^2: dL/dfar = 2 * conj(far) * dL/dI
    # Actually for real-valued loss L depending on I=|far|^2:
    # dL/d(far_real) = 2*far_real*dL/dI, dL/d(far_imag) = 2*far_imag*dL/dI
    # So dL/dfar_conj = far * dL/dI (Wirtinger derivative)
    # For ifft2 backprop we need: dL_dfar as complex gradient
    dL_dfar = 2.0 * far * dL_dI

    # Backprop through FFT: ifftshift -> fft2 -> fftshift
    # Adjoint is: ifftshift -> ifft2 -> fftshift
    dL_dnear = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dL_dfar), norm="ortho"))

    # d(near)/d(phase) = 1j * near => dL/dphase = Re(-1j * dL_dnear * conj(near))
    dL_dphase = np.real(-1j * dL_dnear * np.conj(near))
    dL_dphase[~aperture_mask] = 0.0

    return dL_dphase


def solve_baseline(problem: Dict[str, Any], seed: int | None = None) -> np.ndarray:
    """Multi-restart WGS-Kim with gradient refinement and SPSA polish."""
    seed_value = int(problem["cfg"]["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed_value)

    target_amp = problem["target_amp"].copy()
    aperture = problem["aperture_amp"]
    n = target_amp.shape[0]

    # Precompute masks and constants
    bright_mask = target_amp > 0.30
    dark_mask = target_amp < 0.03
    signal_mask = target_amp > 0.01
    aperture_mask = aperture > 0.5

    # Also create a "transition" mask for regions near dark zones
    from scipy.ndimage import binary_dilation
    dark_border_mask = binary_dilation(dark_mask, iterations=2) & (~dark_mask) & (~bright_mask)

    T = target_amp ** 2
    T_sum = T.sum() + 1e-30
    T_mean = T.mean() + 1e-12

    def score_fn(phase):
        return _score_phase_fast(aperture, phase, bright_mask, dark_mask, T, T_sum, T_mean)

    def fast_forward(phase):
        """Compute far-field intensity quickly."""
        near = aperture * np.exp(1j * phase)
        far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
        return np.abs(far) ** 2

    # Also create wider border mask for more aggressive suppression
    dark_border_mask_wide = binary_dilation(dark_mask, iterations=3) & (~dark_mask) & (~bright_mask)

    # ========== Stage 1: Multi-restart WGS with dark-zone energy redistribution ==========
    num_restarts = 16
    num_wgs_iterations = 1500
    wgs_start = 3

    best_phase = None
    best_score = -1e30
    # Keep top-k phases for later refinement
    top_k = 5
    top_phases = []
    top_scores = []

    # Inter-restart learning: accumulate systematic errors across restarts
    global_error_accum = np.zeros_like(target_amp)
    global_error_count = 0
    effective_target = target_amp.copy()

    for restart in range(num_restarts):
        # Update effective target based on accumulated cross-restart error
        if global_error_count >= 2:
            correction = global_error_accum / global_error_count
            # Only correct in signal regions, gently
            correction_strength = min(0.12, 0.04 * global_error_count)
            effective_target = target_amp + correction_strength * correction * signal_mask
            effective_target = np.clip(effective_target, 0.0, None)
            # Re-normalize
            eff_max = effective_target.max() + 1e-12
            if eff_max > 1.2:
                effective_target = effective_target / eff_max
            effective_target[dark_mask] = 0.0

        random_phase = 2.0 * np.pi * rng.random((n, n))
        Uz = effective_target * np.exp(1j * random_phase)
        slm_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Uz), norm="ortho"))
        slm_phase = np.angle(slm_field)

        weights = np.ones_like(target_amp)

        # Track best within this restart
        restart_best_phase = slm_phase.copy()
        restart_best_score = -1e30

        for iteration in range(num_wgs_iterations):
            near_field = aperture * np.exp(1j * slm_phase)
            far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near_field), norm="ortho"))
            far_amp = np.abs(far_field)
            far_phase = np.angle(far_field)

            # WGS-Kim weight update
            if iteration >= wgs_start:
                far_amp_safe = np.maximum(far_amp, 1e-12)
                ratio = effective_target / far_amp_safe
                ratio = np.clip(ratio, 0.02, 50.0)

                # Smoother adaptive mixing parameter
                if iteration < 6:
                    alpha = 0.10
                elif iteration < 20:
                    alpha = 0.20
                elif iteration < 50:
                    alpha = 0.35
                elif iteration < 120:
                    alpha = 0.50
                elif iteration < 250:
                    alpha = 0.62
                elif iteration < 500:
                    alpha = 0.75
                elif iteration < 800:
                    alpha = 0.85
                elif iteration < 1200:
                    alpha = 0.92
                elif iteration < 1600:
                    alpha = 0.96
                else:
                    alpha = 0.98

                weights = weights * (alpha + (1.0 - alpha) * ratio)

                if signal_mask.any():
                    w_mean = weights[signal_mask].mean() + 1e-12
                    weights = weights / w_mean

                # Aggressive dark zone suppression
                if iteration < 15:
                    dark_factor = max(0.0, 0.015 * (1.0 - iteration / 15.0))
                else:
                    dark_factor = 0.0
                weights[dark_mask] = dark_factor

                # Suppress transition zones near dark areas
                if iteration > 20:
                    weights[dark_border_mask] *= 0.90
                if iteration > 50:
                    weights[dark_border_mask_wide] *= 0.95

                # Extra boost for bright regions that are underrepresented
                if iteration > 30 and iteration % 10 == 0:
                    bright_ratio = far_amp[bright_mask].mean() / (effective_target[bright_mask].mean() + 1e-12)
                    if bright_ratio < 0.90:
                        weights[bright_mask] *= 1.06
                    elif bright_ratio < 0.95:
                        weights[bright_mask] *= 1.03
                    if iteration > 60:
                        leak_energy = far_amp[dark_mask].sum()
                        total_energy = far_amp.sum() + 1e-12
                        if leak_energy / total_energy > 0.10:
                            weights[bright_mask] *= 1.04
                            weights[dark_border_mask] *= 0.88

            weighted_target = weights * effective_target

            # Hard zero dark zones after brief warmup
            if iteration > 3:
                weighted_target[dark_mask] = 0.0

            # Dark zone energy redistribution: take leaked energy and add to bright regions
            if iteration > 10:
                dark_leaked_amp = far_amp[dark_mask].sum()
                if dark_leaked_amp > 0 and bright_mask.any():
                    # Redistribute a fraction of dark zone energy to bright regions
                    redistribute_fraction = min(0.8, 0.2 + iteration / 1000.0)
                    bright_total = weighted_target[bright_mask].sum() + 1e-12
                    # Add proportionally to existing bright amplitudes
                    boost = redistribute_fraction * dark_leaked_amp / bright_total
                    weighted_target[bright_mask] *= (1.0 + boost * 0.05)

            # Suppress dark border regions progressively
            if iteration > 40:
                border_suppress = min(0.55, (iteration - 40) / 350.0)
                weighted_target[dark_border_mask] *= (1.0 - border_suppress)
            if iteration > 100:
                wide_border_suppress = min(0.25, (iteration - 100) / 600.0)
                weighted_target[dark_border_mask_wide] *= (1.0 - wide_border_suppress)

            constrained_far = weighted_target * np.exp(1j * far_phase)
            slm_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(constrained_far), norm="ortho"))
            slm_phase = np.angle(slm_field)

            # Periodically check score and save best
            if iteration >= 80 and iteration % 30 == 0:
                s = score_fn(slm_phase)
                if s > restart_best_score:
                    restart_best_score = s
                    restart_best_phase = slm_phase.copy()

        # Final score
        score = score_fn(slm_phase)
        if score > restart_best_score:
            restart_best_score = score
            restart_best_phase = slm_phase.copy()

        # Collect error statistics for inter-restart learning
        near_final = aperture * np.exp(1j * restart_best_phase)
        far_final = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near_final), norm="ortho"))
        far_amp_final = np.abs(far_final)
        far_scale_final = target_amp[bright_mask].sum() / (far_amp_final[bright_mask].sum() + 1e-12)
        global_error_accum += (target_amp - far_amp_final * far_scale_final) * signal_mask
        global_error_count += 1

        # Update top-k list
        if len(top_scores) < top_k:
            top_scores.append(restart_best_score)
            top_phases.append(restart_best_phase.copy())
        else:
            min_idx = np.argmin(top_scores)
            if restart_best_score > top_scores[min_idx]:
                top_scores[min_idx] = restart_best_score
                top_phases[min_idx] = restart_best_phase.copy()

        if restart_best_score > best_score:
            best_score = restart_best_score
            best_phase = restart_best_phase.copy()

    # ========== Stage 2: Gradient-based Refinement (Adam) on top-k candidates ==========
    overall_best_phase = best_phase.copy()
    overall_best_score = best_score

    for candidate_idx in range(len(top_phases)):
        phase = top_phases[candidate_idx].copy()
        current_score = top_scores[candidate_idx]

        m_adam = np.zeros_like(phase)
        v_adam = np.zeros_like(phase)
        beta1 = 0.9
        beta2 = 0.999
        eps_adam = 1e-8

        num_grad_steps = 800
        best_grad_phase = phase.copy()
        best_grad_score = current_score
        stagnation_count = 0
        num_resets = 0

        for step in range(num_grad_steps):
            t_step = step + 1

            # Learning rate with warmup and cosine decay
            if step < 30:
                lr = 0.028 * (step + 1) / 30.0
            else:
                progress = (step - 30) / max(1, num_grad_steps - 30)
                lr = 0.028 * 0.5 * (1.0 + np.cos(np.pi * progress))

            # Cycle through weight configurations focusing on different objectives
            cycle = step % 5
            if cycle == 0:
                w_p, w_e, w_d = 0.75, 0.05, 0.20  # Heavy pattern
            elif cycle == 1:
                w_p, w_e, w_d = 0.55, 0.10, 0.35  # Heavy dark suppression
            elif cycle == 2:
                w_p, w_e, w_d = 0.65, 0.15, 0.20  # Balanced pattern+dark
            elif cycle == 3:
                w_p, w_e, w_d = 0.55, 0.30, 0.15  # Match scoring weights
            else:
                w_p, w_e, w_d = 0.60, 0.20, 0.20  # Even balance

            grad = _compute_loss_and_gradient(
                aperture, phase, target_amp, T, T_sum, T_mean,
                bright_mask, dark_mask, aperture_mask,
                w_pattern=w_p, w_energy=w_e, w_dark=w_d
            )

            # Adam update
            m_adam = beta1 * m_adam + (1.0 - beta1) * grad
            v_adam = beta2 * v_adam + (1.0 - beta2) * grad ** 2
            m_hat = m_adam / (1.0 - beta1 ** t_step)
            v_hat = v_adam / (1.0 - beta2 ** t_step)

            phase_new = phase - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            new_score = score_fn(phase_new)
            if new_score >= current_score - 0.5:
                phase = phase_new
                current_score = new_score
                if new_score > best_grad_score:
                    best_grad_score = new_score
                    best_grad_phase = phase.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                m_adam *= 0.5
                v_adam *= 0.5
                stagnation_count += 1

            if stagnation_count > 18:
                num_resets += 1
                perturb_scale = 0.015 * (0.8 ** min(num_resets, 5))
                phase = best_grad_phase.copy() + perturb_scale * rng.standard_normal((n, n)) * aperture_mask
                current_score = score_fn(phase)
                m_adam *= 0.0
                v_adam *= 0.0
                stagnation_count = 0

        if best_grad_score > overall_best_score:
            overall_best_score = best_grad_score
            overall_best_phase = best_grad_phase.copy()

    # ========== Stage 2b: Second gradient pass with different weights ==========
    for weight_set in [(0.85, 0.02, 0.13), (0.75, 0.05, 0.20), (0.50, 0.10, 0.40), (0.60, 0.20, 0.20), (0.55, 0.30, 0.15), (0.40, 0.05, 0.55)]:
        phase = overall_best_phase.copy()
        current_score = overall_best_score
        best_grad_phase2 = phase.copy()
        best_grad_score2 = current_score

        m_adam = np.zeros_like(phase)
        v_adam = np.zeros_like(phase)

        n_steps_2b = 250
        for step in range(n_steps_2b):
            t_step = step + 1
            progress = step / float(n_steps_2b)
            lr = 0.012 * 0.5 * (1.0 + np.cos(np.pi * progress))

            grad = _compute_loss_and_gradient(
                aperture, phase, target_amp, T, T_sum, T_mean,
                bright_mask, dark_mask, aperture_mask,
                w_pattern=weight_set[0], w_energy=weight_set[1], w_dark=weight_set[2]
            )

            m_adam = 0.9 * m_adam + 0.1 * grad
            v_adam = 0.999 * v_adam + 0.001 * grad ** 2
            m_hat = m_adam / (1.0 - 0.9 ** t_step)
            v_hat = v_adam / (1.0 - 0.999 ** t_step)

            phase_new = phase - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            new_score = score_fn(phase_new)

            if new_score >= current_score - 0.3:
                phase = phase_new
                current_score = new_score
                if new_score > best_grad_score2:
                    best_grad_score2 = new_score
                    best_grad_phase2 = phase.copy()

        if best_grad_score2 > overall_best_score:
            overall_best_score = best_grad_score2
            overall_best_phase = best_grad_phase2.copy()

    # ========== Stage 2c: WGS refinement from gradient-optimized solution ==========
    phase_for_wgs = overall_best_phase.copy()
    wgs_refine_best_phase = phase_for_wgs.copy()
    wgs_refine_best_score = overall_best_score

    for wgs_refine_restart in range(3):
        slm_phase_r = phase_for_wgs.copy()
        if wgs_refine_restart > 0:
            slm_phase_r = slm_phase_r + 0.025 * rng.standard_normal((n, n)) * aperture_mask

        weights_r = np.ones_like(target_amp)

        for iteration in range(500):
            near_field = aperture * np.exp(1j * slm_phase_r)
            far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near_field), norm="ortho"))
            far_amp = np.abs(far_field)
            far_phase = np.angle(far_field)

            far_amp_safe = np.maximum(far_amp, 1e-12)
            ratio = target_amp / far_amp_safe
            ratio = np.clip(ratio, 0.02, 50.0)

            alpha_r = min(0.97, 0.85 + iteration / 1500.0)
            weights_r = weights_r * (alpha_r + (1.0 - alpha_r) * ratio)

            if signal_mask.any():
                w_mean = weights_r[signal_mask].mean() + 1e-12
                weights_r = weights_r / w_mean

            weights_r[dark_mask] = 0.0
            weights_r[dark_border_mask] *= 0.88
            weights_r[dark_border_mask_wide] *= 0.94

            weighted_target_r = weights_r * target_amp
            weighted_target_r[dark_mask] = 0.0
            border_suppress_r = min(0.55, iteration / 300.0)
            weighted_target_r[dark_border_mask] *= (1.0 - border_suppress_r)

            # Dark zone energy redistribution during refinement
            if iteration > 5:
                dark_leaked = far_amp[dark_mask].sum()
                if dark_leaked > 0 and bright_mask.any():
                    bright_total = weighted_target_r[bright_mask].sum() + 1e-12
                    boost = 0.03 * dark_leaked / bright_total
                    weighted_target_r[bright_mask] *= (1.0 + boost)

            constrained_far = weighted_target_r * np.exp(1j * far_phase)
            slm_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(constrained_far), norm="ortho"))
            slm_phase_r = np.angle(slm_field)

            if iteration >= 50 and iteration % 25 == 0:
                s = score_fn(slm_phase_r)
                if s > wgs_refine_best_score:
                    wgs_refine_best_score = s
                    wgs_refine_best_phase = slm_phase_r.copy()

        s = score_fn(slm_phase_r)
        if s > wgs_refine_best_score:
            wgs_refine_best_score = s
            wgs_refine_best_phase = slm_phase_r.copy()

    if wgs_refine_best_score > overall_best_score:
        overall_best_score = wgs_refine_best_score
        overall_best_phase = wgs_refine_best_phase.copy()

    # ========== Stage 2d: Final gradient polish on the best solution ==========
    # Multiple passes with different weight focuses
    for polish_weights in [(0.80, 0.02, 0.18), (0.65, 0.10, 0.25), (0.55, 0.30, 0.15)]:
        phase = overall_best_phase.copy()
        current_score = overall_best_score
        best_polish_phase = phase.copy()
        best_polish_score = current_score

        m_adam = np.zeros_like(phase)
        v_adam = np.zeros_like(phase)

        for step in range(250):
            t_step = step + 1
            progress = step / 250.0
            lr = 0.010 * 0.5 * (1.0 + np.cos(np.pi * progress))

            grad = _compute_loss_and_gradient(
                aperture, phase, target_amp, T, T_sum, T_mean,
                bright_mask, dark_mask, aperture_mask,
                w_pattern=polish_weights[0], w_energy=polish_weights[1], w_dark=polish_weights[2]
            )

            m_adam = 0.9 * m_adam + 0.1 * grad
            v_adam = 0.999 * v_adam + 0.001 * grad ** 2
            m_hat = m_adam / (1.0 - 0.9 ** t_step)
            v_hat = v_adam / (1.0 - 0.999 ** t_step)

            phase_new = phase - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            new_score = score_fn(phase_new)

            if new_score >= current_score - 0.3:
                phase = phase_new
                current_score = new_score
                if new_score > best_polish_score:
                    best_polish_score = new_score
                    best_polish_phase = phase.copy()

        if best_polish_score > overall_best_score:
            overall_best_score = best_polish_score
            overall_best_phase = best_polish_phase.copy()

    best_grad_phase = overall_best_phase
    best_grad_score = overall_best_score

    # ========== Stage 3: SPSA Polish ==========
    phase = best_grad_phase.copy()
    current_score = best_grad_score

    num_spsa = 250
    for rnd in range(num_spsa):
        t = rnd / num_spsa
        ps = 0.10 * (1.0 - 0.7 * t)

        delta = rng.standard_normal((n, n)) * ps
        delta[~aperture_mask] = 0.0

        phase_plus = phase + delta
        score_plus = score_fn(phase_plus)

        phase_minus = phase - delta
        score_minus = score_fn(phase_minus)

        if score_plus >= score_minus and score_plus > current_score:
            phase = phase_plus
            current_score = score_plus
        elif score_minus > current_score:
            phase = phase_minus
            current_score = score_minus

    # ========== Stage 4: Fine SPSA with smaller perturbations ==========
    num_fine_spsa = 180
    for rnd in range(num_fine_spsa):
        t = rnd / num_fine_spsa
        ps = 0.04 * (1.0 - 0.5 * t)

        delta = rng.choice([-1.0, 1.0], size=(n, n)) * ps
        delta[~aperture_mask] = 0.0

        phase_plus = phase + delta
        score_plus = score_fn(phase_plus)

        phase_minus = phase - delta
        score_minus = score_fn(phase_minus)

        if score_plus >= score_minus and score_plus > current_score:
            phase = phase_plus
            current_score = score_plus
        elif score_minus > current_score:
            phase = phase_minus
            current_score = score_minus

    # ========== Stage 5: Gradient-guided line search ==========
    # More efficient than random SPSA: use gradient to find good directions
    best_final_phase = phase.copy()
    best_final_score = current_score

    for rnd in range(150):
        t = rnd / 150.0

        # Compute gradient with varying weight emphasis
        cycle = rnd % 5
        if cycle == 0:
            w_p, w_e, w_d = 0.75, 0.05, 0.20
        elif cycle == 1:
            w_p, w_e, w_d = 0.55, 0.15, 0.30
        elif cycle == 2:
            w_p, w_e, w_d = 0.65, 0.10, 0.25
        elif cycle == 3:
            w_p, w_e, w_d = 0.55, 0.30, 0.15
        else:
            w_p, w_e, w_d = 0.40, 0.10, 0.50

        grad = _compute_loss_and_gradient(
            aperture, phase, target_amp, T, T_sum, T_mean,
            bright_mask, dark_mask, aperture_mask,
            w_pattern=w_p, w_energy=w_e, w_dark=w_d
        )

        grad_norm = np.sqrt(np.mean(grad[aperture_mask] ** 2)) + 1e-12
        direction = grad / grad_norm

        # Try multiple step sizes along gradient direction (line search)
        base_lr = 0.012 * (1.0 - 0.7 * t)
        best_ls_score = current_score
        best_ls_phase = None

        for lr_mult in [0.25, 0.5, 1.0, 2.0]:
            lr = base_lr * lr_mult
            phase_try = phase - lr * direction
            try_score = score_fn(phase_try)
            if try_score > best_ls_score:
                best_ls_score = try_score
                best_ls_phase = phase_try

        if best_ls_phase is not None:
            phase = best_ls_phase
            current_score = best_ls_score
            if current_score > best_final_score:
                best_final_score = current_score
                best_final_phase = phase.copy()

    # ========== Stage 6: Fine SPSA with gradient-informed perturbations ==========
    phase = best_final_phase.copy()
    current_score = best_final_score

    num_vfine_spsa = 120
    for rnd in range(num_vfine_spsa):
        t = rnd / num_vfine_spsa
        ps = 0.015 * (1.0 - 0.5 * t)

        delta = rng.choice([-1.0, 1.0], size=(n, n)) * ps
        delta[~aperture_mask] = 0.0

        phase_plus = phase + delta
        score_plus = score_fn(phase_plus)

        phase_minus = phase - delta
        score_minus = score_fn(phase_minus)

        if score_plus >= score_minus and score_plus > current_score:
            phase = phase_plus
            current_score = score_plus
            if current_score > best_final_score:
                best_final_score = current_score
                best_final_phase = phase.copy()
        elif score_minus > current_score:
            phase = phase_minus
            current_score = score_minus
            if current_score > best_final_score:
                best_final_score = current_score
                best_final_phase = phase.copy()

    # ========== Stage 7: Ultra-fine SPSA ==========
    phase = best_final_phase.copy()
    current_score = best_final_score

    num_ufine_spsa = 80
    for rnd in range(num_ufine_spsa):
        t = rnd / num_ufine_spsa
        ps = 0.006 * (1.0 - 0.4 * t)

        delta = rng.choice([-1.0, 1.0], size=(n, n)) * ps
        delta[~aperture_mask] = 0.0

        phase_plus = phase + delta
        score_plus = score_fn(phase_plus)

        phase_minus = phase - delta
        score_minus = score_fn(phase_minus)

        if score_plus >= score_minus and score_plus > current_score:
            phase = phase_plus
            current_score = score_plus
            if current_score > best_final_score:
                best_final_score = current_score
                best_final_phase = phase.copy()
        elif score_minus > current_score:
            phase = phase_minus
            current_score = score_minus
            if current_score > best_final_score:
                best_final_score = current_score
                best_final_phase = phase.copy()

    # ========== Stage 8: Final gradient line search with very small steps ==========
    phase = best_final_phase.copy()
    current_score = best_final_score

    for rnd in range(60):
        t = rnd / 60.0

        # Alternate weight emphasis
        if rnd % 3 == 0:
            w_p, w_e, w_d = 0.70, 0.10, 0.20
        elif rnd % 3 == 1:
            w_p, w_e, w_d = 0.55, 0.20, 0.25
        else:
            w_p, w_e, w_d = 0.55, 0.30, 0.15

        grad = _compute_loss_and_gradient(
            aperture, phase, target_amp, T, T_sum, T_mean,
            bright_mask, dark_mask, aperture_mask,
            w_pattern=w_p, w_energy=w_e, w_dark=w_d
        )

        grad_norm = np.sqrt(np.mean(grad[aperture_mask] ** 2)) + 1e-12
        direction = grad / grad_norm

        base_lr = 0.005 * (1.0 - 0.5 * t)
        for lr_mult in [0.3, 0.7, 1.0, 1.5]:
            lr = base_lr * lr_mult
            phase_try = phase - lr * direction
            try_score = score_fn(phase_try)
            if try_score > current_score:
                phase = phase_try
                current_score = try_score
                if current_score > best_final_score:
                    best_final_score = current_score
                    best_final_phase = phase.copy()
                break

    # ========== Stage 9: Nano SPSA ==========
    phase = best_final_phase.copy()
    current_score = best_final_score

    for rnd in range(50):
        ps = 0.003 * (1.0 - 0.3 * rnd / 50.0)
        delta = rng.choice([-1.0, 1.0], size=(n, n)) * ps
        delta[~aperture_mask] = 0.0

        phase_plus = phase + delta
        score_plus = score_fn(phase_plus)

        phase_minus = phase - delta
        score_minus = score_fn(phase_minus)

        if score_plus >= score_minus and score_plus > current_score:
            phase = phase_plus
            current_score = score_plus
            if current_score > best_final_score:
                best_final_score = current_score
                best_final_phase = phase.copy()
        elif score_minus > current_score:
            phase = phase_minus
            current_score = score_minus
            if current_score > best_final_score:
                best_final_score = current_score
                best_final_phase = phase.copy()

    return best_final_phase


def forward_intensity(problem: Dict[str, Any], phase: np.ndarray) -> np.ndarray:
    near = problem["aperture_amp"] * np.exp(1j * phase)
    far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
    return np.abs(far) ** 2


def save_solution(path: Path, problem: Dict[str, Any], phase: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        phase=phase.astype(np.float32),
        x=problem["x"].astype(np.float32),
        y=problem["y"].astype(np.float32),
        aperture_amp=problem["aperture_amp"].astype(np.float32),
        target_amp=problem["target_amp"].astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task02 baseline solver")
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
    phase = solve_baseline(problem)
    save_solution(args.output, problem, phase)

    I = forward_intensity(problem, phase)
    print("[Task02/Baseline] solution saved:", args.output)
    print("[Task02/Baseline] intensity stats: min={:.6g}, max={:.6g}, mean={:.6g}".format(I.min(), I.max(), I.mean()))


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END