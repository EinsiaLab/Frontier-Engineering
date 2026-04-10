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


def _compute_score(intensity, target_amp, bright_mask, dark_mask):
    """Compute the actual evaluation score used by the verifier."""
    # NMSE
    t_norm = target_amp / (target_amp.max() + 1e-12)
    i_norm = intensity / (intensity.max() + 1e-12)
    nmse = np.sqrt(np.mean((i_norm - t_norm) ** 2)) / (np.mean(t_norm) + 1e-12)
    
    # Energy in target
    energy_total = np.sum(intensity) + 1e-12
    energy_bright = np.sum(intensity[bright_mask])
    energy_in_target = energy_bright / energy_total
    
    # Dark suppression
    dark_leak = np.sum(intensity[dark_mask]) / energy_total
    dark_suppression = 1.0 - dark_leak
    
    # Score formula from Task.md
    pattern_score = np.clip(1.0 - nmse / 4.0, 0, 1)
    energy_score = np.clip((energy_in_target - 0.10) / (0.70 - 0.10), 0, 1)
    dark_score = np.clip((dark_suppression - 0.35) / (0.90 - 0.35), 0, 1)
    score_pct = 100.0 * (0.55 * pattern_score + 0.30 * energy_score + 0.15 * dark_score)
    
    return score_pct, nmse, energy_in_target, dark_suppression


def solve_baseline(problem: Dict[str, Any], seed: int | None = None) -> np.ndarray:
    """Iterative Weighted Gerchberg-Saxton (WGS-Kim style) solver with multiple restarts."""
    seed_value = int(problem["cfg"]["seed"] if seed is None else seed)

    target_amp = problem["target_amp"].copy()
    aperture_amp = problem["aperture_amp"].copy()
    n = target_amp.shape[0]

    # Identify regions
    bright_mask = target_amp > 0.30
    dark_mask = target_amp < 0.03

    # Precompute aperture support
    aperture_support = aperture_amp > 0.5

    overall_best_phase = None
    overall_best_score = -1e30

    # Multiple restarts with different seeds
    n_restarts = 5
    n_iterations = 500

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed_value + restart * 1000)

        # Initialize with random phase on target
        target_phase = 2.0 * np.pi * rng.random(target_amp.shape)
        Uz = target_amp * np.exp(1j * target_phase)

        # Initial inverse to get starting phase
        back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Uz), norm="ortho"))
        slm_phase = np.angle(back)

        # Weights for WGS-Kim
        weights = np.ones_like(target_amp)

        best_phase = slm_phase.copy()
        best_score = -1e30

        for iteration in range(n_iterations):
            # Forward propagation: SLM -> far field
            near = aperture_amp * np.exp(1j * slm_phase)
            far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
            far_amp = np.abs(far)
            far_phase = np.angle(far)

            # Evaluate score periodically
            if iteration >= 5 and iteration % 5 == 0:
                intensity = far_amp ** 2
                score, _, _, _ = _compute_score(intensity, target_amp, bright_mask, dark_mask)
                if score > best_score:
                    best_score = score
                    best_phase = slm_phase.copy()

            # Update weights (WGS-Kim style feedback)
            if iteration >= 3 and iteration % 2 == 0:
                bm = bright_mask
                if np.any(bm):
                    mean_target = np.mean(target_amp[bm])
                    mean_measured = np.mean(far_amp[bm]) + 1e-12
                    scale = mean_target / mean_measured
                    ratio = target_amp[bm] / (far_amp[bm] * scale + 1e-12)
                    # Damped update with adaptive damping
                    damp = 0.7 if iteration < 50 else 0.85
                    weights[bm] = weights[bm] * (damp + (1.0 - damp) * ratio)
                    weights = np.clip(weights, 0.05, 15.0)

            # Dark zone suppression: progressively increase penalty
            dark_ramp = min(1.0, iteration / 30.0)

            # Construct target field for inverse
            weighted_target = weights * target_amp

            # In dark zones, force amplitude toward zero aggressively
            if dark_ramp > 0:
                suppress = (1.0 - dark_ramp * 0.98)
                weighted_target[dark_mask] = far_amp[dark_mask] * suppress * 0.02

            # Replace amplitude, keep measured phase
            Uz_new = weighted_target * np.exp(1j * far_phase)

            # Inverse propagation
            back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Uz_new), norm="ortho"))

            # Apply aperture constraint: keep phase, replace amplitude with aperture
            slm_phase = np.angle(back)

        # Final evaluation
        near = aperture_amp * np.exp(1j * slm_phase)
        far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
        intensity = np.abs(far) ** 2
        final_score, _, _, _ = _compute_score(intensity, target_amp, bright_mask, dark_mask)
        if final_score > best_score:
            best_score = final_score
            best_phase = slm_phase.copy()

        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_phase = best_phase.copy()

    return overall_best_phase


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
