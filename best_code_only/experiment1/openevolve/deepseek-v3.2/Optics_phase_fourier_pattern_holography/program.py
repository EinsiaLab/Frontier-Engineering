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


def solve_baseline(problem: Dict[str, Any], seed: int | None = None) -> np.ndarray:
    """Adaptive Gerchberg-Saxton with iterative weighting and phase smoothing."""
    seed_value = int(problem["cfg"]["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed_value)
    
    target_amp = problem["target_amp"]
    aperture_amp = problem["aperture_amp"]
    
    # Adaptive thresholds based on target statistics
    dark_threshold = 0.05  # slightly higher to be more forgiving
    bright_threshold = 0.25  # focus on moderately bright regions
    
    # Initialize phase with a structured pattern: use target's Fourier phase
    far_field = target_amp * np.exp(1j * 2.0 * np.pi * rng.random(target_amp.shape))
    near_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field), norm="ortho"))
    phase = np.angle(near_field) * aperture_amp
    
    num_iter = 120  # More iterations for better convergence
    
    for iter in range(num_iter):
        # Forward propagation: SLM to far field
        near_field = aperture_amp * np.exp(1j * phase)
        far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near_field), norm="ortho"))
        
        # Adaptive weighting: gradually increase emphasis on target regions
        # Early iterations: focus on overall structure, later iterations: precise matching
        weight_factor = 0.5 + 0.5 * (iter / num_iter)  # From 0.5 to 1.0
        
        weighted_target = target_amp.copy()
        dark_mask = target_amp < dark_threshold
        bright_mask = target_amp > bright_threshold
        
        # Apply weights that change with iteration
        weighted_target[dark_mask] *= 0.1 * weight_factor  # More suppression later
        weighted_target[bright_mask] *= 1.2 + 0.3 * (iter / num_iter)  # Boost increases
        
        # Replace amplitude with weighted target, keep phase
        far_field_phase = np.angle(far_field)
        far_field_new = weighted_target * np.exp(1j * far_field_phase)
        
        # Backward propagation: far field to SLM
        back_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field_new), norm="ortho"))
        
        # Update phase with aperture constraint and smoothing
        new_phase = np.angle(back_field)
        
        # Apply smoothing within aperture: average with previous phase to reduce abrupt changes
        smoothing_factor = 0.1 * (1.0 - iter / num_iter)  # Less smoothing later
        phase = aperture_amp * (new_phase * (1.0 - smoothing_factor) + phase * smoothing_factor)
    
    # Post-processing: apply phase unwrapping within aperture to ensure continuity
    aperture_mask = aperture_amp > 0
    if aperture_mask.any():
        # Extract phase within aperture
        phase_aperture = phase[aperture_mask]
        # Simple unwrapping: add 2π jumps when phase difference > π
        # We'll reshape to 1D for simplicity
        phase_flat = phase_aperture.flatten()
        # Instead, just ensure phase is within [0, 2π]
        phase_flat = phase_flat % (2.0 * np.pi)
        phase[aperture_mask] = phase_flat
    
    return phase


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
