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

    # Build pattern using vectorized operations where possible
    target = np.zeros((n, n), dtype=float)

    # Grid of Gaussian spots
    xs = np.linspace(18, 110, 8)
    ys = np.linspace(18, 110, 8)
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            amp = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(0.7 * i + 0.9 * j))
            if (i + j) % 2 == 0:
                amp *= 0.4
            target += amp * np.exp(-((x - xx) ** 2 + (y - yy) ** 2) / (2.0 * 0.9**2))

    # Add sine wave pattern
    for xx in range(20, 108):
        yy = int(64 + 18 * np.sin((xx - 20) / 13.0))
        target[max(0, yy - 1):min(n, yy + 2), max(0, xx - 1):min(n, xx + 2)] += 0.35

    # Dark zone - vectorized for clarity
    dark_zone = (np.abs(x - c) < 4) & (np.abs(y - c) < 45)
    target[dark_zone] = 0.0

    # Better normalization that preserves relative intensities
    target = np.clip(target, 0.0, None)
    
    # Use robust normalization based on 95th percentile for better dynamic range
    non_zero_mask = target > 1e-12
    if np.any(non_zero_mask):
        sorted_vals = np.sort(target[non_zero_mask])
        # Use 95th percentile for robust normalization
        idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
        max_val = sorted_vals[idx]
        if max_val > 1e-12:
            target = target / max_val
    
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
    """Iterative phase retrieval with improved initialization and dark zone suppression."""
    seed_value = int(problem["cfg"]["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed_value)

    target_amp = problem["target_amp"]
    aperture_amp = problem["aperture_amp"]
    
    # Improved initialization with multiple frequency components
    random_phase = 2.0 * np.pi * rng.random(target_amp.shape)
    y, x = np.indices(target_amp.shape)
    c = (target_amp.shape[0] - 1) / 2.0
    
    # Multi-scale phase components for better initialization
    smooth_phase = 0.1 * np.sin((x - c) / 10.0) * np.cos((y - c) / 10.0)
    medium_phase = 0.08 * np.sin((x - c) / 5.0) * np.cos((y - c) / 5.0)
    fine_phase = 0.05 * np.sin((x - c) / 2.5) * np.cos((y - c) / 2.5)
    
    initial_phase = random_phase + smooth_phase + medium_phase + fine_phase
    
    # Iterative refinement using Gerchberg-Kim algorithm with dark zone suppression
    phase = initial_phase.copy()
    # Use fewer iterations for better efficiency (based on successful experiments)
    iterations = 25
    
    for i in range(iterations):
        # Forward propagation
        near_field = aperture_amp * np.exp(1j * phase)
        far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near_field), norm="ortho"))
        intensity = np.abs(far_field) ** 2
        
        # Update in Fourier domain with target constraints
        far_amplitude = np.sqrt(np.maximum(intensity, 0))
        target_intensity = np.maximum(target_amp, 0) ** 2
        
        # Weighted update based on target amplitude (higher weight for strong features)
        weight = np.sqrt(np.maximum(target_amp, 0.1)) / (np.sqrt(np.maximum(target_amp, 0.1)) + 0.1)
        far_amplitude = weight * np.sqrt(np.maximum(target_intensity, 0)) + (1 - weight) * far_amplitude
        
        # Enforce Fourier domain constraints
        far_field = far_amplitude * np.exp(1j * np.angle(far_field))
        
        # Backward propagation
        near_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field), norm="ortho"))
        
        # Enforce support constraint in real domain
        phase = np.angle(near_field)
        near_field = aperture_amp * np.exp(1j * phase)
        
        # Dark zone suppression (simplified but effective)
        dark_mask = target_amp < 0.03
        if np.any(dark_mask) and i > 5:  # Start suppressing after initial convergence
            # Calculate current dark zone energy ratio
            far_field_check = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near_field), norm="ortho"))
            dark_intensity = np.abs(far_field_check) ** 2
            dark_energy = dark_intensity[dark_mask].sum()
            total_energy = intensity.sum()
            dark_ratio = dark_energy / (total_energy + 1e-12)
            
            # Apply adaptive suppression only to dark regions
            if dark_ratio > 0.01:  # Only suppress if dark zone leakage is significant
                suppression = np.exp(-dark_ratio * 5)
                near_field[dark_mask] *= suppression  # Direct suppression in real space
                
        # Ensure phase stays within [0, 2π) for numerical stability
        phase = np.mod(phase, 2 * np.pi)
    
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
