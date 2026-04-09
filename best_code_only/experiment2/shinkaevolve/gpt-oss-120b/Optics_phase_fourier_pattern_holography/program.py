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
    """Enhanced GS-WGS optimizer with aggressive dark suppression and block refinement."""
    seed_value = int(problem["cfg"]["seed"] if seed is None else seed)
    base_rng = np.random.default_rng(seed_value)

    target_amp = problem["target_amp"]
    aperture_amp = problem["aperture_amp"]
    n = target_amp.shape[0]

    # Normalize target amplitude with slight energy boost
    aperture_area = np.sum(aperture_amp)
    target_sum = np.sum(target_amp**2)
    if target_sum > 0:
        target_amp_norm = target_amp * np.sqrt(aperture_area / target_sum) * 0.95
    else:
        target_amp_norm = target_amp

    target_intensity = target_amp_norm ** 2

    # Region masks with expanded dark region for better suppression
    bright_mask = target_amp > 0.30
    dark_mask = target_amp < 0.03
    dark_dilated = target_amp < 0.06  # Expanded dark region for penalty
    mid_mask = ~bright_mask & ~dark_mask

    def compute_score(current_amp):
        """Compute evaluation score (higher is better)."""
        current_intensity = current_amp ** 2

        nmse = np.sqrt(np.mean((current_intensity - target_intensity)**2)) / (np.mean(target_intensity) + 1e-12)
        pattern_score = np.clip(1 - nmse / 4.0, 0, 1)

        total_energy = np.sum(current_intensity) + 1e-12
        energy_in_target = np.sum(current_intensity[bright_mask]) / total_energy if np.any(bright_mask) else 0
        energy_score = np.clip((energy_in_target - 0.10) / (0.70 - 0.10), 0, 1)

        if np.any(dark_mask):
            leak = np.sum(current_intensity[dark_mask]) / total_energy
            dark_suppression = 1 - leak
        else:
            dark_suppression = 1.0
        dark_score = np.clip((dark_suppression - 0.35) / (0.90 - 0.35), 0, 1)

        return 0.55 * pattern_score + 0.30 * energy_score + 0.15 * dark_score

    def gs_iterations(near, num_iter, use_momentum=False):
        """Gerchberg-Saxton iterations with optional momentum for faster convergence."""
        if not use_momentum:
            for _ in range(num_iter):
                far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
                far = target_amp_norm * np.exp(1j * np.angle(far))
                near = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far), norm="ortho"))
                near = aperture_amp * np.exp(1j * np.angle(near))
        else:
            # Momentum-accelerated GS
            prev_phase = np.angle(near)
            momentum_beta = 0.3
            for i in range(num_iter):
                far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
                far = target_amp_norm * np.exp(1j * np.angle(far))
                near = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far), norm="ortho"))
                new_phase = np.angle(near)
                # Apply momentum to phase update
                phase_diff = new_phase - prev_phase
                # Wrap phase difference to [-pi, pi]
                phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
                momentum_phase = prev_phase + (1 + momentum_beta) * phase_diff
                momentum_phase = np.arctan2(np.sin(momentum_phase), np.cos(momentum_phase))
                near = aperture_amp * np.exp(1j * momentum_phase)
                prev_phase = np.angle(near)
        return near

    def block_refinement(phase, num_blocks=20, initial_block_size=7):
        """Multi-scale block refinement with adaptive block sizes for better convergence."""
        best_phase = phase.copy()
        near = aperture_amp * np.exp(1j * phase)
        far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
        best_score = compute_score(np.abs(far))

        current_phase = phase.copy()
        current_score = best_score

        # Multi-scale: start with larger blocks, decrease size
        block_sizes = [initial_block_size, initial_block_size - 2, initial_block_size - 4]
        block_sizes = [max(3, bs) for bs in block_sizes]
        blocks_per_size = num_blocks // len(block_sizes)

        for block_size in block_sizes:
            for block_idx in range(blocks_per_size):
                # Perturb a block of pixels
                cy = base_rng.integers(block_size, n - block_size)
                cx = base_rng.integers(block_size, n - block_size)

                # Create a smooth perturbation
                perturbation = np.zeros((n, n))
                y_idx, x_idx = np.ogrid[cy-block_size:cy+block_size+1, cx-block_size:cx+block_size+1]

                # Gaussian-weighted perturbation
                sigma = block_size / 2.0
                gauss_weight = np.exp(-((y_idx - cy)**2 + (x_idx - cx)**2) / (2 * sigma**2))
                perturbation[cy-block_size:cy+block_size+1, cx-block_size:cx+block_size+1] = gauss_weight

                # Test multiple perturbation sizes for better local search
                best_local_score = current_score
                best_local_phase = current_phase

                for delta_scale in [0.12, 0.25, 0.4, 0.6, 0.85]:
                    delta = base_rng.uniform(-delta_scale, delta_scale)
                    test_phase = current_phase + delta * perturbation

                    near = aperture_amp * np.exp(1j * test_phase)
                    far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
                    score = compute_score(np.abs(far))

                    if score > best_local_score:
                        best_local_score = score
                        best_local_phase = test_phase

                if best_local_score > current_score:
                    current_phase = best_local_phase
                    current_score = best_local_score

                    if current_score > best_score:
                        best_score = current_score
                        best_phase = current_phase.copy()

        return best_phase, best_score

    def pixel_refinement(phase, num_pixels=100):
        """Fine-grained pixel-level refinement for final tuning."""
        best_phase = phase.copy()
        near = aperture_amp * np.exp(1j * best_phase)
        far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
        best_score = compute_score(np.abs(far))

        # Focus on aperture region
        aperture_pixels = np.where(aperture_amp > 0.5)
        if len(aperture_pixels[0]) == 0:
            return best_phase, best_score

        for _ in range(num_pixels):
            # Random pixel within aperture
            idx = base_rng.integers(0, len(aperture_pixels[0]))
            py, px = aperture_pixels[0][idx], aperture_pixels[1][idx]

            # Test phase perturbations
            for delta in [-0.5, -0.25, 0.25, 0.5]:
                test_phase = best_phase.copy()
                test_phase[py, px] = np.arctan2(np.sin(test_phase[py, px] + delta),
                                                 np.cos(test_phase[py, px] + delta))

                near = aperture_amp * np.exp(1j * test_phase)
                far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
                score = compute_score(np.abs(far))

                if score > best_score:
                    best_score = score
                    best_phase = test_phase.copy()

        return best_phase, best_score

    def run_optimization(init_seed: int, strategy: str = 'gs_wgs') -> tuple:
        """Run hybrid optimization with specified strategy."""
        rng = np.random.default_rng(init_seed)

        # Initialize phase with diverse strategies
        if strategy == 'uniform':
            phase = 2.0 * np.pi * rng.random(target_amp.shape)
        else:
            Uz = target_amp_norm * np.exp(1j * 2.0 * np.pi * rng.random(target_amp.shape))
            back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Uz), norm="ortho"))
            phase = np.angle(back)

        near = aperture_amp * np.exp(1j * phase)

        # Phase 1: GS pre-iteration with momentum for faster convergence
        gs_iters = 60 if strategy == 'gs_heavy' else 40
        near = gs_iterations(near, gs_iters, use_momentum=True)

        # Initialize weights with very aggressive dark suppression
        weight = np.ones_like(target_amp)
        weight[bright_mask] = 4.0
        weight[mid_mask] = 1.2
        weight[dark_mask] = 0.02  # Very low weight for dark regions

        weight_momentum = np.zeros_like(weight)
        momentum_alpha = 0.25

        best_phase = np.angle(near)
        best_score = 0.0
        stagnation = 0
        prev_score = 0.0

        max_iter = 250

        for iteration in range(max_iter):
            progress = iteration / max_iter

            # Adaptive gamma: start aggressive, become conservative
            gamma = 0.75 - 0.25 * progress

            # Adaptive smoothing
            smoothing = 0.92 - 0.12 * progress

            # Forward propagate
            far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
            far_amp = np.abs(far)
            far_phase = np.angle(far)

            # Enhanced weighted amplitude constraint with explicit dark penalty
            new_far_amp = weight * target_amp_norm + (1 - weight) * far_amp

            # Additional dark region suppression - gentler for better pattern matching
            dark_penalty = 1.0 - 0.18 * dark_dilated.astype(float)
            new_far_amp = new_far_amp * dark_penalty

            far = new_far_amp * np.exp(1j * far_phase)

            # Back propagate
            near = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far), norm="ortho"))
            near = aperture_amp * np.exp(1j * np.angle(near))

            # Compute score
            current_far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))
            current_amp = np.abs(current_far)
            total_score = compute_score(current_amp)

            if total_score > best_score:
                best_score = total_score
                best_phase = np.angle(near)

            # Check stagnation with recovery mechanism
            if abs(total_score - prev_score) < 1e-6:
                stagnation += 1
                if stagnation > 25:
                    # Apply strategic perturbation to escape local minimum
                    perturbation_strength = 0.15 * (1 - progress)
                    random_phase = 2.0 * np.pi * base_rng.random(target_amp.shape)
                    near_phase = np.angle(near)
                    perturbed_phase = near_phase + perturbation_strength * (random_phase - near_phase)
                    near = aperture_amp * np.exp(1j * perturbed_phase)
                    stagnation = 0
                    if progress > 0.7:
                        break
            else:
                stagnation = 0
            prev_score = total_score

            # Kim's weight update with momentum
            amp_ratio = target_amp_norm / (current_amp + 1e-12)
            weight_update = np.power(np.clip(amp_ratio, 0.1, 10.0), gamma)

            # Momentum smoothing
            weight_momentum = momentum_alpha * weight_momentum + (1 - momentum_alpha) * weight_update
            new_weight = weight * (1 + 0.10 * weight_momentum)
            weight = smoothing * weight + (1 - smoothing) * new_weight

            # Adaptive weight bounds - higher bright weights for better pattern fidelity
            dark_upper = 0.05 - 0.03 * progress  # Tighter upper bound
            dark_lower = 0.003  # Lower minimum for dark regions
            weight[bright_mask] = np.clip(weight[bright_mask], 2.5, 10.0)
            weight[mid_mask] = np.clip(weight[mid_mask], 0.5, 4.0)
            weight[dark_mask] = np.clip(weight[dark_mask], dark_lower, dark_upper)

        # Phase 3: Multi-scale block-based refinement
        best_phase, best_score = block_refinement(best_phase, num_blocks=60, initial_block_size=8)

        # Phase 4: Pixel-level fine tuning
        best_phase, best_score = pixel_refinement(best_phase, num_pixels=150)

        return best_phase, best_score

    # Multi-start with diverse strategies and multiple attempts per strategy
    strategies = ['gs_wgs', 'gs_heavy', 'uniform']
    best_overall_phase = None
    best_overall_score = 0.0

    for strategy in strategies:
        # Two attempts per strategy with different seeds
        for attempt in range(2):
            start_seed = base_rng.integers(0, 2**31)
            phase, score = run_optimization(start_seed, strategy)

            if score > best_overall_score:
                best_overall_score = score
                best_overall_phase = phase

    return best_overall_phase


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