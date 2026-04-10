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


def _eval_score(intensity, target_amp, target_mask, dark_mask):
    """Evaluate using metrics matching the actual scoring formula."""
    I_n = intensity / (intensity.mean() + 1e-12)
    T_n = target_amp**2 / ((target_amp**2).mean() + 1e-12)
    nmse_val = float(np.sqrt(((I_n - T_n)**2).mean()))
    
    bright_mask = target_amp > 0.30
    energy_in = float(intensity[bright_mask].sum() / (intensity.sum() + 1e-12))
    
    dark_eval_mask = target_amp < 0.03
    leak = float(intensity[dark_eval_mask].sum() / (intensity.sum() + 1e-12))
    dark_sup = 1.0 - leak
    
    pattern_score = np.clip(1.0 - nmse_val / 4.0, 0.0, 1.0)
    energy_score = np.clip((energy_in - 0.10) / 0.60, 0.0, 1.0)
    dark_score = np.clip((dark_sup - 0.35) / 0.55, 0.0, 1.0)
    return 100.0 * (0.55 * pattern_score + 0.30 * energy_score + 0.15 * dark_score)


def _fwd(aperture, slm_phase):
    near = aperture * np.exp(1j * slm_phase)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(near), norm="ortho"))


def _bwd(Uz):
    return np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Uz), norm="ortho")))


def _run_wgs_kim(target_amp, aperture, rng, target_mask, dark_mask,
                  num_iters=300, feedback_exp=0.78, init_phase=None):
    """WGS-Kim style iterative algorithm."""
    target_for_opt = np.maximum(target_amp, 1e-4)
    if init_phase is not None:
        slm_phase = init_phase.copy()
    else:
        far_phase = 2.0 * np.pi * rng.random(target_amp.shape)
        slm_phase = _bwd(target_for_opt * np.exp(1j * far_phase))
    weights = np.ones_like(target_amp)
    best_phase, best_score = slm_phase.copy(), -1e30

    for iteration in range(num_iters):
        far = _fwd(aperture, slm_phase)
        far_amp = np.abs(far)
        far_phase_cur = np.angle(far)

        if iteration % 8 == 0 or iteration == num_iters - 1:
            score = _eval_score(far_amp ** 2, target_amp, target_mask, dark_mask)
            if score > best_score:
                best_score, best_phase = score, slm_phase.copy()

        if iteration > 0 and iteration % max(1, num_iters // 4) == 0:
            weights = np.power(weights, 0.5)
        weights *= np.power(target_for_opt / (far_amp + 1e-12), feedback_exp)
        weights = np.clip(weights / (weights.mean() + 1e-12), 0.03, 20.0)
        new_far_amp = weights * target_for_opt
        progress = iteration / num_iters
        new_far_amp[dark_mask] = far_amp[dark_mask] * max(0.0002, 0.35 * (1.0 - progress) ** 1.8)
        slm_phase = _bwd(new_far_amp * np.exp(1j * far_phase_cur))

    score = _eval_score(np.abs(_fwd(aperture, slm_phase)) ** 2, target_amp, target_mask, dark_mask)
    if score > best_score:
        best_score, best_phase = score, slm_phase.copy()
    return best_phase, best_score


def _run_wgs_with_scaling(target_amp, aperture, rng, target_mask, dark_mask,
                           num_iters=300, feedback_exp=0.78):
    """WGS with energy concentration outside bright regions."""
    target_for_opt = np.maximum(target_amp, 1e-4)
    far_phase = 2.0 * np.pi * rng.random(target_amp.shape)
    slm_phase = _bwd(target_for_opt * np.exp(1j * far_phase))
    weights = np.ones_like(target_amp)
    best_phase, best_score = slm_phase.copy(), -1e30
    outside_bright = ~target_mask & ~dark_mask

    for iteration in range(num_iters):
        far = _fwd(aperture, slm_phase)
        far_amp = np.abs(far)
        far_phase_cur = np.angle(far)

        if iteration % 8 == 0 or iteration == num_iters - 1:
            score = _eval_score(far_amp ** 2, target_amp, target_mask, dark_mask)
            if score > best_score:
                best_score, best_phase = score, slm_phase.copy()

        progress = iteration / num_iters
        if iteration > 0 and iteration % (num_iters // 3) == 0:
            weights = np.ones_like(target_amp)
        weights *= np.power(target_for_opt / (far_amp + 1e-12), feedback_exp)
        weights = np.clip(weights / (weights.mean() + 1e-12), 0.05, 15.0)
        new_far_amp = weights * target_for_opt
        new_far_amp[dark_mask] = far_amp[dark_mask] * max(0.0002, 0.3 * (1.0 - progress) ** 2)
        if progress > 0.25:
            new_far_amp[outside_bright] *= max(0.4, 1.0 - 0.5 * progress)
        slm_phase = _bwd(new_far_amp * np.exp(1j * far_phase_cur))

    score = _eval_score(np.abs(_fwd(aperture, slm_phase)) ** 2, target_amp, target_mask, dark_mask)
    if score > best_score:
        best_score, best_phase = score, slm_phase.copy()
    return best_phase, best_score


def solve_baseline(problem: Dict[str, Any], seed: int | None = None) -> np.ndarray:
    """WGS-Kim style iterative weighted GS for holography."""
    seed_value = int(problem["cfg"]["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed_value)

    target_amp = problem["target_amp"]
    aperture = problem["aperture_amp"]
    n = target_amp.shape[0]

    target_mask = target_amp > 0.30
    c = (n - 1) / 2.0
    y, x = np.indices((n, n))
    dark_mask = target_amp < 0.03

    best_phase, best_score = None, -1e30
    base_seeds = rng.integers(0, 2**31, size=20)
    idx = 0

    for ni, fe in [(600, 0.78), (600, 0.60), (500, 0.50), (500, 0.92),
                    (500, 0.70), (400, 0.42), (400, 0.85)]:
        r = np.random.default_rng(base_seeds[idx]); idx += 1
        phase, score = _run_wgs_kim(target_amp, aperture, r, target_mask, dark_mask, ni, fe)
        if score > best_score:
            best_score, best_phase = score, phase.copy()

    for ni, fe in [(600, 0.78), (500, 0.55), (500, 0.90)]:
        r = np.random.default_rng(base_seeds[idx]); idx += 1
        phase, score = _run_wgs_with_scaling(target_amp, aperture, r, target_mask, dark_mask, ni, fe)
        if score > best_score:
            best_score, best_phase = score, phase.copy()

    if best_phase is not None:
        for fe in [0.78, 0.55, 0.85]:
            r = np.random.default_rng(base_seeds[idx % len(base_seeds)]); idx += 1
            phase, score = _run_wgs_kim(target_amp, aperture, r, target_mask, dark_mask, 300, fe, init_phase=best_phase)
            if score > best_score:
                best_score, best_phase = score, phase.copy()

    return best_phase


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
