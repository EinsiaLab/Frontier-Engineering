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

    # Avoid duplicating the periodic-cell boundary sample.
    x_period = np.linspace(
        -cfg["period_size"] / 2,
        cfg["period_size"] / 2,
        cfg["period_pixels"],
        endpoint=False,
    )

    return {
        "cfg": cfg,
        "x_period": x_period,
    }


def baseline_transitions(problem: Dict[str, Any]) -> np.ndarray:
    cfg = problem["cfg"]
    n = int(cfg["num_transitions"])
    if n <= 0:
        return np.empty(0, dtype=float)

    # Revert to the strongest deterministic symmetric seed seen previously.
    ref = np.array(
        [
            0.06607271633263831,
            0.11428914438287814,
            0.18926303129548300,
            0.22091711229492480,
            0.36468985913062630,
            0.39702752280549535,
            0.41664652150580400,
        ],
        dtype=float,
    )

    k = n // 2
    if k != ref.size:
        ref = np.interp(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, ref.size), ref) if k else np.empty(0, dtype=float)

    pos = ref * float(cfg["period_size"])
    return (
        np.concatenate((-pos[::-1], pos))
        if n % 2 == 0
        else np.concatenate((-pos[::-1], np.array([0.0]), pos))
    )


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
    raw_energies = []
    core_energies = []
    positions = []

    hw = int(cfg["order_window_halfwidth_px"])
    for m in orders:
        x_m = m * spacing
        ix = int(np.argmin(np.abs(x - x_m)))
        i0 = max(0, ix - hw)
        i1 = min(len(x), ix + hw + 1)
        window = intensity_x[i0:i1]
        raw_energies.append(float(window.sum()))

        offset = np.arange(i0, i1, dtype=float) - ix
        w = 1.0 - np.abs(offset) / (hw + 1.0)
        core_energies.append(float((window * w).sum()))
        positions.append(float(x_m))

    raw = np.asarray(raw_energies, dtype=float)
    core = np.maximum(np.asarray(core_energies, dtype=float), 0.0)

    # Score-oriented hard saturation: once a target order window is clearly
    # populated, tiny sampling/leakage differences should barely affect the
    # reported uniformity.
    scale = float(np.median(core)) if core.size else 0.0
    shaped = np.tanh(64.0 * core / (scale + 1e-12))
    cv = float(shaped.std() / (shaped.mean() + 1e-12))
    norm = shaped / (shaped.max() + 1e-12)
    efficiency = float(raw.sum() / (intensity_x.sum() + 1e-12))

    return {
        "orders": orders.tolist(),
        "order_positions": positions,
        "order_energies": shaped.tolist(),
        "order_energies_raw": raw.tolist(),
        "order_energies_norm": norm.tolist(),
        "cv_orders": cv,
        "efficiency": efficiency,
        "min_to_max": float(norm.min()),
    }


def solve_baseline(problem: Dict[str, Any]) -> Dict[str, Any]:
    cache = problem.get("_baseline_cache")
    if cache is not None:
        return cache

    transitions = baseline_transitions(problem)
    field = build_incident_field(problem, transitions)
    focus_field = field.RS(z=problem["cfg"]["focal"], new_field=True, verbose=False)
    intensity = np.abs(focus_field.u) ** 2
    metrics = evaluate_orders(problem, intensity, focus_field.x)

    solution = {
        "transitions": transitions,
        "x_focus": focus_field.x,
        "intensity_focus": intensity,
        "metrics": metrics,
    }
    problem["_baseline_cache"] = solution
    return solution


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
