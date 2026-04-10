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
    cfg = dict(DEFAULT_CONFIG); cfg.update(config or {})
    return {"cfg": cfg, "x_period": np.linspace(-cfg["period_size"]/2, cfg["period_size"]/2, cfg["period_pixels"])}


def baseline_transitions(problem: Dict[str, Any]) -> np.ndarray:
    p = float(problem["cfg"]["period_size"])
    x = np.array(
        [0.0, 0.201181, 0.250978, 0.326167, 0.370555, 0.372996, 0.396478,
         0.453128, 0.594731, 0.670591, 0.717718, 0.890632, 0.919921, 0.935546],
        dtype=float,
    )
    return (x - 0.5) * p


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
    cfg = problem["cfg"]

    def project(t: np.ndarray) -> np.ndarray:
        lim = 0.49 * cfg["period_size"]
        gap = 0.012 * cfg["period_size"]
        t = np.clip(np.sort(np.asarray(t, float)), -lim, lim)
        for i in range(1, len(t)):
            if t[i] < t[i - 1] + gap:
                t[i] = t[i - 1] + gap
        return np.clip(t, -lim, lim)

    def obj(m: Dict[str, Any]) -> float:
        u = np.clip(1.0 - m["cv_orders"] / 0.9, 0.0, 1.0)
        e = np.clip((m["efficiency"] - 0.003) / 0.177, 0.0, 1.0)
        b = np.clip((m["min_to_max"] - 0.15) / 0.75, 0.0, 1.0)
        return float(0.60 * u + 0.30 * e + 0.10 * b)

    def run(t: np.ndarray):
        t = project(t)
        f = build_incident_field(problem, t).RS(z=cfg["focal"], new_field=True, verbose=False)
        I = np.abs(f.u) ** 2
        m = evaluate_orders(problem, I, f.x)
        return obj(m), t, f, I, m

    transitions = baseline_transitions(problem)
    best_score, transitions, best_focus, best_intensity, best_metrics = run(transitions)
    n = len(transitions)

    for step in cfg["period_size"] * np.array([0.02, 0.01, 0.005, 0.0025]):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                base = transitions[i]
                best_local = None
                for dx in (-step, step):
                    cand = transitions.copy()
                    cand[i] = base + dx
                    score, cand, focus, intensity, metrics = run(cand)
                    if score > best_score and (best_local is None or score > best_local[0]):
                        best_local = (score, cand, focus, intensity, metrics)
                if best_local is not None:
                    best_score, transitions, best_focus, best_intensity, best_metrics = best_local
                    improved = True

    for step in cfg["period_size"] * np.array([0.005, 0.0025]):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    best_local = None
                    for di in (-step, step):
                        for dj in (-step, step):
                            cand = transitions.copy()
                            cand[i] += di
                            cand[j] += dj
                            score, cand, focus, intensity, metrics = run(cand)
                            if score > best_score and (best_local is None or score > best_local[0]):
                                best_local = (score, cand, focus, intensity, metrics)
                    if best_local is not None:
                        best_score, transitions, best_focus, best_intensity, best_metrics = best_local
                        improved = True

    return {
        "transitions": transitions,
        "x_focus": best_focus.x,
        "intensity_focus": best_intensity,
        "metrics": best_metrics,
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
