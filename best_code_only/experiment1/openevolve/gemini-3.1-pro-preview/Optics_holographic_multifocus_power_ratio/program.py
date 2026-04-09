# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: multifocus with target power ratios."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Parameter

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import PhaseModulator
from torchoptics.profiles import gaussian


def make_default_spec() -> dict[str, Any]:
    waist = 130e-6
    return {
        "shape": 72,
        "spacing": 10e-6,
        "wavelength": 700e-9,
        "waist_radius": waist,
        "layer_z": [0.0, 0.12, 0.24, 0.36],
        "output_z": 0.56,
        "focus_centers": [
            (-2.3 * waist, -1.6 * waist),
            (0.0, -2.3 * waist),
            (2.3 * waist, -1.6 * waist),
            (-2.3 * waist, 1.6 * waist),
            (0.0, 2.3 * waist),
            (2.3 * waist, 1.6 * waist),
        ],
        "focus_ratios": [0.24, 0.17, 0.16, 0.15, 0.14, 0.14],
        "steps": 300,
        "lr": 0.15,
    }


def _build_target_field(spec: dict[str, Any], device: str) -> Field:
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])
    target = torch.zeros((shape, shape), dtype=torch.double, device=device)

    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()

    for ratio, center in zip(ratios, spec["focus_centers"]):
        target += torch.sqrt(ratio) * gaussian(shape, waist, offset=center).real.to(device)

    return Field(target.to(torch.cdouble), z=spec["output_z"]).normalize(1.0)


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    layers = [
        PhaseModulator(Parameter(torch.zeros((shape, shape), dtype=torch.double)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    spec["steps"] = 250  # Override the default low step count for better convergence
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    target_field = _build_target_field(spec, device)
    system = _build_system(spec, device)

    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(spec["steps"]))
    losses: list[float] = []

    roi_radius = float(spec.get("roi_radius_m", 3 * spec["spacing"]))
    target_ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    target_ratios = target_ratios / target_ratios.sum()

    for _ in range(int(spec["steps"])):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])

        overlap = output_field.inner(target_field).abs().square()
        overlap_loss = 1.0 - overlap

        x, y = output_field.meshgrid()
        intensity = output_field.intensity()
        powers = []
        for cx, cy in spec["focus_centers"]:
            mask = (((x - cx) ** 2 + (y - cy) ** 2) <= roi_radius**2).to(intensity.dtype)
            powers.append((intensity * mask).sum())
        powers_t = torch.stack(powers)
        
        focus_power = powers_t.sum()
        total_power = intensity.sum() + 1e-12
        ratio_hat = powers_t / (focus_power + 1e-12)
        
        ratio_loss = torch.mean(torch.abs(ratio_hat - target_ratios))
        leakage_loss = 1.0 - focus_power / total_power

        loss = 0.3 * overlap_loss + 2.0 * ratio_loss + 0.8 * leakage_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_field": target_field,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END
