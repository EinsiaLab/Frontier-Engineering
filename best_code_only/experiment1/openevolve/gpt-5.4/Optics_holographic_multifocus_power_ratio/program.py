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
        "roi_radius_m": 0.9 * waist,
        "steps": 180,
        "lr": 0.075,
    }


def _build_target_field(spec: dict[str, Any], device: str) -> Field:
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])
    target = sum(
        torch.sqrt(torch.tensor(r, dtype=torch.double, device=device))
        * gaussian(shape, waist, offset=c).real.to(device)
        for r, c in zip(spec["focus_ratios"], spec["focus_centers"])
    )
    return Field(target.to(torch.cdouble), z=spec["output_z"]).normalize(1.0)


def _roi_masks(spec: dict[str, Any], field: Field) -> torch.Tensor:
    x, y = field.meshgrid()
    r2 = float(spec["roi_radius_m"]) ** 2
    return torch.stack([(((x - cx) ** 2 + (y - cy) ** 2) <= r2).to(torch.double) for cx, cy in spec["focus_centers"]])


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    layers = [
        PhaseModulator(Parameter(torch.zeros((shape, shape), dtype=torch.double)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    shape, waist = int(spec["shape"]), float(spec["waist_radius"])
    input_field = Field(gaussian(shape, waist), z=0).normalize(1.0).to(device)
    target_field = _build_target_field(spec, device)
    system = _build_system(spec, device)
    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    losses: list[float] = []

    masks = _roi_masks(spec, input_field)
    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()
    steps = int(spec["steps"])

    for i in range(steps):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])
        intensity = output_field.intensity().to(torch.double)
        powers = (intensity.unsqueeze(0) * masks).sum(dim=(-2, -1))
        focus_power = powers.sum()
        total_power = intensity.sum() + 1e-12
        norm_powers = powers / (focus_power + 1e-12)

        ratio_loss = (norm_powers - ratios).square().mean()
        leakage_loss = 1.0 - focus_power / total_power
        overlap_loss = 1.0 - output_field.inner(target_field).abs().square()

        t = i / max(1, steps - 1)
        loss = (0.2 + 0.5 * t) * ratio_loss + (0.8 - 0.3 * t) * leakage_loss + 0.01 * overlap_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_field": target_field,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END
