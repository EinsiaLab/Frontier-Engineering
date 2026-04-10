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
        "steps": 120,
        "lr": 0.065,
        "roi_radius_m": 30e-6,
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


def _roi_powers(field: Field, centers: list[tuple[float, float]], radius: float, masks: list | None = None) -> torch.Tensor:
    intensity = field.intensity()
    if masks is not None:
        powers = [(intensity * m).sum() for m in masks]
        return torch.stack(powers)
    x, y = field.meshgrid()
    powers = []
    for cx, cy in centers:
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).to(intensity.dtype)
        powers.append((intensity * mask).sum())
    return torch.stack(powers)


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    target_field = _build_target_field(spec, device)
    system = _build_system(spec, device)

    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    losses: list[float] = []

    target_ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    target_ratios = target_ratios / target_ratios.sum()
    roi_radius = float(spec.get("roi_radius_m", 30e-6))

    # Precompute masks (avoids repeated meshgrid inside loop)
    x, y = input_field.meshgrid()
    masks = []
    for cx, cy in spec["focus_centers"]:
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= roi_radius**2).to(torch.double)
        masks.append(mask)

    for _ in range(int(spec["steps"])):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])

        overlap = output_field.inner(target_field).abs().square()
        overlap_loss = 1.0 - overlap

        powers = _roi_powers(output_field, spec["focus_centers"], roi_radius, masks=masks)
        focus_power = powers.sum()
        total_power = output_field.intensity().sum() + 1e-12
        ratio_hat = powers / (focus_power + 1e-12)
        ratio_loss = torch.mean(torch.abs(ratio_hat - target_ratios))
        leakage_loss = 1.0 - focus_power / total_power

        # Balanced composite loss (improves ratio accuracy + efficiency vs pure overlap)
        loss = 0.6 * overlap_loss + 1.8 * ratio_loss + 0.4 * leakage_loss

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
