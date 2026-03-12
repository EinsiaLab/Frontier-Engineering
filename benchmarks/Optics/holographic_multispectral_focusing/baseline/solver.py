"""Baseline solver for Task 3: multi-wavelength focusing/splitting."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Parameter

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import PolychromaticPhaseModulator
from torchoptics.profiles import gaussian


def make_default_spec() -> dict[str, Any]:
    waist = 130e-6
    return {
        "shape": 72,
        "spacing": 10e-6,
        "wavelengths": [450e-9, 520e-9, 590e-9, 660e-9],
        "waist_radius": waist,
        "layer_z": [0.0, 0.18, 0.36],
        "output_z": 0.62,
        "target_centers": [
            (-2.4 * waist, -0.8 * waist),
            (-0.8 * waist, 1.8 * waist),
            (0.9 * waist, -1.8 * waist),
            (2.3 * waist, 0.8 * waist),
        ],
        "target_spectral_ratios": [0.30, 0.24, 0.26, 0.20],
        "steps": 180,
        "lr": 0.075,
    }


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    layers = [
        PolychromaticPhaseModulator(Parameter(torch.zeros((shape, shape), dtype=torch.double)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def _make_input_fields(spec: dict[str, Any], device: str) -> list[Field]:
    fields = []
    for wl in spec["wavelengths"]:
        field = Field(gaussian(spec["shape"], spec["waist_radius"]), wavelength=wl, z=0).normalize(1.0)
        fields.append(field.to(device))
    return fields


def _roi_power(field: Field, center: tuple[float, float], radius: float) -> torch.Tensor:
    x, y = field.meshgrid()
    intensity = field.intensity()
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius**2
    return (intensity * mask.to(intensity.dtype)).sum()


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelengths"][1])

    roi_radius = float(spec.get("roi_radius_m", 4 * spec["spacing"]))

    system = _build_system(spec, device)
    input_fields = _make_input_fields(spec, device)

    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    losses: list[float] = []

    for _ in range(int(spec["steps"])):
        optimizer.zero_grad()

        per_wavelength_losses = []
        for field, center in zip(input_fields, spec["target_centers"]):
            out = system.measure_at_z(field, z=spec["output_z"])
            target_power = _roi_power(out, center, roi_radius)
            total_power = out.intensity().sum() + 1e-12
            per_wavelength_losses.append(1.0 - target_power / total_power)

        loss = torch.stack(per_wavelength_losses).mean()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return {
        "spec": spec,
        "system": system,
        "input_fields": input_fields,
        "loss_history": losses,
    }
