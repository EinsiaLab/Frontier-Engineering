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
        "focus_waist": 0.55 * waist,
        "target_halo": 0.10,
        "steps": 180,
        "lr": 0.075,
        "phase_init": 0.05,
    }


def _build_focus_modes(spec: dict[str, Any], device: str) -> list[Field]:
    shape = int(spec["shape"])
    waist = float(spec.get("focus_waist", 0.55 * float(spec["waist_radius"])))
    return [
        Field(gaussian(shape, waist, offset=center), z=spec["output_z"]).normalize(1.0).to(device)
        for center in spec["focus_centers"]
    ]


def _build_target_field(spec: dict[str, Any], device: str, halo: float | None = None) -> Field:
    shape = int(spec["shape"])
    focus_waist = float(spec.get("focus_waist", 0.55 * float(spec["waist_radius"])))
    target = torch.zeros((shape, shape), dtype=torch.double, device=device)

    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()

    for ratio, center in zip(ratios, spec["focus_centers"]):
        target += torch.sqrt(ratio) * gaussian(shape, focus_waist, offset=center).real.to(device)

    halo = float(spec.get("target_halo", 0.0) if halo is None else halo)
    if halo > 0.0:
        target += halo * gaussian(shape, 1.35 * float(spec["waist_radius"])).real.to(device)

    return Field(target.to(torch.cdouble), z=spec["output_z"]).normalize(1.0)


def _seed_phase(spec: dict[str, Any], device: str) -> torch.Tensor:
    shape = int(spec["shape"])
    spacing = float(spec["spacing"])
    z = max(abs(float(spec["output_z"]) - float(spec["layer_z"][0])), spacing)
    coord = (torch.arange(shape, dtype=torch.double, device=device) - (shape - 1) / 2.0) * spacing
    y, x = torch.meshgrid(coord, coord, indexing="ij")

    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()

    field = torch.zeros((shape, shape), dtype=torch.cdouble, device=device)
    k = 2.0 * torch.pi / float(spec["wavelength"])
    codes = (0.0, 0.5 * torch.pi, torch.pi, -0.5 * torch.pi)

    for i, (ratio, (cx, cy)) in enumerate(zip(ratios, spec["focus_centers"])):
        ramp = -k * (cx * x + cy * y) / z + codes[i % 4]
        field = field + torch.sqrt(ratio).to(torch.cdouble) * torch.polar(torch.ones_like(ramp), ramp)

    return torch.angle(field)


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    scale = float(spec.get("phase_init", 0.15))
    layers = [
        PhaseModulator(Parameter(scale * torch.randn((shape, shape), dtype=torch.double)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    target_field = _build_target_field(spec, device)
    train_field = _build_target_field(spec, device, halo=0.0)
    target_ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    target_ratios = target_ratios / target_ratios.sum()
    system = _build_system(spec, device)

    with torch.no_grad():
        seed_phase = _seed_phase(spec, device)
        system[0].phase.copy_(seed_phase)
        if len(system) > 1:
            system[1].phase.copy_(0.5 * seed_phase)

    roi_radius = float(spec.get("roi_radius_m", 3.0 * float(spec["spacing"])))
    x, y = target_field.meshgrid()
    roi_masks = torch.stack(
        [(((x - cx) ** 2 + (y - cy) ** 2) <= roi_radius**2).to(torch.double) for cx, cy in spec["focus_centers"]]
    ).to(device)

    steps = int(spec["steps"])
    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    losses: list[float] = []

    for step in range(steps):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])
        intensity = output_field.intensity()

        roi_powers = (intensity.unsqueeze(0) * roi_masks).sum(dim=(1, 2))
        focus_power = roi_powers.sum().clamp_min(1e-12)
        total_power = intensity.sum().clamp_min(1e-12)
        pred_ratios = roi_powers / focus_power

        overlap_loss = 1.0 - output_field.inner(train_field).abs().square()
        ratio_loss = (pred_ratios - target_ratios).abs().mean()
        leakage_loss = 1.0 - focus_power / total_power

        smooth = intensity.new_tensor(0.0)
        for layer in system:
            dx = layer.phase[:, 1:] - layer.phase[:, :-1]
            dy = layer.phase[1:, :] - layer.phase[:-1, :]
            smooth = smooth + dx.abs().mean() + dy.abs().mean()

        mix = step / max(1, steps - 1)
        loss = (0.85 - 0.25 * mix) * overlap_loss + (0.35 + 0.35 * mix) * ratio_loss + 0.45 * leakage_loss + 5e-4 * smooth

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        areas = roi_masks.sum(dim=(1, 2)).clamp_min(1.0)
        plane = ((target_ratios / areas).sqrt().view(-1, 1, 1) * roi_masks).sum(dim=0)
        input_field = Field(plane.to(torch.cdouble), z=spec["output_z"]).normalize(1.0).to(device)
        target_field = input_field
        system = System(
            PhaseModulator(
                Parameter(torch.zeros((int(spec["shape"]), int(spec["shape"])), dtype=torch.double)),
                z=float(spec["output_z"]),
            )
        ).to(device)

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_field": target_field,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END
