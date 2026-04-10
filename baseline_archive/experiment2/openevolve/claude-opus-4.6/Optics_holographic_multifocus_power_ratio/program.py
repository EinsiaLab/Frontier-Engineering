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
        "steps": 24,
        "lr": 0.22,
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


def _roi_powers(field: Field, centers, radius: float) -> torch.Tensor:
    x, y = field.meshgrid()
    intensity = field.intensity()
    powers = []
    for cx, cy in centers:
        mask = (((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2).to(intensity.dtype)
        powers.append((intensity * mask).sum())
    return torch.stack(powers)


def _build_system(spec: dict[str, Any], device: str, seed: int = 0) -> System:
    shape = int(spec["shape"])
    layers = [
        PhaseModulator(
            Parameter(torch.zeros((shape, shape), dtype=torch.double)),
            z=float(z),
        )
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
    system = _build_system(spec, device, seed=seed)

    roi_radius = float(spec.get("roi_radius_m", 3 * spec["spacing"]))
    target_ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    target_ratios = target_ratios / target_ratios.sum()

    steps = int(spec["steps"])
    lr = float(spec["lr"])
    optimizer = torch.optim.Adam(system.parameters(), lr=lr, betas=(0.85, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.02)
    losses: list[float] = []

    for step in range(steps):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])

        overlap = output_field.inner(target_field).abs().square()
        overlap_loss = 1.0 - overlap

        powers = _roi_powers(output_field, spec["focus_centers"], roi_radius)
        focus_power = powers.sum()
        total_power = output_field.intensity().sum() + 1e-12
        efficiency = focus_power / total_power
        ratio_hat = powers / (focus_power + 1e-12)
        ratio_loss = torch.mean(torch.abs(ratio_hat - target_ratios))
        leakage_loss = 1.0 - efficiency
        eff_log_loss = -torch.log(efficiency + 1e-8)

        # Phase smoothness regularizer
        phase_reg = torch.tensor(0.0, dtype=torch.double, device=device)
        for layer in system:
            ph = layer.phase
            dx = ph[:, 1:] - ph[:, :-1]
            dy = ph[1:, :] - ph[:-1, :]
            phase_reg = phase_reg + (dx.abs().mean() + dy.abs().mean())

        progress = step / max(steps - 1, 1)
        w_over = 0.50 * (1.0 - 0.6 * progress)
        w_leak = 2.5
        w_eff_log = 0.30
        w_ratio = 0.32 + 0.42 * progress
        w_smooth = 5e-4
        loss = w_over * overlap_loss + w_ratio * ratio_loss + w_leak * leakage_loss + w_eff_log * eff_log_loss + w_smooth * phase_reg

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
