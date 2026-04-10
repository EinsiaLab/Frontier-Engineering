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
        "steps": 250,
        "lr": 0.08,
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
    # Initialize with larger random phases (-π to π) for better exploration, as in top-performing programs
    layers = [
        PhaseModulator(Parameter((torch.rand((shape, shape), dtype=torch.double) * 2 * torch.pi - torch.pi)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def _roi_powers(field: Field, centers: list[tuple[float, float]], radius: float) -> torch.Tensor:
    """Compute power in circular ROIs around each focus center."""
    x, y = field.meshgrid()
    intensity = field.intensity()
    powers = []
    for cx, cy in centers:
        mask = (((x - cx) ** 2 + (y - cy) ** 2) <= radius**2).to(intensity.dtype)
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

    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]) * 1.5)  # Increase initial LR as in top programs
    # Use StepLR scheduler with faster decay to converge quickly
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    losses: list[float] = []

    target_ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    target_ratios = target_ratios / target_ratios.sum()
    roi_radius = float(spec.get("roi_radius_m", spec["waist_radius"] / 4))

    for step in range(int(spec["steps"])):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])

        # Use cosine similarity for overlap loss: more stable gradient
        overlap = output_field.inner(target_field).abs()
        overlap_loss = 1.0 - overlap

        # Compute ratio error
        powers = _roi_powers(output_field, spec["focus_centers"], roi_radius)
        total_focus_power = powers.sum()
        pred_ratios = powers / (total_focus_power + 1e-12)
        ratio_loss = torch.mean(torch.abs(pred_ratios - target_ratios))

        # Compute leakage (efficiency) loss
        total_power = output_field.intensity().sum() + 1e-12
        efficiency = total_focus_power / total_power
        leakage_loss = 1.0 - efficiency

        # Add phase regularization to encourage smooth phases which can improve focusing
        phase_reg = torch.tensor(0.0, dtype=torch.double, device=device)
        for layer in system:
            dx = layer.phase[:, 1:] - layer.phase[:, :-1]
            dy = layer.phase[1:, :] - layer.phase[:-1, :]
            phase_reg = phase_reg + (dx.abs().mean() + dy.abs().mean())
        
        # Weighted loss: prioritize efficiency (leakage_loss) which has highest exponent in score
        # Adjust weights based on reference solver but with more focus on efficiency
        # Use weights: 0.25 overlap, 0.15 ratio, 0.9 leakage, 0.005 phase
        loss = 0.25 * overlap_loss + 0.15 * ratio_loss + 0.9 * leakage_loss + 0.005 * phase_reg

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
