# EVOLVE-BLOCK-START
"""Baseline solver for Task 2: multi-plane focusing."""

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
        "planes": [
            {
                "z": 0.48,
                "centers": [(-2.2 * waist, -1.4 * waist), (0.0, -1.9 * waist), (2.2 * waist, -1.4 * waist)],
                "ratios": [0.50, 0.30, 0.20],
            },
            {
                "z": 0.62,
                "centers": [(-2.0 * waist, 1.8 * waist), (0.0, 1.2 * waist), (2.0 * waist, 1.8 * waist)],
                "ratios": [0.20, 0.55, 0.25],
            },
            {
                "z": 0.76,
                "centers": [(-1.8 * waist, 0.0), (0.0, 0.0), (1.8 * waist, 0.0)],
                "ratios": [0.25, 0.50, 0.25],
            },
        ],
        "steps": 300,
        "lr": 0.075,
        "weight_decay": 5e-5,
        "scheduler_eta_min": 0.001,
    }


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    layers = [
        PhaseModulator(Parameter(torch.empty((shape, shape), dtype=torch.double).uniform_(-torch.pi/2, torch.pi/2)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def _build_target_field_for_plane(spec: dict[str, Any], plane_cfg: dict[str, Any], device: str) -> Field:
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])

    target = torch.zeros((shape, shape), dtype=torch.double, device=device)
    ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()

    for ratio, center in zip(ratios, plane_cfg["centers"]):
        target += torch.sqrt(ratio) * gaussian(shape, waist, offset=center).real.to(device)

    return Field(target.to(torch.cdouble), z=plane_cfg["z"]).normalize(1.0)


def _roi_powers(field: Field, centers: list[tuple[float, float]], radius: float) -> torch.Tensor:
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
    system = _build_system(spec, device)
    target_fields = [_build_target_field_for_plane(spec, p, device) for p in spec["planes"]]

    roi_radius = float(spec["roi_radius_m"])
    optimizer = torch.optim.AdamW(system.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(spec["steps"]), eta_min=float(spec["scheduler_eta_min"]))
    losses: list[float] = []

    for _ in range(int(spec["steps"])):
        optimizer.zero_grad()
        plane_losses = []
        for plane_cfg, target_field in zip(spec["planes"], target_fields):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])
            
            # Original overlap loss (field alignment)
            overlap_loss = 1.0 - output.inner(target_field).abs().square()
            
            # Ratio loss: match target power distribution between spots
            powers = _roi_powers(output, plane_cfg["centers"], roi_radius)
            focus_power = powers.sum()
            pred_ratios = powers / (focus_power + 1e-12)
            target_ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
            target_ratios = target_ratios / target_ratios.sum()
            ratio_loss = torch.mean(torch.abs(pred_ratios - target_ratios))
            
            # Leakage loss: maximize power in focus regions (improve efficiency)
            total_power = output.intensity().sum() + 1e-12
            leakage_loss = 1.0 - focus_power / total_power
            
            # Combine terms with weights aligned to scoring metric priorities
            combined_plane_loss = 0.35 * overlap_loss + 0.9 * ratio_loss + 0.6 * leakage_loss
            plane_losses.append(combined_plane_loss)

        # Dynamic weighting: prioritize underperforming planes to balance results across depths
        plane_losses_t = torch.stack(plane_losses)
        weights = torch.softmax(plane_losses_t.detach() / 0.1, dim=0)
        loss = (weights * plane_losses_t).sum()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_fields": target_fields,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END
