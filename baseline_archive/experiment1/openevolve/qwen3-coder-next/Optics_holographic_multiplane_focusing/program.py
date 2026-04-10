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
        "steps": 180,
        "lr": 0.075,
    }


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    layers = []
    for i, z in enumerate(spec["layer_z"]):
        # Initialize with random phases for better exploration
        init_phases = torch.rand((shape, shape), dtype=torch.double) * 2 * torch.pi
        layers.append(PhaseModulator(Parameter(init_phases), z=float(z)))
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


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    system = _build_system(spec, device)
    target_fields = [_build_target_field_for_plane(spec, p, device) for p in spec["planes"]]

    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(spec["steps"]))
    losses: list[float] = []

    # ROI radius for efficiency calculation
    roi_radius = float(spec.get("roi_radius_m", 3 * spec["spacing"]))
    steps = int(spec["steps"])
    
    for step in range(steps):
        optimizer.zero_grad()
        
        per_plane_losses = []
        for plane_cfg, target_field in zip(spec["planes"], target_fields):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])
            
            # Overlap loss (existing)
            overlap_loss = 1.0 - output.inner(target_field).abs().square()
            
            # Calculate powers in ROI regions for ratio and efficiency
            x, y = output.meshgrid()
            intensity = output.intensity()
            powers = []
            for cx, cy in plane_cfg["centers"]:
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= roi_radius**2
                powers.append((intensity * mask.to(intensity.dtype)).sum())
            powers_t = torch.stack(powers)
            
            focus_power = powers_t.sum()
            total_power = intensity.sum() + 1e-12
            
            # Ratio loss (new)
            pred_ratios = powers_t / (focus_power + 1e-12)
            target_ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
            target_ratios = target_ratios / target_ratios.sum()
            ratio_loss = torch.mean(torch.abs(pred_ratios - target_ratios))
            
            # Leakage loss (new)
            leakage_loss = 1.0 - focus_power / total_power
            
            # Combine losses with weights similar to reference solver
            per_plane_losses.append(0.40 * overlap_loss + 0.95 * ratio_loss + 0.35 * leakage_loss)
        
        # Dynamic plane weighting based on loss values
        per_plane_losses_t = torch.stack(per_plane_losses)
        weights = torch.softmax(per_plane_losses_t.detach() / 0.20, dim=0)
        loss = (weights * per_plane_losses_t).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
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
