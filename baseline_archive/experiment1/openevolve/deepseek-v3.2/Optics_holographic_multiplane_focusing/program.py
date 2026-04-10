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
        "steps": 250,
        "lr": 0.1,
    }


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    # Create a radial phase pattern to encourage focusing
    yy, xx = torch.meshgrid(
        torch.arange(shape, dtype=torch.double),
        torch.arange(shape, dtype=torch.double),
        indexing='ij'
    )
    center = shape / 2.0
    radius = torch.sqrt((yy - center)**2 + (xx - center)**2)
    # Radial phase: quadratic to approximate lens - stronger coefficient for better focusing
    radial_phase = 0.15 * (radius**2)  # increased coefficient
    
    layers = []
    for z in spec["layer_z"]:
        # Combine radial pattern with random noise, but with different patterns per layer
        # to create diversity in the phase modulation
        # Scale random noise differently per layer to avoid symmetry
        scale = 0.03 * (1.0 + 0.1 * float(z))  # reduced noise scale
        # Also add a sinusoidal pattern to help with multi-plane focusing
        # Different sinusoidal patterns per layer to create diversity for multi-plane focusing
        freq = 0.5 + 0.1 * float(z)  # vary frequency with depth
        sin_pattern = 0.05 * torch.sin(freq * radius) * torch.cos(0.3 * radius)
        base_phase = radial_phase + sin_pattern + torch.randn((shape, shape), dtype=torch.double) * scale
        layers.append(PhaseModulator(Parameter(base_phase), z=float(z)))
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
    # Use CosineAnnealingLR for smooth decay without restarts, which may be more stable
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(spec["steps"]), eta_min=0.005
    )
    losses: list[float] = []

    roi_radius = float(spec.get("roi_radius_m", 3 * spec["spacing"]))
    for step in range(int(spec["steps"])):
        optimizer.zero_grad()
        per_plane_losses = []
        for plane_cfg, target_field in zip(spec["planes"], target_fields):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])
            overlap_loss = 1.0 - output.inner(target_field).abs().square()

            # Compute ROI powers for ratio and efficiency
            x, y = output.meshgrid()
            intensity = output.intensity()
            powers = []
            for cx, cy in plane_cfg["centers"]:
                mask = (((x - cx) ** 2 + (y - cy) ** 2) <= roi_radius**2).to(intensity.dtype)
                powers.append((intensity * mask).sum())
            powers_t = torch.stack(powers)
            focus_power = powers_t.sum()
            total_power = intensity.sum() + 1e-12
            pred_ratios = powers_t / (focus_power + 1e-12)
            target_ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
            target_ratios = target_ratios / target_ratios.sum()

            ratio_loss = torch.mean(torch.abs(pred_ratios - target_ratios))
            leakage_loss = 1.0 - focus_power / total_power
            # Efficiency maximization: directly maximize focus_power/total_power
            efficiency_loss = 1.0 - focus_power / total_power
            
            # Shape matching loss: encourage similarity between normalized intensity distributions
            pred_norm = intensity / (total_power + 1e-12)
            target_intensity = target_field.intensity().to(intensity.device)
            target_norm = target_intensity / (target_intensity.sum() + 1e-12)
            shape_loss = 1.0 - torch.dot(pred_norm.flatten(), target_norm.flatten()) / (
                torch.norm(pred_norm.flatten()) * torch.norm(target_norm.flatten()) + 1e-12)
            
            # Composite loss with weights aligned to scoring formula: efficiency^0.50, ratio^0.35, shape^0.15
            # Overlap loss is not directly scored, so weight it lower
            # Adjust weights slightly to better balance objectives
            plane_loss = 0.08 * overlap_loss + 0.32 * ratio_loss + 0.55 * efficiency_loss + 0.20 * shape_loss
            per_plane_losses.append(plane_loss)

        per_plane_losses_t = torch.stack(per_plane_losses)
        # Dynamic weighting: focus on harder planes
        weights = torch.softmax(per_plane_losses_t.detach() / 0.15, dim=0)
        loss = (weights * per_plane_losses_t).sum()
        
        # Add phase regularization to prevent extreme values
        reg_loss = 0.0
        for layer in system:
            if isinstance(layer, PhaseModulator):
                phase = layer.phase.data
                # Encourage smooth phases
                dx = torch.diff(phase, dim=0).abs().mean()
                dy = torch.diff(phase, dim=1).abs().mean()
                reg_loss += dx + dy
        loss += 0.005 * reg_loss
        
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
        
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
