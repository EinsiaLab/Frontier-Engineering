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
    layers = [
        PhaseModulator(Parameter(torch.zeros((shape, shape), dtype=torch.double)), z=float(z))
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


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    system = _build_system(spec, device)
    target_fields = [_build_target_field_for_plane(spec, p, device) for p in spec["planes"]]

    # Pre-compute target intensity maps and ratio targets for custom loss
    num_planes = len(spec["planes"])
    target_intensities = []
    target_ratio_tensors = []
    for plane_cfg, target_field in zip(spec["planes"], target_fields):
        t_intensity = target_field.data.abs().square()
        t_intensity = t_intensity / (t_intensity.sum() + 1e-30)
        target_intensities.append(t_intensity)
        ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
        ratios = ratios / ratios.sum()
        target_ratio_tensors.append(ratios)

    # Use more optimization steps and a multi-phase approach
    total_steps = 600
    optimizer = torch.optim.Adam(system.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    losses: list[float] = []
    plane_weights = torch.ones(num_planes, dtype=torch.double, device=device)

    for step in range(total_steps):
        optimizer.zero_grad()
        plane_losses = []

        for pi, (plane_cfg, target_field) in enumerate(zip(spec["planes"], target_fields)):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])

            # Overlap fidelity loss
            overlap = output.inner(target_field).abs().square()
            fidelity_loss = 1.0 - overlap

            # Intensity-based shape loss (cosine similarity on intensities)
            out_intensity = output.data.abs().square()
            out_norm = out_intensity / (out_intensity.sum() + 1e-30)
            tgt_norm = target_intensities[pi]

            cosine_num = (out_norm * tgt_norm).sum()
            cosine_den = torch.sqrt((out_norm ** 2).sum() * (tgt_norm ** 2).sum() + 1e-30)
            shape_loss = 1.0 - cosine_num / cosine_den

            # Ratio loss: compute energy at each spot center
            shape_val = int(spec["shape"])
            waist = float(spec["waist_radius"])
            spacing = float(spec["spacing"])
            centers = plane_cfg["centers"]
            ratios_target = target_ratio_tensors[pi]

            spot_energies = []
            for center in centers:
                # Create a Gaussian mask centered at the spot
                mask = gaussian(shape_val, waist, offset=center).real.to(device)
                mask = mask.abs().square()
                mask = mask / (mask.sum() + 1e-30)
                energy = (out_intensity * mask).sum()
                spot_energies.append(energy)

            spot_energies_t = torch.stack(spot_energies)
            total_spot_energy = spot_energies_t.sum() + 1e-30
            pred_ratios = spot_energies_t / total_spot_energy
            ratio_loss = (pred_ratios - ratios_target).abs().mean()

            # Combined loss with evolving weights
            if step < 200:
                combined = 0.6 * fidelity_loss + 0.2 * shape_loss + 0.2 * ratio_loss
            elif step < 400:
                combined = 0.4 * fidelity_loss + 0.3 * shape_loss + 0.3 * ratio_loss
            else:
                combined = 0.3 * fidelity_loss + 0.3 * shape_loss + 0.4 * ratio_loss

            plane_losses.append(plane_weights[pi] * combined)

        loss = torch.stack(plane_losses).sum() / plane_weights.sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

        # Dynamic plane reweighting every 50 steps
        if (step + 1) % 50 == 0 and step < total_steps - 50:
            with torch.no_grad():
                plane_loss_vals = torch.tensor([float(pl.item()) for pl in plane_losses], device=device)
                # Increase weight for harder planes
                plane_weights = (plane_loss_vals / (plane_loss_vals.mean() + 1e-30)).clamp(0.5, 2.0)

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_fields": target_fields,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END
