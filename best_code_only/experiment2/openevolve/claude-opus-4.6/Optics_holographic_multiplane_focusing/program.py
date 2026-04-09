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
        "steps": 350,
        "lr": 0.15,
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

    roi_radius = float(spec.get("roi_radius_m", 3 * spec["spacing"]))

    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    num_steps = int(spec["steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-3)
    losses: list[float] = []

    for step in range(num_steps):
        optimizer.zero_grad()
        per_plane_losses = []
        for plane_cfg, target_field in zip(spec["planes"], target_fields):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])

            # Overlap loss
            overlap_loss = 1.0 - output.inner(target_field).abs().square()

            # ROI-based ratio and leakage losses
            x, y = output.meshgrid()
            intensity = output.intensity()
            roi_powers = []
            for cx, cy in plane_cfg["centers"]:
                mask = (((x - cx) ** 2 + (y - cy) ** 2) <= roi_radius ** 2).to(intensity.dtype)
                roi_powers.append((intensity * mask).sum())
            roi_powers_t = torch.stack(roi_powers)
            focus_power = roi_powers_t.sum()
            total_power = intensity.sum() + 1e-12

            pred_ratios = roi_powers_t / (focus_power + 1e-12)
            target_ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
            target_ratios = target_ratios / target_ratios.sum()

            ratio_loss = torch.mean(torch.abs(pred_ratios - target_ratios))
            leakage_loss = 1.0 - focus_power / total_power

            # Intensity-based shape loss: directly compare normalized intensity patterns
            target_intensity = target_field.intensity().to(intensity.device)
            target_inorm = target_intensity / (target_intensity.sum() + 1e-12)
            pred_inorm = intensity / (total_power)
            shape_loss = 1.0 - torch.sum(pred_inorm * target_inorm) / (
                torch.sqrt(torch.sum(pred_inorm ** 2) * torch.sum(target_inorm ** 2)) + 1e-12
            )

            # Weighted combination emphasizing efficiency and shape
            per_plane_losses.append(
                0.30 * overlap_loss + 0.60 * ratio_loss + 0.55 * leakage_loss + 0.25 * shape_loss
            )

        # Dynamic reweighting: harder planes get more weight
        per_plane_t = torch.stack(per_plane_losses)
        weights = torch.softmax(per_plane_t.detach() / 0.15, dim=0)
        loss = (weights * per_plane_t).sum()

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
