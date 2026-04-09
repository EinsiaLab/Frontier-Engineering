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
        "steps": 120,
        "lr": 0.09,
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


def _build_plane_masks(spec: dict[str, Any], plane_cfg: dict[str, Any], device: str, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])
    r2 = float(spec.get("roi_radius_m", 3 * spec["spacing"])) ** 2
    spot_masks = torch.stack([
        (g := gaussian(shape, waist * 0.34, offset=c).real.to(device=device, dtype=torch.double)) / (g.sum() + 1e-12)
        for c in plane_cfg["centers"]
    ])
    roi_masks = torch.stack([(((x - cx) ** 2 + (y - cy) ** 2) <= r2).to(torch.double) for cx, cy in plane_cfg["centers"]])
    return spot_masks, roi_masks


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    system = _build_system(spec, device)
    target_fields = [_build_target_field_for_plane(spec, p, device) for p in spec["planes"]]
    n = int(spec["shape"])
    axis = (torch.arange(n, device=device, dtype=torch.double) - (n - 1) / 2) * float(spec["spacing"])
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    plane_data = [_build_plane_masks(spec, p, device, x, y) for p in spec["planes"]]
    plane_ratios = []
    for p in spec["planes"]:
        t = torch.tensor(p["ratios"], dtype=torch.double, device=device)
        plane_ratios.append(t / t.sum())

    steps = int(spec["steps"])
    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    losses: list[float] = []

    for i in range(steps):
        optimizer.zero_grad()
        per_plane = []
        for plane_cfg, target_field, (spot_masks, roi_masks), ratios in zip(spec["planes"], target_fields, plane_data, plane_ratios):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])
            intensity = output.data.abs().square().real.to(torch.double)
            total_power = intensity.sum() + 1e-12
            pred_norm = intensity / total_power
            tgt = target_field.intensity().to(device=device, dtype=torch.double)
            tgt_norm = tgt / (tgt.sum() + 1e-12)

            spot_powers = (spot_masks * intensity.unsqueeze(0)).sum(dim=(-1, -2))
            focus_soft = spot_powers.sum()
            pred_ratios = spot_powers / (focus_soft + 1e-12)

            roi_powers = (roi_masks * intensity.unsqueeze(0)).sum(dim=(-1, -2))
            focus_roi = roi_powers.sum()
            roi_ratios = roi_powers / (focus_roi + 1e-12)

            ratio_loss = (pred_ratios - ratios).square().mean().sqrt()
            roi_ratio_loss = (roi_ratios - ratios).abs().mean()
            overlap_loss = 1.0 - output.inner(target_field).abs().square()
            shape_loss = (pred_norm - tgt_norm).abs().mean()
            leakage_loss = 1.0 - focus_roi / total_power
            eff_loss = 1.0 - 0.5 * (focus_soft.clamp(0.0, 1.0) + focus_roi.clamp(0.0, 1.0))

            a = i / max(1, steps - 1)
            per_plane.append(
                (0.74 - 0.14 * a) * ratio_loss
                + 0.10 * roi_ratio_loss
                + (0.08 + 0.08 * a) * overlap_loss
                + 0.05 * shape_loss
                + (0.26 + 0.08 * a) * leakage_loss
                + 0.05 * eff_loss
            )

        per_plane_t = torch.stack(per_plane)
        a = i / max(1, steps - 1)
        temp = 0.18 - 0.06 * a
        loss = (torch.softmax(per_plane_t.detach() / temp, dim=0) * per_plane_t).sum()
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
