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
        "steps": 180,
        "lr": 0.075,
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
    layers = [
        PhaseModulator(Parameter(torch.zeros((shape, shape), dtype=torch.double)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def _get_focus_rois(spec, shape, device, radius_factor=1.2):
    """Get ROI masks for each focus spot."""
    spacing = float(spec["spacing"])
    waist = float(spec["waist_radius"])
    half = shape // 2
    coords = (torch.arange(shape, dtype=torch.double, device=device) - half) * spacing
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')

    roi_radius = waist * radius_factor
    masks = []
    for center in spec["focus_centers"]:
        cy, cx = center
        dist_sq = (yy - cy)**2 + (xx - cx)**2
        mask = dist_sq < roi_radius**2
        masks.append(mask)
    return masks


def _get_soft_focus_rois(spec, shape, device, sigma_factor=1.0):
    """Get soft Gaussian ROI weights for each focus spot."""
    spacing = float(spec["spacing"])
    waist = float(spec["waist_radius"])
    half = shape // 2
    coords = (torch.arange(shape, dtype=torch.double, device=device) - half) * spacing
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')

    sigma = waist * sigma_factor
    weights = []
    for center in spec["focus_centers"]:
        cy, cx = center
        dist_sq = (yy - cy)**2 + (xx - cx)**2
        w = torch.exp(-dist_sq / (2.0 * sigma**2))
        weights.append(w)
    return weights


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    shape = int(spec["shape"])
    input_field = Field(gaussian(shape, spec["waist_radius"]), z=0).normalize(1.0).to(device)
    target_field = _build_target_field(spec, device)
    system = _build_system(spec, device)

    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()
    roi_masks = _get_focus_rois(spec, shape, device, radius_factor=1.2)
    soft_weights = _get_soft_focus_rois(spec, shape, device, sigma_factor=1.0)

    # Build a weighted target intensity for shape matching
    target_intensity = target_field.data.abs().square()
    target_intensity_norm = target_intensity / (target_intensity.sum() + 1e-12)

    losses: list[float] = []

    # Phase 1: overlap-based warm-up with Gerchberg-Saxton-like initialization
    # Use random phase init for diversity
    for param in system.parameters():
        param.data.uniform_(-0.5, 0.5)

    optimizer1 = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=100, eta_min=0.01)

    for step in range(100):
        optimizer1.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])
        overlap = output_field.inner(target_field).abs().square()
        loss = 1.0 - overlap
        loss.backward()
        optimizer1.step()
        scheduler1.step()
        losses.append(float(loss.item()))

    # Phase 2: ratio-aware + efficiency + shape cosine loss
    optimizer2 = torch.optim.Adam(system.parameters(), lr=0.035)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=300, eta_min=0.002)

    for step in range(300):
        optimizer2.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])

        intensity = output_field.data.abs().square()

        # Overlap loss
        overlap = output_field.inner(target_field).abs().square()
        loss_overlap = 1.0 - overlap

        # Ratio loss using soft weights
        spot_powers = []
        for w in soft_weights:
            spot_powers.append((intensity * w).sum())
        spot_powers_t = torch.stack(spot_powers)
        total_spot = spot_powers_t.sum() + 1e-12
        pred_ratios = spot_powers_t / total_spot
        ratio_err = (pred_ratios - ratios).abs().mean()

        # Efficiency loss (energy in ROIs vs total)
        hard_spot_total = sum((intensity * mask).sum() for mask in roi_masks)
        total_energy = intensity.sum() + 1e-12
        efficiency = hard_spot_total / total_energy
        loss_eff = 1.0 - efficiency

        # Shape cosine loss
        int_norm = intensity / (intensity.sum() + 1e-12)
        cosine_sim = (int_norm * target_intensity_norm).sum() / (
            torch.sqrt((int_norm**2).sum() * (target_intensity_norm**2).sum()) + 1e-12
        )
        loss_shape = 1.0 - cosine_sim

        # Weighted ratio MSE for better gradient
        ratio_mse = ((pred_ratios - ratios)**2).mean()

        # Combined loss with progressive weighting
        progress = step / 300.0
        w_overlap = max(0.2, 0.6 - 0.4 * progress)
        w_ratio = min(0.45, 0.1 + 0.35 * progress)
        w_eff = 0.15
        w_shape = min(0.15, 0.05 + 0.1 * progress)

        loss = (w_overlap * loss_overlap +
                w_ratio * (ratio_err + 0.5 * ratio_mse) +
                w_eff * loss_eff +
                w_shape * loss_shape)

        loss.backward()
        optimizer2.step()
        scheduler2.step()
        losses.append(float(loss.item()))

    # Phase 3: Fine-tuning with strong ratio emphasis
    optimizer3 = torch.optim.Adam(system.parameters(), lr=0.01)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=150, eta_min=0.001)

    for step in range(150):
        optimizer3.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])
        intensity = output_field.data.abs().square()

        overlap = output_field.inner(target_field).abs().square()
        loss_overlap = 1.0 - overlap

        spot_powers = []
        for w in soft_weights:
            spot_powers.append((intensity * w).sum())
        spot_powers_t = torch.stack(spot_powers)
        total_spot = spot_powers_t.sum() + 1e-12
        pred_ratios = spot_powers_t / total_spot
        ratio_err = (pred_ratios - ratios).abs().mean()

        hard_spot_total = sum((intensity * mask).sum() for mask in roi_masks)
        total_energy = intensity.sum() + 1e-12
        efficiency = hard_spot_total / total_energy
        loss_eff = 1.0 - efficiency

        int_norm = intensity / (intensity.sum() + 1e-12)
        cosine_sim = (int_norm * target_intensity_norm).sum() / (
            torch.sqrt((int_norm**2).sum() * (target_intensity_norm**2).sum()) + 1e-12
        )
        loss_shape = 1.0 - cosine_sim

        loss = 0.2 * loss_overlap + 0.45 * ratio_err + 0.15 * loss_eff + 0.2 * loss_shape

        loss.backward()
        optimizer3.step()
        scheduler3.step()
        losses.append(float(loss.item()))

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_field": target_field,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END
