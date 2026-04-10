# EVOLVE-BLOCK-START
"""Improved solver for Task 2: multi-plane focusing."""

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
        "steps": 1500,
        "lr": 0.22,
        "num_trials": 2,
    }


def _build_system(spec: dict[str, Any], device: str, init_random: bool = True) -> System:
    shape = int(spec["shape"])
    layers = []
    for z in spec["layer_z"]:
        if init_random:
            phase_init = torch.randn((shape, shape), dtype=torch.double) * 1.2
        else:
            phase_init = torch.zeros((shape, shape), dtype=torch.double)
        layers.append(PhaseModulator(Parameter(phase_init), z=float(z)))
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


def _build_individual_spot_fields(spec: dict[str, Any], plane_cfg: dict[str, Any], device: str) -> list[Field]:
    """Build individual normalized Gaussian spot fields for each center in a plane."""
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])
    fields = []
    for center in plane_cfg["centers"]:
        spot = gaussian(shape, waist, offset=center).real.to(device)
        f = Field(spot.to(torch.cdouble), z=plane_cfg["z"]).normalize(1.0)
        fields.append(f)
    return fields


def _build_target_intensity(spec: dict[str, Any], plane_cfg: dict[str, Any], device: str) -> torch.Tensor:
    """Build a target intensity pattern for a plane (sum of weighted Gaussians)."""
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])

    target = torch.zeros((shape, shape), dtype=torch.double, device=device)
    ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()

    for ratio, center in zip(ratios, plane_cfg["centers"]):
        g = gaussian(shape, waist, offset=center).real.to(device)
        target += ratio * (g ** 2)

    # Normalize to unit sum
    target = target / (target.sum() + 1e-15)
    return target


def _run_single_trial(
    spec: dict[str, Any],
    device: str,
    input_field: Field,
    target_fields: list[Field],
    target_intensities: list[torch.Tensor],
    spot_fields_per_plane: list[list[Field]],
    target_ratios_per_plane: list[torch.Tensor],
    spot_masks_per_plane: list[list[torch.Tensor]],
    roi_masks: list[torch.Tensor],
    trial_seed: int,
) -> tuple[System, list[float], float]:
    """Run a single optimization trial and return system, losses, and final score."""
    torch.manual_seed(trial_seed)
    system = _build_system(spec, device, init_random=True)

    num_steps = int(spec["steps"])
    lr = float(spec["lr"])

    optimizer = torch.optim.Adam(system.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.005)

    losses: list[float] = []

    for step in range(num_steps):
        optimizer.zero_grad()
        plane_losses = []
        progress = step / max(num_steps - 1, 1)

        # Phase 1: overlap + shape focus (first 30%)
        # Phase 2: add efficiency + ratio (30-100%)
        # Phase 3: fine-tune all (last 30%)
        phase1_end = 0.25
        phase2_end = 0.7

        for plane_idx, (plane_cfg, target_field) in enumerate(zip(spec["planes"], target_fields)):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])

            # 1) Overlap loss (shape + phase matching)
            overlap = output.inner(target_field).abs().square()
            overlap_loss = 1.0 - overlap

            # 2) Intensity shape loss (cosine similarity)
            out_intensity = output.data.abs().square()
            tgt_intensity = target_intensities[plane_idx]

            out_flat = out_intensity.flatten()
            tgt_flat = tgt_intensity.flatten()
            cos_sim = torch.dot(out_flat, tgt_flat) / (
                torch.norm(out_flat) * torch.norm(tgt_flat) + 1e-15
            )
            shape_loss = 1.0 - cos_sim

            # 2b) MSE intensity loss (normalized)
            out_norm = out_intensity / (out_intensity.sum() + 1e-15)
            mse_loss = ((out_norm - tgt_intensity) ** 2).sum()

            # 3) Ratio loss using ROI masks
            spot_masks = spot_masks_per_plane[plane_idx]
            target_ratios = target_ratios_per_plane[plane_idx]
            roi_powers = []
            for sm in spot_masks:
                roi_p = (out_intensity * sm).sum()
                roi_powers.append(roi_p)
            roi_powers_t = torch.stack(roi_powers)
            roi_total = roi_powers_t.sum() + 1e-12
            pred_ratios = roi_powers_t / roi_total
            ratio_loss = (pred_ratios - target_ratios).square().sum()

            # 4) Efficiency: maximize power in ROI
            roi_mask = roi_masks[plane_idx]
            total_power = out_intensity.sum() + 1e-12
            roi_power = (out_intensity * roi_mask).sum()
            efficiency = roi_power / total_power
            efficiency_loss = 1.0 - efficiency

            # 5) Negative log efficiency for stronger gradient when efficiency is low
            neg_log_eff = -torch.log(efficiency + 1e-8)

            # 6) Field overlap with individual spots (weighted by target ratios)
            spot_fields = spot_fields_per_plane[plane_idx]
            field_spot_powers = []
            for sf, tr in zip(spot_fields, target_ratios):
                sp = output.inner(sf).abs().square()
                field_spot_powers.append(tr * sp)
            field_efficiency = sum(field_spot_powers)

            # 7) Peak intensity at spot centers vs background
            # Encourage high contrast

            # Dynamic weighting based on training phase
            if progress < phase1_end:
                # Phase 1: focus on overlap and shape
                p = progress / phase1_end
                w_overlap = 2.0
                w_shape = 1.0 + 2.0 * p
                w_mse = 0.5 + 1.0 * p
                w_ratio = 0.2
                w_efficiency = 0.5 + 1.0 * p
                w_neg_log = 0.1
                w_field_eff = 0.5
            elif progress < phase2_end:
                # Phase 2: ramp up efficiency and ratio
                p = (progress - phase1_end) / (phase2_end - phase1_end)
                w_overlap = 1.5
                w_shape = 3.0 + 2.0 * p
                w_mse = 1.5 + 1.5 * p
                w_ratio = 0.5 + 1.5 * p
                w_efficiency = 2.0 + 3.0 * p
                w_neg_log = 0.3 + 0.5 * p
                w_field_eff = 1.0 + 0.5 * p
            else:
                # Phase 3: fine-tune everything, strong shape + efficiency
                p = (progress - phase2_end) / (1.0 - phase2_end)
                w_overlap = 1.0
                w_shape = 5.0 + 2.0 * p
                w_mse = 3.0 + 2.0 * p
                w_ratio = 2.0 + 1.0 * p
                w_efficiency = 5.0 + 3.0 * p
                w_neg_log = 0.8 + 0.5 * p
                w_field_eff = 1.5 + 0.5 * p

            plane_loss = (w_overlap * overlap_loss
                        + w_shape * shape_loss
                        + w_mse * mse_loss
                        + w_ratio * ratio_loss
                        + w_efficiency * efficiency_loss
                        + w_neg_log * neg_log_eff
                        - w_field_eff * field_efficiency)
            plane_losses.append(plane_loss)

        # Smoothness regularization on phase patterns
        smooth_weight = 0.01 * (1.0 - 0.5 * progress)  # decrease over time
        smooth_loss = torch.tensor(0.0, dtype=torch.double, device=device)
        for param in system.parameters():
            if param.dim() == 2:
                dx = (param[:, 1:] - param[:, :-1]).square().mean()
                dy = (param[1:, :] - param[:-1, :]).square().mean()
                smooth_loss = smooth_loss + dx + dy

        loss = torch.stack(plane_losses).mean() + smooth_weight * smooth_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=5.0)

        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    # Evaluate final score
    final_score = 0.0
    with torch.no_grad():
        for plane_idx, (plane_cfg, target_field) in enumerate(zip(spec["planes"], target_fields)):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])
            out_intensity = output.data.abs().square()
            tgt_intensity = target_intensities[plane_idx]
            out_flat = out_intensity.flatten()
            tgt_flat = tgt_intensity.flatten()
            cos_sim = torch.dot(out_flat, tgt_flat) / (
                torch.norm(out_flat) * torch.norm(tgt_flat) + 1e-15
            )
            # Use shape cosine as proxy for score
            roi_mask = roi_masks[plane_idx]
            total_power = out_intensity.sum() + 1e-12
            roi_power = (out_intensity * roi_mask).sum()
            eff = roi_power / total_power
            final_score += float(cos_sim.item()) + float(eff.item()) * 0.5

    return system, losses, final_score


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)

    target_fields = [_build_target_field_for_plane(spec, p, device) for p in spec["planes"]]
    target_intensities = [_build_target_intensity(spec, p, device) for p in spec["planes"]]
    spot_fields_per_plane = [_build_individual_spot_fields(spec, p, device) for p in spec["planes"]]
    target_ratios_per_plane = [
        torch.tensor(p["ratios"], dtype=torch.double, device=device) / sum(p["ratios"])
        for p in spec["planes"]
    ]

    # Build ROI masks
    shape_n = int(spec["shape"])
    spacing = float(spec["spacing"])
    waist = float(spec["waist_radius"])
    roi_radius = spec.get("roi_radius_m", waist * 2.0)
    half_extent = shape_n * spacing / 2.0
    coords = torch.linspace(-half_extent + spacing / 2, half_extent - spacing / 2, shape_n,
                            dtype=torch.double, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    spot_masks_per_plane = []
    roi_masks = []
    for plane_cfg in spec["planes"]:
        spot_masks = []
        plane_mask = torch.zeros((shape_n, shape_n), dtype=torch.double, device=device)
        for center in plane_cfg["centers"]:
            cx, cy = center
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            spot_mask = (dist_sq <= roi_radius ** 2).double()
            spot_masks.append(spot_mask)
            plane_mask = torch.clamp(plane_mask + spot_mask, 0, 1)
        spot_masks_per_plane.append(spot_masks)
        roi_masks.append(plane_mask)

    # Run multiple trials and pick the best
    num_trials = int(spec.get("num_trials", 3))
    best_system = None
    best_losses = None
    best_score = -float('inf')

    for trial in range(num_trials):
        trial_seed = seed + trial * 12345
        system, losses, score = _run_single_trial(
            spec, device, input_field,
            target_fields, target_intensities,
            spot_fields_per_plane, target_ratios_per_plane,
            spot_masks_per_plane, roi_masks,
            trial_seed,
        )
        if score > best_score:
            best_score = score
            best_system = system
            best_losses = losses

    return {
        "spec": spec,
        "system": best_system,
        "input_field": input_field,
        "target_fields": target_fields,
        "loss_history": best_losses,
    }
# EVOLVE-BLOCK-END