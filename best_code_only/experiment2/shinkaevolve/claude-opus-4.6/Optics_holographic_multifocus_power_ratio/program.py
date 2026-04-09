# EVOLVE-BLOCK-START
"""Baseline solver for Task 1: multifocus with target power ratios."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Parameter
import math

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
        "steps": 80,
        "lr": 0.15,
    }


def _build_target_field(spec: dict[str, Any], device: str, focus_waist: float | None = None) -> Field:
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])
    fw = focus_waist if focus_waist is not None else waist * 0.45
    target = torch.zeros((shape, shape), dtype=torch.double, device=device)

    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()

    for ratio, center in zip(ratios, spec["focus_centers"]):
        target += torch.sqrt(ratio) * gaussian(shape, fw, offset=center).real.to(device)

    return Field(target.to(torch.cdouble), z=spec["output_z"]).normalize(1.0)


def _build_roi_masks(spec: dict[str, Any], device: str) -> torch.Tensor:
    """Build binary ROI masks for each focus spot."""
    shape = int(spec["shape"])
    spacing = float(spec["spacing"])
    roi_radius = float(spec.get("roi_radius_m", 3e-5))

    half = shape // 2
    coords = (torch.arange(shape, dtype=torch.double, device=device) - half) * spacing

    masks = []
    for center in spec["focus_centers"]:
        cy, cx = float(center[0]), float(center[1])
        yy = coords - cy
        xx = coords - cx
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        dist = torch.sqrt(Y**2 + X**2)
        mask = (dist <= roi_radius).to(torch.double)
        masks.append(mask)

    return torch.stack(masks, dim=0)  # (n_foci, shape, shape)


def _build_system(spec: dict[str, Any], device: str, init_scale: float = 0.0) -> System:
    shape = int(spec["shape"])
    layers = [
        PhaseModulator(
            Parameter(torch.randn((shape, shape), dtype=torch.double) * init_scale),
            z=float(z),
        )
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def _compute_roi_loss(output_field: Field, roi_masks: torch.Tensor,
                      target_ratios: torch.Tensor) -> torch.Tensor:
    """Compute a loss based on ROI power ratios and total efficiency."""
    intensity = output_field.data.abs().square()
    # Power in each ROI
    roi_powers = (intensity * roi_masks).sum(dim=(-2, -1))  # (n_foci,)
    total_roi_power = roi_powers.sum()
    total_power = intensity.sum()

    # Efficiency loss: maximize fraction of power in ROIs
    efficiency = total_roi_power / (total_power + 1e-12)
    eff_loss = 1.0 - efficiency

    # Ratio loss: match target ratios
    pred_ratios = roi_powers / (total_roi_power + 1e-12)
    ratio_loss = (pred_ratios - target_ratios).abs().mean()

    return eff_loss, ratio_loss


def _build_target_intensity_pattern(spec: dict[str, Any], device: str, roi_masks: torch.Tensor,
                                     target_ratios: torch.Tensor) -> torch.Tensor:
    """Build a target intensity pattern for shape cosine optimization."""
    shape = int(spec["shape"])
    spacing = float(spec["spacing"])
    roi_radius = float(spec.get("roi_radius_m", 3e-5))
    sigma = roi_radius * 0.4
    half = shape // 2
    coords = (torch.arange(shape, dtype=torch.double, device=device) - half) * spacing

    target_int = torch.zeros((shape, shape), dtype=torch.double, device=device)
    for i, center in enumerate(spec["focus_centers"]):
        cy, cx = float(center[0]), float(center[1])
        yy = coords - cy
        xx = coords - cx
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        g = torch.exp(-(Y**2 + X**2) / (2 * sigma**2))
        roi_integral = (g * roi_masks[i]).sum()
        if roi_integral > 0:
            g = g * target_ratios[i] / roi_integral
        target_int += g
    return target_int


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])
    input_field = Field(gaussian(shape, waist), z=0).normalize(1.0).to(device)

    # Build ROI masks for direct ratio/efficiency optimization
    roi_masks = _build_roi_masks(spec, device)
    target_ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    target_ratios = target_ratios / target_ratios.sum()

    # Build target intensity for shape cosine loss
    target_int = _build_target_intensity_pattern(spec, device, roi_masks, target_ratios)
    # Precompute flattened target for shape cosine
    tgt_flat = target_int.flatten()
    tgt_flat_norm = tgt_flat.norm()

    # Combined ROI mask (union)
    roi_union = roi_masks.sum(dim=0).clamp(max=1.0)

    lr = float(spec["lr"])

    # Single config with maximum steps for best convergence
    total_steps = 400

    best_score = -1.0
    best_system = None
    best_losses = None
    best_target_field = None

    # Single well-tuned config
    fw = waist * 0.38
    target_field = _build_target_field(spec, device, focus_waist=fw)

    torch.manual_seed(seed)
    system = _build_system(spec, device, init_scale=0.5)

    losses: list[float] = []

    # Save best checkpoint during training
    best_checkpoint_score = -1.0
    best_checkpoint_state = None

    # Use a single optimizer with cosine annealing
    optimizer = torch.optim.Adam(system.parameters(), lr=lr * 1.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.02)

    for step in range(total_steps):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])

        # Compute all loss components
        overlap = output_field.inner(target_field).abs().square()
        overlap_loss = 1.0 - overlap

        intensity = output_field.data.abs().square()
        roi_powers = (intensity * roi_masks).sum(dim=(-2, -1))
        total_roi_power = roi_powers.sum()
        total_power = intensity.sum()

        efficiency = total_roi_power / (total_power + 1e-12)
        eff_loss = 1.0 - efficiency

        pred_ratios = roi_powers / (total_roi_power + 1e-12)
        ratio_loss = (pred_ratios - target_ratios).abs().mean()

        int_flat = intensity.flatten()
        cos_sim = torch.dot(int_flat, tgt_flat) / (int_flat.norm() * tgt_flat_norm + 1e-12)
        shape_loss = 1.0 - cos_sim

        roi_power_total = (intensity * roi_union).sum()
        leakage_loss = 1.0 - roi_power_total / (total_power + 1e-12)

        neg_log_eff = -torch.log(efficiency + 1e-8)

        # Progress-dependent weights: gradually shift from overlap to efficiency/shape
        t = step / max(1, total_steps - 1)

        if t < 0.08:
            # Pure overlap warmup
            loss = overlap_loss
        elif t < 0.20:
            # Transition from overlap to mixed
            alpha = (t - 0.08) / 0.12
            mixed = (0.30 * eff_loss + 0.25 * ratio_loss * 5.0 +
                     0.25 * shape_loss + 0.10 * leakage_loss + 0.10 * neg_log_eff * 0.3)
            loss = (1.0 - alpha) * overlap_loss + alpha * mixed
        elif t < 0.50:
            # Balanced phase
            loss = (0.12 * overlap_loss + 0.22 * eff_loss + 0.12 * ratio_loss * 5.0 +
                    0.22 * shape_loss + 0.12 * leakage_loss + 0.20 * neg_log_eff * 0.3)
        elif t < 0.75:
            # Efficiency + shape emphasis
            loss = (0.06 * overlap_loss + 0.25 * eff_loss + 0.12 * ratio_loss * 5.0 +
                    0.25 * shape_loss + 0.12 * leakage_loss + 0.20 * neg_log_eff * 0.3)
        else:
            # Final fine-tuning: directly optimize score proxy
            # score = ratio_score * eff_score * shape_cos
            # ratio_score = clamp(1 - ratio_mae/0.1, 0, 1)
            # eff_score = clamp(eff/0.2, 0, 1)
            # Differentiable proxy:
            d_ratio_score = 1.0 - ratio_loss / 0.1
            d_eff_score = torch.clamp(efficiency / 0.2, max=1.0)
            d_score = d_ratio_score * d_eff_score * cos_sim
            score_loss = 1.0 - d_score

            # Also keep some direct losses for gradient stability
            loss = (0.40 * score_loss + 0.15 * eff_loss + 0.10 * ratio_loss * 5.0 +
                    0.15 * shape_loss + 0.10 * leakage_loss + 0.10 * neg_log_eff * 0.3)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(overlap_loss.item()))

        # Checkpoint best every 20 steps after warmup
        if step >= 40 and step % 20 == 0:
            with torch.no_grad():
                out_f = system.measure_at_z(input_field, z=spec["output_z"])
                inten = out_f.data.abs().square()
                rp = (inten * roi_masks).sum(dim=(-2, -1))
                tr = rp.sum()
                tp = inten.sum()
                e = (tr / (tp + 1e-12)).item()
                pr = rp / (tr + 1e-12)
                rm = (pr - target_ratios).abs().mean().item()
                ifl = inten.flatten()
                sc = (torch.dot(ifl, tgt_flat) / (ifl.norm() * tgt_flat_norm + 1e-12)).item()
                rs = max(0, 1.0 - rm / 0.1)
                es = min(e / 0.2, 1.0)
                approx = rs * es * sc
                if approx > best_checkpoint_score:
                    best_checkpoint_score = approx
                    best_checkpoint_state = {k: v.clone() for k, v in system.state_dict().items()}

    # Final evaluation
    with torch.no_grad():
        output_field = system.measure_at_z(input_field, z=spec["output_z"])
        intensity = output_field.data.abs().square()
        roi_powers = (intensity * roi_masks).sum(dim=(-2, -1))
        total_roi = roi_powers.sum()
        total_power = intensity.sum()
        eff = (total_roi / (total_power + 1e-12)).item()
        pred_r = roi_powers / (total_roi + 1e-12)
        ratio_mae = (pred_r - target_ratios).abs().mean().item()

        int_flat = intensity.flatten()
        shape_cos = (torch.dot(int_flat, tgt_flat) / (
            int_flat.norm() * tgt_flat_norm + 1e-12
        )).item()

        ratio_score = max(0, 1.0 - ratio_mae / 0.1)
        eff_score = min(eff / 0.2, 1.0)
        final_score = ratio_score * eff_score * shape_cos

    # Use best checkpoint if it's better
    if best_checkpoint_state is not None and best_checkpoint_score > final_score:
        system.load_state_dict(best_checkpoint_state)
        best_score = best_checkpoint_score
    else:
        best_score = final_score

    best_system = system
    best_losses = losses
    best_target_field = target_field

    return {
        "spec": spec,
        "system": best_system,
        "input_field": input_field,
        "target_field": best_target_field,
        "loss_history": best_losses,
    }
# EVOLVE-BLOCK-END