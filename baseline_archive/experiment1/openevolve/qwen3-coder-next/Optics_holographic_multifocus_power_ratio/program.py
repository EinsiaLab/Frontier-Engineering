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
        "shape": 72, "spacing": 10e-6, "wavelength": 700e-9, "waist_radius": waist,
        "layer_z": [0.0, 0.12, 0.24, 0.36], "output_z": 0.56,
        "focus_centers": [(-2.3*waist,-1.6*waist),(0.0,-2.3*waist),(2.3*waist,-1.6*waist),
                         (-2.3*waist,1.6*waist),(0.0,2.3*waist),(2.3*waist,1.6*waist)],
        "focus_ratios": [0.24,0.17,0.16,0.15,0.14,0.14], "steps": 180, "lr": 0.075
    }


def _build_target_field(spec: dict[str, Any], device: str) -> Field:
    shape, waist = int(spec["shape"]), float(spec["waist_radius"])
    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device) / sum(spec["focus_ratios"])
    
    # Build target field with proper scaling based on ratios
    target = torch.zeros((shape, shape), dtype=torch.double, device=device)
    for r, c in zip(ratios, spec["focus_centers"]):
        target += torch.sqrt(r) * gaussian(shape, waist, offset=c).real.to(device)
    
    # Normalize to match expected total intensity
    target = target / (torch.sqrt(target.sum()) + 1e-12)
    
    return Field(target.to(torch.cdouble), z=spec["output_z"]).normalize(1.0)


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    layers = [PhaseModulator(Parameter(torch.zeros((shape, shape), dtype=torch.double)), z=float(z)) 
              for z in spec["layer_z"]]
    return System(*layers).to(device)


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    target_field = _build_target_field(spec, device)
    system = _build_system(spec, device)

    # Get phase modulators from system modules
    phase_modulators = [mod for mod in system.modules() if isinstance(mod, PhaseModulator)]

    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    # Use CosineAnnealingWarmRestarts for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(spec["steps"]//4), T_mult=1)
    losses: list[float] = []

    # Add warmup period for learning rate scheduler
    warmup_steps = int(spec["steps"] * 0.1)
    
    for step in range(int(spec["steps"])):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])
        
        # Primary loss: overlap with target
        overlap = output_field.inner(target_field).abs().square()
        overlap_loss = 1.0 - overlap
        
        # Compute intensity for ratio-based losses
        output_intensity = output_field.intensity()
        total_power = output_intensity.sum() + 1e-12
        
        # Compute power in each focus region using Gaussian-weighted masks for smoother gradients
        x, y = output_field.meshgrid()
        roi_radius = float(spec.get("roi_radius_m", 3 * spec["spacing"]))
        sigma_pix = roi_radius / 2.0  # Gaussian sigma for smoother integration
        focus_powers = []
        for cx, cy in spec["focus_centers"]:
            # Gaussian-weighted mask for smoother gradients and better convergence
            mask = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma_pix ** 2))
            focus_powers.append((output_intensity * mask).sum())
        
        focus_powers = torch.stack(focus_powers)
        focus_power_total = focus_powers.sum()
        predicted_ratios = focus_powers / (focus_power_total + 1e-12)
        
        # Target ratios
        target_ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
        target_ratios = target_ratios / target_ratios.sum()
        
        # Ratio MAE loss (critical for scoring)
        ratio_loss = torch.mean(torch.abs(predicted_ratios - target_ratios))
        
        # Efficiency loss (leakage penalty)
        efficiency = focus_power_total / total_power
        leakage_loss = 1.0 - efficiency
        
        # Phase regularization with smoothness penalty
        phase_reg = torch.tensor(0.0, dtype=torch.double, device=device)
        phase_smoothness = torch.tensor(0.0, dtype=torch.double, device=device)
        
        for mod in phase_modulators:
            # Center phase
            phase_reg = phase_reg + (mod.phase - mod.phase.mean()).square().mean()
            
            # Smoothness penalty - penalize large phase gradients
            dx = mod.phase[:, 1:] - mod.phase[:, :-1]
            dy = mod.phase[1:, :] - mod.phase[:-1, :]
            phase_smoothness = phase_smoothness + (dx.abs().mean() + dy.abs().mean())
        
        if len(phase_modulators) > 0:
            phase_reg = phase_reg / len(phase_modulators)
        
        # Combined loss - weights tuned for scoring formula components
        # efficiency: 58% (leakage_penalty), ratio: 22% (ratio_loss), shape: 20% (overlap)
        # Using more aggressive weights for efficiency optimization as it has highest weight in scoring
        loss = 0.1 * overlap_loss + 0.5 * ratio_loss + 3.0 * leakage_loss + 1e-4 * phase_reg + 5e-3 * phase_smoothness
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        # Learning rate warmup in early steps
        if step < warmup_steps:
            warmup_factor = (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = float(spec["lr"]) * warmup_factor
        
        scheduler.step()
        losses.append(float(loss.item()))

    return {"spec": spec, "system": system, "input_field": input_field,
            "target_field": target_field, "loss_history": losses}
# EVOLVE-BLOCK-END
