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
        "shape": 64,
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
        "steps": 15,
        "lr": 0.25,
    }


def _build_system(spec: dict[str, Any], device: str) -> System:
    shape = int(spec["shape"])
    layers = [
        PhaseModulator(Parameter(torch.zeros((shape, shape), dtype=torch.float32)), z=float(z))
        for z in spec["layer_z"]
    ]
    return System(*layers).to(device)


def _build_target_field_for_plane(spec: dict[str, Any], plane_cfg: dict[str, Any], device: str) -> Field:
    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])
    target = torch.zeros((shape, shape), dtype=torch.float32, device=device)
    ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.float32, device=device)
    ratios = ratios / ratios.sum()
    for ratio, center in zip(ratios, plane_cfg["centers"]):
        target += torch.sqrt(ratio) * gaussian(shape, waist, offset=center).real.to(device)
    return Field(target.to(torch.complex64), z=plane_cfg["z"]).normalize(1.0)


def _roi_powers(field: Field, centers: list, radius: float) -> torch.Tensor:
    x, y = field.meshgrid()
    intensity = field.intensity()
    powers = []
    for cx, cy in centers:
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2).to(intensity.dtype)
        powers.append((intensity * mask).sum())
    return torch.stack(powers)


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    import time
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])
    
    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    system = _build_system(spec, device)
    target_fields = [_build_target_field_for_plane(spec, p, device) for p in spec["planes"]]
    roi_radius = float(spec.get("roi_radius_m", 2.5 * spec["waist_radius"]))
    
    optimizer = torch.optim.Adam(system.parameters(), lr=float(spec["lr"]))
    losses = []
    best_loss = float('inf')
    best_params = None
    no_improve = 0
    start_time = time.time()
    max_time = 25.0
    
    for step in range(int(spec["steps"])):
        if time.time() - start_time > max_time:
            break
        optimizer.zero_grad()
        plane_losses = []
        
        for plane_cfg, target_field in zip(spec["planes"], target_fields):
            output = system.measure_at_z(input_field, z=plane_cfg["z"])
            overlap = output.inner(target_field).abs().square()
            overlap_loss = 1.0 - overlap
            
            powers = _roi_powers(output, plane_cfg["centers"], roi_radius)
            focus_power = powers.sum()
            total_power = output.intensity().sum() + 1e-8
            pred_ratios = powers / (focus_power + 1e-8)
            target_ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.float32, device=device)
            target_ratios = target_ratios / target_ratios.sum()
            ratio_loss = torch.mean(torch.abs(pred_ratios - target_ratios))
            leakage_loss = 1.0 - focus_power / total_power
            
            # Stronger ratio loss weight (matching oracle's emphasis)
            plane_loss = 0.35 * overlap_loss + 0.95 * ratio_loss + 0.35 * leakage_loss
            plane_losses.append(plane_loss)
        
        # Adaptive plane weighting: focus on harder planes (like oracle)
        plane_losses_t = torch.stack(plane_losses)
        weights = torch.softmax(plane_losses_t.detach() / 0.2, dim=0)
        total_loss = (weights * plane_losses_t).sum()
        
        total_loss.backward()
        optimizer.step()
        loss_val = float(total_loss.item())
        losses.append(loss_val)
        
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_params = [p.clone() for p in system.parameters()]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4:
                break
    
    if best_params is not None:
        for p, bp in zip(system.parameters(), best_params):
            p.data.copy_(bp)
    
    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_fields": target_fields,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END
