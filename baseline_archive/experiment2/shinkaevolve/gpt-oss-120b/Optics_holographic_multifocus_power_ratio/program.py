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


def _compute_efficiency(output_field: Field, focus_masks: list[torch.Tensor], device: str) -> torch.Tensor:
    """Compute power concentration at target focus regions using precomputed masks."""
    intensity = output_field.intensity()
    total_power_in_foci = torch.tensor(0.0, dtype=torch.double, device=device)
    for mask in focus_masks:
        total_power_in_foci = total_power_in_foci + (intensity * mask).sum()
    total_output_power = intensity.sum()
    return total_power_in_foci / (total_output_power + 1e-10)


def _initialize_phases(system: System, spec: dict[str, Any], seed: int, device: str) -> None:
    """Initialize phases using superposition of blazed grating patterns for each focus."""
    shape = int(spec["shape"])
    spacing = float(spec["spacing"])
    wavelength = float(spec["wavelength"])
    output_z = float(spec["output_z"])

    # Create coordinate grid
    x = torch.linspace(-shape//2, shape//2 - 1, shape, dtype=torch.double, device=device) * spacing
    y = torch.linspace(-shape//2, shape//2 - 1, shape, dtype=torch.double, device=device) * spacing
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Build phase pattern as weighted superposition of linear phases (blazed gratings)
    # Each focus requires a specific phase gradient to steer light there
    ratios = torch.tensor(spec["focus_ratios"], dtype=torch.double, device=device)
    ratios = ratios / ratios.sum()

    # Calculate required k-vectors for each focus
    k = 2 * torch.pi / wavelength

    # Initialize with superposition of steering phases weighted by sqrt of ratio
    combined_phase = torch.zeros((shape, shape), dtype=torch.double, device=device)

    for ratio, (cx, cy) in zip(ratios, spec["focus_centers"]):
        # Linear phase gradient to steer beam toward focus center
        # Phase = k * (sin_theta_x * x + sin_theta_y * y)
        # sin_theta ≈ cx / output_z for small angles
        sin_theta_x = cx / output_z
        sin_theta_y = cy / output_z
        steering_phase = k * (sin_theta_x * X + sin_theta_y * Y)
        # Weight by sqrt of ratio (amplitude weighting)
        combined_phase = combined_phase + torch.sqrt(ratio) * steering_phase

    # Normalize combined phase
    combined_phase = combined_phase / ratios.sqrt().sum()

    # Add structured noise and layer-specific variations
    torch.manual_seed(seed)
    for i, layer in enumerate(system):
        # Layer-dependent phase offset and noise
        layer_phase = combined_phase * (0.8 + 0.1 * i)
        noise = (torch.rand((shape, shape), dtype=torch.double, device=device) - 0.5) * 0.3 * (1 + 0.2 * i)
        layer.phase.data = layer_phase + noise


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    shape = int(spec["shape"])
    waist = float(spec["waist_radius"])

    # Precompute focus masks for efficiency calculation
    focus_masks = [gaussian(shape, waist, offset=center).real.to(device) for center in spec["focus_centers"]]

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    target_field = _build_target_field(spec, device)
    system = _build_system(spec, device)

    # Physics-informed phase initialization with grating superposition
    _initialize_phases(system, spec, seed, device)

    steps = int(spec["steps"])

    # AdamW optimizer with higher learning rate and cosine annealing
    optimizer = torch.optim.AdamW(system.parameters(), lr=0.25, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps//3, T_mult=2, eta_min=0.01
    )

    losses: list[float] = []
    best_score = 0.0
    best_efficiency = 0.0
    best_phases = None

    for step in range(steps):
        optimizer.zero_grad()
        output_field = system.measure_at_z(input_field, z=spec["output_z"])

        # Hybrid loss: prioritize efficiency more to boost power concentration
        overlap = output_field.inner(target_field).abs().square()
        efficiency = _compute_efficiency(output_field, focus_masks, device)

        # Weighted combination: emphasize efficiency to improve power concentration
        loss = 0.4 * (1.0 - overlap) + 0.6 * (1.0 - efficiency)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.5)

        optimizer.step()
        scheduler.step()

        current_overlap = float(overlap.item())
        current_efficiency = float(efficiency.item())
        losses.append(float(loss.item()))

        # Track best solution based on combined metric (score formula approximation)
        # score ~ ratio_score * (efficiency/target)^exp, prioritize efficiency
        current_score = current_overlap * 0.5 + current_efficiency * 0.5
        if current_score > best_score or best_phases is None:
            best_score = current_score
            best_efficiency = current_efficiency
            best_phases = [layer.phase.data.clone() for layer in system]

    # Restore best solution
    if best_phases is not None:
        with torch.no_grad():
            for layer, phase in zip(system, best_phases):
                layer.phase.data = phase

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_field": target_field,
        "loss_history": losses,
    }
# EVOLVE-BLOCK-END