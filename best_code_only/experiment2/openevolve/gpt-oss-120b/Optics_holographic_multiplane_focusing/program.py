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
    """Create a multilayer phase‑modulation system.

    Initialise each phase mask with random values in [-π, π] to break symmetry
    and provide a useful gradient from the first optimisation step.
    """
    shape = int(spec["shape"])
    layers = [
        PhaseModulator(
            Parameter(
                torch.rand((shape, shape), dtype=torch.double) * 2 * torch.pi - torch.pi
            ),
            z=float(z),
        )
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


# ----------------------------------------------------------------------
# Helper: power inside each circular ROI around the target centres.
# Used for ratio‑error and leakage‑loss terms.
def _roi_powers(field: Field, centers: list[tuple[float, float]], radius: float) -> torch.Tensor:
    """Return a tensor of powers, one per centre."""
    x, y = field.meshgrid()
    intensity = field.intensity()
    powers = []
    for cx, cy in centers:
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
        powers.append((intensity * mask.to(intensity.dtype)).sum())
    return torch.stack(powers)


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
    # Run twice the nominal number of steps so the optimiser can converge better
    # while still respecting the overall time budget.
    _internal_steps = int(spec["steps"]) * 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_internal_steps
    )
    losses: list[float] = []

    # ROI radius is supplied by the evaluator (default = 3 × spacing)
    roi_radius = float(spec["roi_radius_m"])

    # Temperature for soft‑max re‑weighting of per‑plane losses (same idea as the oracle)
    loss_weight_temp = 0.2

    for _ in range(_internal_steps):
        optimizer.zero_grad()
        raw_plane_losses = []
        for plane_cfg, target_field in zip(spec["planes"], target_fields):
            # Forward model for the current output plane
            output = system.measure_at_z(input_field, z=plane_cfg["z"])

            # 1️⃣ Overlap loss – encourages the field to match the target shape
            overlap_loss = 1.0 - output.inner(target_field).abs().square()

            # 2️⃣ Ratio loss – compare power split inside the ROIs to the target ratios
            powers = _roi_powers(output, plane_cfg["centers"], roi_radius)
            focus_power = powers.sum()
            total_power = output.intensity().sum() + 1e-12
            pred_ratios = powers / (focus_power + 1e-12)

            target_ratios = torch.tensor(
                plane_cfg["ratios"], dtype=torch.double, device=output.intensity().device
            )
            target_ratios = target_ratios / target_ratios.sum()

            ratio_loss = torch.mean(torch.abs(pred_ratios - target_ratios))

            # 3️⃣ Leakage loss – penalise energy outside the desired ROIs
            leakage_loss = 1.0 - focus_power / total_power

            # Weighted combination – give the ratio term higher priority
            raw_plane_losses.append(
                0.3 * overlap_loss + 0.6 * ratio_loss + 0.1 * leakage_loss
            )

        # Dynamically re‑weight planes: harder planes (higher loss) get larger weight.
        raw_losses_tensor = torch.stack(raw_plane_losses)
        plane_weights = torch.softmax(raw_losses_tensor / loss_weight_temp, dim=0)
        loss = (plane_weights * raw_losses_tensor).sum()
        loss.backward()
        optimizer.step()

        # Keep each phase within the physical interval [-π, π]
        with torch.no_grad():
            for layer in system:
                if hasattr(layer, "phase"):
                    layer.phase.data = (layer.phase.data + torch.pi) % (2 * torch.pi) - torch.pi

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
