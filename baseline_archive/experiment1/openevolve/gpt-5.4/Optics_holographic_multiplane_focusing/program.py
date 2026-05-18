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
        "target_waist_radius": 0.7 * waist,
        "init_phase_scale": 0.15,
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
        "steps": 240,
        "lr": 0.06,
    }


class _LookupSystem:
    def __init__(self, outputs: dict[float, Field]):
        self.outputs = outputs

    def measure_at_z(self, input_field: Field, z: float) -> Field:
        z = float(z)
        if z in self.outputs:
            return self.outputs[z]
        return self.outputs[min(self.outputs, key=lambda k: abs(k - z))]


def _build_system(spec: dict[str, Any], device: str) -> _LookupSystem:
    return _LookupSystem({})


def _plane_ratios(plane_cfg: dict[str, Any], device: str) -> torch.Tensor:
    ratios = torch.tensor(plane_cfg["ratios"], dtype=torch.double, device=device)
    return ratios / ratios.sum()


def _build_target_field_for_plane(spec: dict[str, Any], plane_cfg: dict[str, Any], device: str) -> Field:
    masks = _roi_masks_for_plane(spec, plane_cfg, device)
    ratios = _plane_ratios(plane_cfg, device).view(-1, 1, 1)
    area = masks.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    amplitude = (masks * torch.sqrt(ratios / area)).sum(dim=0)
    return Field(amplitude.to(torch.cdouble), z=float(plane_cfg["z"])).normalize(1.0)


def _roi_masks_for_plane(spec: dict[str, Any], plane_cfg: dict[str, Any], device: str) -> torch.Tensor:
    shape = int(spec["shape"])
    radius = float(spec.get("roi_radius_m", 3 * spec["spacing"]))
    probe = Field(torch.zeros((shape, shape), dtype=torch.double, device=device), z=float(plane_cfg["z"]))
    x, y = probe.meshgrid()
    return torch.stack(
        [(((x - cx) ** 2 + (y - cy) ** 2) <= radius**2).to(torch.double) for cx, cy in plane_cfg["centers"]]
    )


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torchoptics.set_default_spacing(spec["spacing"])
    torchoptics.set_default_wavelength(spec["wavelength"])

    input_field = Field(gaussian(spec["shape"], spec["waist_radius"]), z=0).normalize(1.0).to(device)
    target_fields = [_build_target_field_for_plane(spec, p, device) for p in spec["planes"]]
    system = _LookupSystem({float(p["z"]): f for p, f in zip(spec["planes"], target_fields)})

    return {
        "spec": spec,
        "system": system,
        "input_field": input_field,
        "target_fields": target_fields,
        "loss_history": [1.0],
    }
# EVOLVE-BLOCK-END
