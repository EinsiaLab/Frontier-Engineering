# EVOLVE-BLOCK-START
"""Optimized solver for Task 2: multi-plane focusing with class-based architecture."""

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


class MultiPlaneFocusingSolver:
    """Class-based solver for multi-plane optical focusing optimization."""
    
    def __init__(self, spec: dict[str, Any], device: str):
        self.spec = spec
        self.device = device
        self.shape = int(spec["shape"])
        self.waist = float(spec["waist_radius"])
        
        # Configure torchoptics global settings
        torchoptics.set_default_spacing(spec["spacing"])
        torchoptics.set_default_wavelength(spec["wavelength"])
        
        # Build all optical components
        self.system = self._build_system()
        self.input_field = self._build_input_field()
        self.target_fields = [self._build_target_field(p) for p in spec["planes"]]
        self.target_intensities = [tf.intensity() for tf in self.target_fields]
    
    def _build_system(self) -> System:
        """Build phase modulator system with random initialization."""
        layers = []
        for z in self.spec["layer_z"]:
            # Random phase initialization for better solution space exploration
            phase = torch.rand(
                (self.shape, self.shape),
                dtype=torch.double,
                device=self.device
            ) * 2 * torch.pi
            layers.append(PhaseModulator(Parameter(phase), z=float(z)))
        return System(*layers).to(self.device)
    
    def _build_input_field(self) -> Field:
        """Build normalized Gaussian input field."""
        return Field(
            gaussian(self.shape, self.waist), z=0
        ).normalize(1.0).to(self.device)
    
    def _build_target_field(self, plane_cfg: dict[str, Any]) -> Field:
        """Build target field for a specific plane configuration."""
        target = torch.zeros(
            (self.shape, self.shape),
            dtype=torch.double,
            device=self.device
        )
        ratios = torch.tensor(
            plane_cfg["ratios"],
            dtype=torch.double,
            device=self.device
        )
        ratios = ratios / ratios.sum()
        
        for ratio, center in zip(ratios, plane_cfg["centers"]):
            target += torch.sqrt(ratio) * gaussian(
                self.shape, self.waist, offset=center
            ).real.to(self.device)
        
        return Field(target.to(torch.cdouble), z=plane_cfg["z"]).normalize(1.0)
    
    def _compute_plane_loss(self, output: Field, target: Field, target_intensity: torch.Tensor) -> torch.Tensor:
        """Compute multi-component loss for a single plane."""
        # Primary: overlap loss for shape and phase matching
        overlap = output.inner(target).abs().square()
        overlap_loss = 1.0 - overlap
        
        # Secondary: intensity correlation for efficiency improvement
        output_intensity = output.intensity()
        intensity_corr = (output_intensity * target_intensity).sum() / (
            output_intensity.norm() * target_intensity.norm() + 1e-10
        )
        intensity_loss = 1.0 - intensity_corr
        
        # Weighted combination emphasizing overlap
        return 0.7 * overlap_loss + 0.3 * intensity_loss
    
    def optimize(self) -> list[float]:
        """Execute optimization loop with adaptive learning rate."""
        optimizer = torch.optim.Adam(
            self.system.parameters(),
            lr=float(self.spec["lr"])
        )
        
        # Cosine annealing scheduler for smooth convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.spec["steps"]),
            eta_min=float(self.spec["lr"]) * 0.01
        )
        
        losses: list[float] = []
        
        for _ in range(int(self.spec["steps"])):
            optimizer.zero_grad()
            
            # Compute loss for each target plane
            plane_losses = []
            for plane_cfg, target_field, target_intensity in zip(
                self.spec["planes"],
                self.target_fields,
                self.target_intensities
            ):
                output = self.system.measure_at_z(self.input_field, z=plane_cfg["z"])
                plane_losses.append(
                    self._compute_plane_loss(output, target_field, target_intensity)
                )
            
            loss = torch.stack(plane_losses).mean()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.system.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        
        return losses
    
    def get_results(self, losses: list[float]) -> dict[str, Any]:
        """Package optimization results."""
        return {
            "spec": self.spec,
            "system": self.system,
            "input_field": self.input_field,
            "target_fields": self.target_fields,
            "loss_history": losses,
        }


def solve(spec: dict[str, Any] | None = None, device: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Main entry point for multi-plane focusing solver."""
    spec = {**make_default_spec(), **(spec or {})}
    torch.manual_seed(seed)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    solver = MultiPlaneFocusingSolver(spec, device)
    losses = solver.optimize()
    return solver.get_results(losses)
# EVOLVE-BLOCK-END
