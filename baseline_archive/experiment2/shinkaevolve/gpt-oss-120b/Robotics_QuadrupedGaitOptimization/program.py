# EVOLVE-BLOCK-START
"""Optimized solution for QuadrupedGaitOptimization.

Uses a modular parametric gait model with physics-based heuristics for speed optimization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

@dataclass
class GaitConfig:
    """Container for quadruped gait parameters with semantic naming."""
    step_frequency: float      # Step cycle frequency in Hz [0.5, 4.0]
    duty_factor: float         # Fraction of cycle in stance [0.30, 0.85]
    step_length: float         # Forward step length in meters [0.04, 0.40]
    step_height: float         # Vertical clearance in meters [0.02, 0.15]
    phase_FR: float            # Phase offset for FR leg [0.0, 1.0)
    phase_RL: float            # Phase offset for RL leg [0.0, 1.0)
    phase_RR: float            # Phase offset for RR leg [0.0, 1.0)
    lateral_distance: float    # Lateral foot spread in meters [0.08, 0.20]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GaitOptimizer:
    """Generates optimized gait parameters for quadruped locomotion.
    
    Implements physics-based heuristics for speed optimization while
    maintaining stability constraints for the ant-style quadruped model.
    """
    
    # Parameter bounds from task specification
    BOUNDS: Dict[str, Tuple[float, float]] = {
        'step_frequency': (0.5, 4.0),
        'duty_factor': (0.30, 0.85),
        'step_length': (0.04, 0.40),
        'step_height': (0.02, 0.15),
        'phase_FR': (0.0, 1.0),
        'phase_RL': (0.0, 1.0),
        'phase_RR': (0.0, 1.0),
        'lateral_distance': (0.08, 0.20),
    }
    
    @classmethod
    def create_speed_optimized_trot(cls) -> GaitConfig:
        """Create an optimized trot gait for maximum forward speed.
        
        Trot gait characteristics:
        - Diagonal leg pairs move in synchronization
        - FL (front-left) and RR (rear-right) at phase 0.0
        - FR (front-right) and RL (rear-left) at phase 0.5
        - Provides good stability with efficient forward motion
        
        Speed optimization strategy:
        - Higher frequency increases step rate
        - Larger step_length covers more ground per step
        - Lower duty_factor allows more swing time
        - Moderate step_height ensures clearance without excess energy
        - Narrower lateral_distance improves efficiency
        """
        return GaitConfig(
            step_frequency=2.2,      # Increased for faster stepping
            duty_factor=0.45,        # Balanced stance/swing ratio
            step_length=0.35,        # Large step for ground coverage
            step_height=0.08,        # Adequate clearance
            phase_FR=0.5,            # Opposite to FL (trot pattern)
            phase_RL=0.5,            # Opposite to FL (trot pattern)
            phase_RR=0.0,            # Same as FL (diagonal pair)
            lateral_distance=0.13,   # Moderate stance width
        )
    
    @classmethod
    def validate_config(cls, config: GaitConfig) -> bool:
        """Validate that all parameters are within bounds."""
        for param, (low, high) in cls.BOUNDS.items():
            value = getattr(config, param)
            if param.startswith('phase'):
                # Phase uses half-open interval [0, 1)
                if not (low <= value < high):
                    return False
            else:
                if not (low <= value <= high):
                    return False
        return True


class SubmissionWriter:
    """Handles writing gait configurations to submission files."""
    
    @staticmethod
    def write(config: GaitConfig, filepath: str = "submission.json") -> None:
        """Write gait configuration to JSON file with proper formatting."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
    
    @staticmethod
    def display(config: GaitConfig) -> None:
        """Print the configuration to stdout."""
        print("Optimized submission written to submission.json")
        print(json.dumps(config.to_dict(), indent=2))


def main() -> None:
    """Main entry point for gait parameter generation."""
    # Generate optimized trot gait
    optimizer = GaitOptimizer()
    gait_config = optimizer.create_speed_optimized_trot()
    
    # Validate parameters
    if not optimizer.validate_config(gait_config):
        raise ValueError("Generated gait parameters are out of bounds")
    
    # Write and display submission
    writer = SubmissionWriter()
    writer.write(gait_config)
    writer.display(gait_config)


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END