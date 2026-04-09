# EVOLVE-BLOCK-START
"""Baseline solution for QuadrupedGaitOptimization.

Provides a conservative trot-like gait parameter set for MuJoCo evaluation.
"""

from __future__ import annotations

import json

# Bound gait - front legs together, rear legs together
# FL+FR in phase, RL+RR in phase with 0.5 cycle offset
submission = {
    "step_frequency": 2.5,      # Moderate frequency for stability
    "duty_factor": 0.42,        # Lower duty factor for dynamic motion
    "step_length": 0.28,        # Longer strides for speed
    "step_height": 0.11,        # Good clearance for bounding
    "phase_FR": 0.0,            # Front legs in phase with FL
    "phase_RL": 0.5,            # Rear legs half cycle offset
    "phase_RR": 0.5,            # Rear legs in phase with RL
    "lateral_distance": 0.14,   # Moderate stance width
}

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print("Baseline submission written to submission.json")
print(json.dumps(submission, indent=2))
# EVOLVE-BLOCK-END
