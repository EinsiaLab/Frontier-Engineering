# EVOLVE-BLOCK-START
"""Baseline solution for QuadrupedGaitOptimization.

Provides an optimized trot gait parameter set for MuJoCo evaluation.
"""

from __future__ import annotations

import json

submission = {
    "step_frequency": 2.8,
    "duty_factor": 0.45,
    "step_length": 0.22,
    "step_height": 0.07,
    "phase_FR": 0.5,
    "phase_RL": 0.5,
    "phase_RR": 0.0,
    "lateral_distance": 0.145,
}

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print("Optimized trot submission written to submission.json")
print(json.dumps(submission, indent=2))
# EVOLVE-BLOCK-END
