# EVOLVE-BLOCK-START
"""Optimized bound gait for maximum forward speed.

Bound gait synchronizes front legs (FL & FR) together and hind legs (RL & RR) together
with 0.5 phase offset. Using parameters from the best-performing bound gait (score 0.0799)
to maximize nominal speed while maintaining feasibility.
"""

from __future__ import annotations

import json

submission = {
    "step_frequency": 2.5,      # Higher frequency for faster movement
    "duty_factor": 0.35,       # Lower duty factor for longer swing phase
    "step_length": 0.30,       # Longer step for higher nominal speed
    "step_height": 0.10,       # Moderate step height for clearance
    "phase_FR": 0.0,           # Front legs synchronized (FL=0.0, FR=0.0)
    "phase_RL": 0.5,           # Hind legs synchronized with 0.5 offset
    "phase_RR": 0.5,           # RR synchronized with RL
    "lateral_distance": 0.15,   # Slightly narrower stance for better forward motion
}

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print("High-speed bound gait written to submission.json")
print(json.dumps(submission, indent=2))
# EVOLVE-BLOCK-END
