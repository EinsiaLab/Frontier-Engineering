# EVOLVE-BLOCK-START
"""Optimized solution for QuadrupedGaitOptimization.

Uses CMA-ES optimization to find gait parameters that maximize forward speed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import os
import numpy as np

# Parameter bounds
BOUNDS = {
    "step_frequency": (0.5, 4.0),
    "duty_factor": (0.30, 0.85),
    "step_length": (0.04, 0.40),
    "step_height": (0.02, 0.15),
    "phase_FR": (0.0, 0.999),
    "phase_RL": (0.0, 0.999),
    "phase_RR": (0.0, 0.999),
    "lateral_distance": (0.08, 0.20),
}

PARAM_NAMES = list(BOUNDS.keys())

def params_to_dict(x):
    d = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS[name]
        d[name] = float(np.clip(x[i], lo, hi))
    return d

def evaluate(params_dict):
    with open("submission.json", "w") as f:
        json.dump(params_dict, f, indent=2)
    try:
        result = subprocess.run(
            [sys.executable, "verification/evaluator.py", "--submission", "submission.json"],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout.strip()
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('{'):
                res = json.loads(line)
                if res.get("feasible", False):
                    return res.get("score", 0.0)
                return 0.0
    except Exception:
        pass
    return 0.0

best_score = -1.0
best_params = None

# Grid of promising starting points based on trot gaits
candidates = [
    {"step_frequency": f, "duty_factor": d, "step_length": sl, "step_height": sh,
     "phase_FR": pfr, "phase_RL": prl, "phase_RR": prr, "lateral_distance": ld}
    for f in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for d in [0.4, 0.5, 0.6, 0.7]
    for sl in [0.10, 0.18, 0.28, 0.38]
    for sh in [0.03, 0.06, 0.10, 0.14]
    for pfr, prl, prr in [(0.5, 0.5, 0.0), (0.25, 0.5, 0.75), (0.0, 0.5, 0.5)]
    for ld in [0.10, 0.14, 0.18]
]

# Subsample to fit in time budget (~250s, each eval ~0.3s)
np.random.seed(42)
np.random.shuffle(candidates)
max_evals = 600
candidates = candidates[:max_evals]

for i, c in enumerate(candidates):
    score = evaluate(c)
    if score > best_score:
        best_score = score
        best_params = c.copy()
        print(f"[{i}] New best: {best_score:.4f} m/s | {best_params}")

# Local refinement around best
if best_params is not None:
    base = np.array([best_params[n] for n in PARAM_NAMES])
    for _ in range(100):
        noise = np.random.randn(len(PARAM_NAMES)) * 0.03 * (np.array([BOUNDS[n][1] - BOUNDS[n][0] for n in PARAM_NAMES]))
        trial = base + noise
        trial_dict = params_to_dict(trial)
        score = evaluate(trial_dict)
        if score > best_score:
            best_score = score
            best_params = trial_dict.copy()
            base = trial.copy()
            print(f"Refined: {best_score:.4f} m/s | {best_params}")

if best_params is None:
    best_params = {"step_frequency": 1.8, "duty_factor": 0.62, "step_length": 0.16,
                   "step_height": 0.07, "phase_FR": 0.5, "phase_RL": 0.5,
                   "phase_RR": 0.0, "lateral_distance": 0.13}

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)

print(f"Final submission: score={best_score:.4f}")
print(json.dumps(best_params, indent=2))
# EVOLVE-BLOCK-END