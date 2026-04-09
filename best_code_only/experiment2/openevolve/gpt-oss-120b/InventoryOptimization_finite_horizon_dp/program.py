# EVOLVE-BLOCK-START
"""Baseline implementation for Task 04.

No stockpyl DP solver is used here.
"""

from __future__ import annotations

# Tunable heuristic constants – chosen to give a tighter, variability‑aware policy.
SAFETY_MULT = 0.40          # multiplier of σ used for safety stock (lower → less holding)
BUFFER_FACTOR = 0.50        # fraction of σ added as a dynamic buffer
MIN_GAP = 4                 # minimum required gap between S and s (tighter control)

def _calc_levels(mean: float, sd: float) -> tuple[int, int]:
    """
    Compute (s, S) for a single period with a simple
    coefficient‑of‑variation (CV) adjustment.

    * CV = sd / mean (0 when mean ≤ 0)
    * safety multiplier = SAFETY_MULT * (1 + CV)
    * buffer   = max(1, round(BUFFER_FACTOR * sd * (1 + CV)))
    * s = max(0, round(mean - safety multiplier * sd))
    * S = round(mean + safety multiplier * sd + buffer)
    * enforce S ≥ s + MIN_GAP
    """
    # Avoid division‑by‑zero; treat zero/negative mean as no variability.
    cv = sd / mean if mean > 0 else 0.0

    # Scale safety stock and buffer with demand variability.
    safety_mult = SAFETY_MULT * (1.0 + cv)

    # Reorder point – never negative.
    s = max(0, round(mean - safety_mult * sd))

    # Dynamic buffer – at least 1 unit to avoid zero buffer.
    buffer = max(1, round(BUFFER_FACTOR * sd * (1.0 + cv)))

    # Order‑up‑to level before enforcing the minimum gap.
    S = round(mean + safety_mult * sd + buffer)

    # Enforce a minimum gap between S and s.
    if S < s + MIN_GAP:
        S = s + MIN_GAP
    return s, S


def solve(demand_mean, demand_sd):
    """Variable‑buffer (s, S) policy.

    Returns two parallel lists: s_levels, S_levels.
    """
    # Guard against empty inputs.
    if not demand_mean:
        return [], []

    s_levels: list[int] = []
    S_levels: list[int] = []

    for m, sd in zip(demand_mean, demand_sd):
        s, S = _calc_levels(m, sd)
        s_levels.append(s)
        S_levels.append(S)

    return s_levels, S_levels
# EVOLVE-BLOCK-END
