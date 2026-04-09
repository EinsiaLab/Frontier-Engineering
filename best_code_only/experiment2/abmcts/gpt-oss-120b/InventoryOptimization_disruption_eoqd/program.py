# EVOLVE-BLOCK-START
"""Baseline implementation for Task 05.

No stockpyl EOQD optimizer is used here.
"""

from __future__ import annotations

import math
from typing import Tuple


def classic_eoq(fixed_cost: float, holding_cost: float, demand_rate: float) -> float:
    """Classic Economic Order Quantity (EOQ) formula."""
    if holding_cost == 0:
        raise ValueError("holding_cost must be non‑zero")
    return math.sqrt(2.0 * fixed_cost * demand_rate / holding_cost)


def _disruption_impact(disruption_rate: float, recovery_rate: float) -> float:
    """
    Normalized disruption impact in the range [0, 1].

    The impact rises with a higher disruption rate and falls with a higher
    recovery rate.  If both rates are zero the impact is defined as zero.
    """
    denominator = disruption_rate + recovery_rate
    if denominator == 0.0:
        return 0.0
    return disruption_rate / denominator


def _safety_multiplier(impact: float) -> float:
    """
    Convert a disruption impact into a safety multiplier.

    The mapping is deliberately non‑linear: low impact values lead to a modest
    increase, while high impact values trigger a stronger (but still bounded)
    safety boost.  The exponent 0.6 provides a gentle curve, and the factor
    0.9 caps the maximum multiplier at roughly 1.9.
    """
    # Ensure impact stays within [0, 1] even if caller supplies out‑of‑range values.
    impact = max(0.0, min(1.0, impact))
    return 1.0 + 0.9 * (impact ** 0.6)


def solve(cfg: dict) -> Tuple[float, float, float]:
    """
    Compute EOQ values and a disruption‑aware safety multiplier.

    Returns
    -------
    tuple
        (q_classic, q_manual, safety_multiplier)
        * q_classic – classic EOQ without disruption considerations.
        * q_manual – EOQ adjusted by the safety multiplier.
        * safety_multiplier – factor applied to q_classic (>= 1).
    """
    q_classic = classic_eoq(
        cfg["fixed_cost"], cfg["holding_cost"], cfg["demand_rate"]
    )

    impact = _disruption_impact(cfg["disruption_rate"], cfg["recovery_rate"])
    safety_multiplier = _safety_multiplier(impact)

    q_manual = q_classic * safety_multiplier
    return q_classic, q_manual, safety_multiplier
# EVOLVE-BLOCK-END
