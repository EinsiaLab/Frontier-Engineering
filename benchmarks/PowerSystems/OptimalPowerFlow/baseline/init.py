"""Power Systems DC Optimal Power Flow - baseline solution.

Equal-dispatch heuristic: distributes total load equally across all 5
generators. This guarantees power balance and generation limits are met,
but ignores network topology, congestion, and cost minimisation.

Source: IEEE 14-bus test system
  MATPOWER - https://github.com/MATPOWER/matpower  (>1 000 stars, BSD)
  PGLib-OPF - https://github.com/power-grid-lib/pglib-opf  (>500 stars, CC-BY)
Known optimal DC-OPF cost (with line thermal limits): 7892.76 $/h
"""
from __future__ import annotations

from typing import Any


# ======================== EVOLVE-BLOCK-START ========================
def solve(instance: dict[str, Any]) -> list[float]:
    """Solve the DC Optimal Power Flow and return generator active power outputs.

    Args:
        instance: dict with keys:
            'base_mva' (float): system MVA base (100 MVA).
            'generators': list of 5 dicts, each with:
                'bus'    (int)   - bus number (1-indexed)
                'pmin'   (float) - minimum active power output (MW)
                'pmax'   (float) - maximum active power output (MW)
                'cost_a' (float) - quadratic cost coefficient ($/MW^2/h)
                'cost_b' (float) - linear cost coefficient ($/MWh)
            'buses': list of 14 dicts, each with:
                'id' (int)  - bus number (1-indexed)
                'pd' (float)- active power demand (MW)
            'branches': list of 20 dicts, each with:
                'from_bus' (int)   - sending-end bus (1-indexed)
                'to_bus'   (int)   - receiving-end bus (1-indexed)
                'x'        (float) - series reactance (per-unit, 100 MVA base)
                'rate_a'   (float) - thermal rating (MW)

    Returns:
        list[float] of length 5 - generator active power outputs in MW,
        in the same order as instance['generators'].

    Constraints that the evaluator enforces:
        1. Power balance: |sum(Pg) - total_load| <= 0.5 MW
        2. Generation limits: pmin[i] <= Pg[i] <= pmax[i] for all i
        3. DC line flows: |P_ij| <= rate_a for all branches
           (computed via DC power flow: B*theta = P_net)
    """
    generators = instance["generators"]
    buses = instance["buses"]

    # Total active load
    total_load = sum(b["pd"] for b in buses)
    n_gen = len(generators)

    # Baseline: equal dispatch - ignores cost structure and network
    Pg_each = total_load / n_gen
    return [
        min(g["pmax"], max(g["pmin"], Pg_each)) for g in generators
    ]
# ======================== EVOLVE-BLOCK-END ========================
