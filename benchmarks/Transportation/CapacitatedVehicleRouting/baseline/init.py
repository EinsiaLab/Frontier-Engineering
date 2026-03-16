"""Capacitated Vehicle Routing - baseline solution.

Nearest-neighbour greedy heuristic. Each vehicle starts at the depot,
repeatedly picks the closest unvisited customer that still fits within
the remaining capacity, and returns to the depot when no more customers
can be added.

Source benchmark: Augerat et al. A-n32-k5 (CVRPLIB)
  https://github.com/PyVRP/PyVRP  (>1 000 stars)
  http://vrp.atd-lab.inf.puc-rio.br/index.php/en/
Known optimal distance: 784
"""
from __future__ import annotations

import math
from typing import Any


# ======================== EVOLVE-BLOCK-START ========================
def solve(instance: dict[str, Any]) -> list[list[int]]:
    """Solve CVRP and return a list of routes.

    Args:
        instance: dict with keys:
            - 'coords'   : list of (x, y) tuples; index 0 is the depot.
            - 'demands'  : list of int demands; index 0 is depot (demand 0).
            - 'capacity' : vehicle capacity (int).

    Returns:
        List of routes. Each route is a list of *customer* indices
        (1-indexed; depot index 0 is NOT included in routes).
        Every customer must appear in exactly one route.
        Each route must satisfy sum(demands[c] for c in route) <= capacity.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    n = len(coords) - 1  # number of customers

    def euc2d(i: int, j: int) -> int:
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        return int(math.sqrt(dx * dx + dy * dy) + 0.5)

    unvisited = set(range(1, n + 1))
    routes: list[list[int]] = []

    while unvisited:
        route: list[int] = []
        load = 0
        current = 0  # start at depot

        while True:
            best: int | None = None
            best_d = float("inf")
            for c in unvisited:
                if load + demands[c] <= capacity:
                    d = euc2d(current, c)
                    if d < best_d:
                        best_d = d
                        best = c
            if best is None:
                break
            route.append(best)
            load += demands[best]
            unvisited.remove(best)
            current = best

        if route:
            routes.append(route)

    return routes
# ======================== EVOLVE-BLOCK-END ========================
