#!/usr/bin/env python3
"""Evaluate a CVRP solution on the Augerat A-n32-k5 benchmark instance.

Invoke from the benchmark root directory:
    python verification/evaluate.py

The evaluator imports baseline.init.solve(instance) -> list[list[int]].
It validates feasibility (capacity, full coverage, no duplicate visits)
and computes a normalised score in [0, 1]:

    score = min(1.0, HUMAN_BEST / total_distance)

where HUMAN_BEST = 784 (known optimal for A-n32-k5, Augerat et al. 1995).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

# ---------------------------------------------------------------------------
# A-n32-k5 instance (Augerat et al., 1995) — EUC_2D, capacity = 100
# Node 1 of the VRPLIB file is the depot; re-indexed here to 0.
# Customers are indices 1 … 31.
# ---------------------------------------------------------------------------
INSTANCE: dict[str, Any] = {
    "coords": [
        (82, 76),  # 0 — depot
        (96, 44), (50,  5), (49,  8), (13,  7), (29, 89),
        (58, 30), (84, 39), (14, 24), ( 2, 39), ( 3, 82),
        ( 5, 10), (98, 52), (84, 25), (61, 59), ( 1, 65),
        (88, 51), (91,  2), (19, 32), (93,  3), (50, 93),
        (98, 14), ( 5, 42), (42,  9), (61, 62), ( 9, 97),
        (80, 55), (57, 69), (23, 15), (20, 70), (85, 60),
        (98,  5),
    ],
    "demands": [
         0,                                          # depot
        19, 21,  6, 19,  7, 12, 16,  6, 16,  8,
        14, 21, 16,  3, 22, 18, 19,  1, 24,  8,
        12,  4,  8, 24, 24,  2, 20, 15,  2, 14,  9,
    ],
    "capacity": 100,
}

HUMAN_BEST: float = 784.0  # known optimal distance for A-n32-k5


def euc2d(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def validate_and_score(
    routes: list[list[int]], instance: dict[str, Any]
) -> dict[str, Any]:
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    n_customers = len(coords) - 1

    all_customers = set(range(1, n_customers + 1))
    visited: set[int] = set()
    errors: list[str] = []
    total_dist = 0.0
    n_routes = 0

    for i, route in enumerate(routes):
        if not route:
            continue
        n_routes += 1
        load = 0
        for c in route:
            if not isinstance(c, int) or c < 1 or c > n_customers:
                errors.append(f"Route {i}: invalid customer index {c!r}")
                continue
            if c in visited:
                errors.append(f"Route {i}: customer {c} visited more than once")
            visited.add(c)
            load += demands[c]

        if load > capacity:
            errors.append(
                f"Route {i}: capacity exceeded ({load} > {capacity})"
            )

        path = [0] + list(route) + [0]
        for j in range(len(path) - 1):
            total_dist += euc2d(coords[path[j]], coords[path[j + 1]])

    missing = all_customers - visited
    if missing:
        errors.append(f"Missing customers: {sorted(missing)}")

    valid = len(errors) == 0
    return {
        "valid": valid,
        "errors": errors,
        "total_distance": total_dist,
        "n_routes": n_routes,
    }


def main() -> None:
    output_dir = TASK_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    from baseline.init import solve  # noqa: E402

    routes = solve(INSTANCE)
    result = validate_and_score(routes, INSTANCE)
    dist = result["total_distance"]
    valid = result["valid"]

    if valid and dist > 0:
        score = min(1.0, HUMAN_BEST / dist)
    else:
        score = 0.0

    comparison: dict[str, Any] = {
        "task": "capacitated_vehicle_routing",
        "baseline_final_score": score,
        "baseline_distance": dist,
        "human_best_distance": HUMAN_BEST,
        "n_routes": result["n_routes"],
        "valid": valid,
        "errors": result["errors"],
    }

    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    print(f"Valid:            {valid}")
    if result["errors"]:
        for e in result["errors"]:
            print(f"  ERROR: {e}")
    print(f"Distance:         {dist:.1f}  (human best: {HUMAN_BEST})")
    print(f"Routes:           {result['n_routes']}")
    print(f"Score:            {score:.6f}")
    print(json.dumps({"valid": int(valid), "score": score, "combined_score": score}))


if __name__ == "__main__":
    main()
