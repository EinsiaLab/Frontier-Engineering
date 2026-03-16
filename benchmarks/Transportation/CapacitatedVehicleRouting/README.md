# Capacitated Vehicle Routing (CVRP) — Augerat A-n32-k5

## Overview

Given a depot and **31 customers** with known locations and demands, route a
fleet of identical vehicles (capacity = 100) so that every customer is visited
exactly once, each vehicle stays within capacity, and the **total travel
distance is minimised**.

This benchmark uses the canonical **Augerat A-n32-k5** instance (32 nodes,
minimum 5 vehicles, EUC_2D distances). The known optimal tour length is **784**.

## Source

| Field        | Detail |
|--------------|--------|
| Benchmark    | CVRPLIB — Augerat et al. set A (1995) |
| Solver lib   | [PyVRP](https://github.com/PyVRP/PyVRP) (>1 000 ★) |
| Instance URL | http://vrp.atd-lab.inf.puc-rio.br/ |
| License      | Academic / public-domain instances |

## Input / Output

**Input** (`instance` dict passed to `solve()`):
- `coords` — list of 32 `(x, y)` integer tuples; index 0 is the depot.
- `demands` — list of 32 integer demands; `demands[0] = 0` (depot).
- `capacity` — vehicle capacity (100).

**Output**: `list[list[int]]` — each inner list is one route of customer
indices (1-indexed). The depot is implicit (routes start/end there).

## Scoring

```
score = min(1.0, 784 / total_distance)
```

- `valid = 1` when all capacity constraints hold and every customer appears
  exactly once.
- `combined_score = score` (higher is better; maximum = 1.0 at optimum).

## Human Best

| Metric         | Value |
|----------------|-------|
| Known optimal  | 784   |
| Source         | Augerat et al. (1995), confirmed by multiple exact solvers |

Baseline (nearest-neighbour greedy) achieves ≈ 900–950 → score ≈ 0.83–0.87.

## Quick Start

```bash
cd benchmarks/Transportation/CapacitatedVehicleRouting
pip install -r verification/requirements.txt
python verification/evaluate.py
```
