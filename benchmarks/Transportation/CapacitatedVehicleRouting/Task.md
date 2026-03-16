# Task: Capacitated Vehicle Routing — A-n32-k5

## Problem Statement

You must implement `solve(instance)` in `baseline/init.py`.

Given a depot (index 0) and 31 customers, each with a known demand, assign
customers to vehicle routes such that:

1. Every customer appears in **exactly one** route.
2. The total demand on each route does not exceed **capacity = 100**.
3. The total EUC_2D distance (depot → customers → depot, summed over all
   routes) is **minimised**.

## Interface

```python
def solve(instance: dict) -> list[list[int]]:
    ...
```

| Key         | Type              | Description                              |
|-------------|-------------------|------------------------------------------|
| `coords`    | `list[(int,int)]` | (x, y) pairs; index 0 = depot           |
| `demands`   | `list[int]`       | demand per node; demands[0] = 0          |
| `capacity`  | `int`             | vehicle capacity = 100                   |

Return a `list` of routes. Each route is a `list[int]` of customer indices
(1 … 31). Do **not** include the depot index 0.

## Evaluation

Distance uses EUC_2D: `round(sqrt((x1-x2)²+(y1-y2)²))`.

```
score = min(1.0, 784.0 / total_distance)
```

`valid = 1` requires: correct indices, no duplicate visits, no capacity
violation, all 31 customers covered.

## Human Best

Known optimal distance for A-n32-k5: **784** (score = 1.0).
Source: Augerat et al. (1995), http://vrp.atd-lab.inf.puc-rio.br/

## Baseline

Nearest-neighbour greedy heuristic (provided in `baseline/init.py`).
Typical result: distance ≈ 920, score ≈ 0.85.
