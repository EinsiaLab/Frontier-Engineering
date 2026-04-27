# Grid Path Planning with Obstacles Task

## Problem

Plan collision-free paths on a family of 2D occupancy grids and minimize hidden-case average path cost.

This benchmark no longer uses a single frozen map. The evaluator now runs multiple public and hidden grids with different corridor layouts and obstacle bottlenecks. The goal is to return valid paths that remain short across the full case family.

## What Is Frozen

- The public and hidden grid cases in `runtime/problem.py`.
- The movement rule: each step must stay in free space and move between adjacent cells.
- The path-cost definition: length minus one.

## Submission Contract

Submit one Python file that defines:

```python
def plan_path(grid, start, goal):
    ...
```

Return a path as a sequence of `(x, y)` cells. A dict with key `path` is also accepted.

## Evaluation

1. Load each public and hidden grid case.
2. Call `plan_path(grid, start, goal)` independently on every case.
3. Validate endpoints, adjacency, and obstacle avoidance.
4. Aggregate path cost across cases; scoring uses the hidden-case average.

## Metrics

- `combined_score`: `-hidden_avg_cost`
- `valid`: `1.0` only if all cases return valid collision-free paths
- `public_avg_cost`
- `hidden_avg_cost`
- `baseline_hidden_avg_cost`
- `num_public_cases`
- `num_hidden_cases`

## Invalid Submissions

- `plan_path(...)` is missing or crashes
- The returned value cannot be parsed into a path
- Any path has the wrong start or goal
- Any path contains a non-adjacent move or enters an obstacle
- Any public or hidden case fails during evaluation

<!-- AI_GENERATED -->
