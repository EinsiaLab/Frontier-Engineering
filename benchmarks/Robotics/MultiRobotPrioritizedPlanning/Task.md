# Multi-Robot Prioritized Planning Task

## Problem

Plan collision-free multi-robot paths on a family of grid cases and minimize hidden-case average total cost.

The evaluator now uses multiple public and hidden multi-robot maps. Each case fixes the grid, robot starts, and robot goals, while you provide the path set. The scoring objective is total path cost, with makespan reported as an additional diagnostic.

## What Is Frozen

- The public and hidden grid cases in `runtime/problem.py`.
- The robot start/goal assignments extracted from each case.
- The collision rules: vertex collisions and edge-swap collisions are both illegal.
- The cost definitions for total cost and makespan.

## Submission Contract

Submit one Python file that defines:

```python
def plan_paths(grid, starts, goals):
    ...
```

Return a list of per-robot paths. A dict with key `paths` is also accepted.

## Evaluation

1. Load each public and hidden case.
2. Call `plan_paths(grid, starts, goals)` on every case.
3. Validate per-robot endpoints, adjacency, obstacle avoidance, vertex collisions, and edge-swap collisions.
4. Aggregate total cost across cases; scoring uses hidden-case average total cost.

## Metrics

- `combined_score`: `-hidden_avg_total_cost`
- `valid`: `1.0` only if all cases return valid collision-free path sets
- `public_avg_total_cost`
- `hidden_avg_total_cost`
- `baseline_hidden_avg_total_cost`
- `hidden_avg_makespan`
- `num_public_cases`
- `num_hidden_cases`

## Invalid Submissions

- `plan_paths(...)` is missing or crashes
- The returned value cannot be parsed into per-robot paths
- Any robot path has the wrong start or goal
- Any path contains a non-adjacent move or enters an obstacle
- Any vertex collision or edge-swap collision occurs
- Any public or hidden case fails during evaluation

<!-- AI_GENERATED -->
