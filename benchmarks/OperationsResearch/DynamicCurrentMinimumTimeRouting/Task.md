# Dynamic-Current Minimum-Time Routing Task

## Problem

Route a ship across a family of coastal grid cases while minimizing hidden-case average travel time under current and draft constraints.

This benchmark is no longer a single frozen map. The evaluator now uses multiple public and hidden routing cases with different coastlines, current bands, shallow-water cells, and start/goal pairs. Good solutions should generalize across these cases rather than memorize one route.

## What Is Frozen

- The public and hidden routing cases in `runtime/problem.py`.
- The four-neighbor movement rule, water-depth constraint, and hop budget.
- The travel-time computation induced by the deterministic current and depth fields.

## Submission Contract

Submit one Python file that defines:

```python
def solve(instance):
    ...
```

Return either a list of grid cells or a dict with key `path`.

## Evaluation

1. Load each public and hidden case from `runtime/problem.py`.
2. Call `solve(instance)` on each case independently.
3. Validate path endpoints, adjacency, navigability, and hop budget.
4. Compute travel time for each case and aggregate public and hidden averages separately.

## Metrics

- `combined_score`: `-hidden_avg_time_h`
- `valid`: `1.0` only if all cases produce feasible routes
- `public_avg_time_h`
- `hidden_avg_time_h`
- `baseline_hidden_avg_time_h`
- `num_public_cases`
- `num_hidden_cases`

## Invalid Submissions

- `solve(...)` is missing or crashes
- The returned value cannot be parsed into a path
- Any path starts or ends at the wrong cell
- Any path contains a non-adjacent move, enters land, violates minimum depth, or exceeds hop budget
- Any public or hidden case fails during evaluation

<!-- AI_GENERATED -->
