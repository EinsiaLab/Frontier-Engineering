# Fuel-Minimizing Ship Weather Routing Task

## Problem

Route a ship across a family of weather-routing cases while minimizing hidden-case average fuel use under a latest-arrival constraint.

The evaluator now uses multiple public and hidden cases with different wind bands, current patterns, coastlines, and arrival budgets. A good method should balance fuel and travel time across the full case family rather than optimize one frozen map.

## What Is Frozen

- The public and hidden routing cases in `runtime/problem.py`.
- The four-neighbor movement rule and land mask.
- The per-leg fuel and travel-time model, including headwind/current effects.
- The latest-arrival constraint for each case.

## Submission Contract

Submit one Python file that defines:

```python
def solve(instance):
    ...
```

Return either a list of grid cells or a dict with key `path`.

## Evaluation

1. Load each public and hidden case from `runtime/problem.py`.
2. Call `solve(instance)` independently on every case.
3. Validate geometry, adjacency, and legality of the returned route.
4. Compute fuel and travel time for each case; reject routes that miss the case deadline.

## Metrics

- `combined_score`: `-hidden_avg_fuel`
- `valid`: `1.0` only if all cases produce feasible on-time routes
- `public_avg_fuel`
- `hidden_avg_fuel`
- `baseline_hidden_avg_fuel`
- `num_public_cases`
- `num_hidden_cases`

## Invalid Submissions

- `solve(...)` is missing or crashes
- The returned value cannot be parsed into a path
- Any route enters land or contains a non-adjacent move
- Any route misses the latest-arrival constraint
- Any public or hidden case fails during evaluation

<!-- AI_GENERATED -->
