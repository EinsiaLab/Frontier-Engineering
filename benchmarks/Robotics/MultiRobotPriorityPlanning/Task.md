# Multi-Robot Priority Planning Task

## Objective

Plan collision-free paths for three robots on a frozen occupancy grid while minimizing total path cost.

The benchmark uses one frozen multi-robot occupancy grid in `runtime/problem.py`.

## Submission Contract

Submit one Python file that defines:

```python
def plan_paths(grid, starts, goals):
    ...
```

The function must return a list of paths, one per robot. A dict with key `paths` is also accepted.

## Metrics

- `combined_score`: `-candidate_total_cost`
- `valid`: `1.0` only if all robot paths are collision-free
- `candidate_total_cost`
- `baseline_total_cost`
- `candidate_makespan`
- `lower_bound_total_cost`
