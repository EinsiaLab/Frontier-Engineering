# Narrow-Passage Planning Task

## Objective

Plan a collision-free path through a single-cell narrow passage on a frozen occupancy grid.

The benchmark uses one frozen occupancy grid in `runtime/problem.py`.

## Submission Contract

Submit one Python file that defines:

```python
def plan_path(grid, start, goal):
    ...
```

Inputs:

- `grid`: tuple of strings, where `#` means obstacle and `.` means free space
- `start`: `(x, y)`
- `goal`: `(x, y)`

The function must return a path as a sequence of `(x, y)` cells. A dict with key `path` is also accepted.

## Metrics

- `combined_score`: `-candidate_cost`
- `valid`: `1.0` only if the returned path is finite and collision-free
- `candidate_cost`
- `baseline_cost`
- `reference_cost`
