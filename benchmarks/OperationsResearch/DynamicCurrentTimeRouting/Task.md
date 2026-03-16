# Dynamic Current Time Routing Task

## Objective

Route a ship across a frozen coastal grid while minimizing travel time under deterministic current and depth fields.

## Submission Contract

Submit one Python file that defines:

```python
def solve(instance):
    ...
```

Return either a list of grid cells or a dict with key `path`.

The path must:

1. Start at `instance["start"]`
2. End at `instance["goal"]`
3. Move only between adjacent grid cells
4. Stay on water cells with depth at least `instance["min_depth"]`

## Fixed World Model

- The map, synthetic current field, and synthetic depth raster are fixed in `runtime/problem.py`.
- The upstream lineage is dynamic-current minimum-time routing from `HALEM`, but the actual environmental data here is benchmark-local synthetic data with a fixed generator.

## Evaluation

The evaluator will:

1. Load the frozen routing instance
2. Validate your path against the land mask and minimum-depth rule
3. Compute travel time along the route
4. Log the shortest-hop baseline and Dijkstra reference metrics for context while scoring candidate travel time directly

## Metrics

- `combined_score`: `-candidate_time_h`
- `valid`: `1.0` only if the route is feasible
- `candidate_time_h`
- `baseline_time_h`
- `reference_time_h`
- `candidate_hops`
- `baseline_hops`
