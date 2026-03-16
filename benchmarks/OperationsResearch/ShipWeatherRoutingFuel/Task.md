# Ship Weather Routing Fuel Task

## Objective

Route a ship across a frozen coastal grid while minimizing total fuel consumption under synthetic wind and current fields.

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
4. Stay on navigable water cells

## Fixed World Model

- The map, start/goal pair, synthetic wind field, and synthetic current field are fixed in `runtime/problem.py`.
- The upstream lineage is weather-aware ship routing from `WeatherRoutingTool`, but the actual grid data here is benchmark-local synthetic data with a fixed generator.

## Evaluation

The evaluator will:

1. Load the frozen routing instance
2. Validate your path mechanically
3. Compute total fuel use and travel time along the path
4. Log the shortest-hop baseline and Dijkstra reference metrics for context while scoring candidate fuel directly

## Metrics

- `combined_score`: `-candidate_fuel`
- `valid`: `1.0` only if the route is feasible
- `candidate_fuel`
- `baseline_fuel`
- `reference_fuel`
- `candidate_time_h`
- `baseline_time_h`
