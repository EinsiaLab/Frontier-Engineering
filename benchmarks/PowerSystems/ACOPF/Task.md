# Task: AC Optimal Power Flow (5-bus DC-OPF)

## Task Name

`acopf`

## Description

Implement a `solve(instance)` function that returns a feasible generation dispatch minimizing total generation cost for the given 5-bus DC-OPF instance. The instance contains bus/branch/gen data and cost coefficients; output is `total_cost` (scalar) or a dict that allows computing total cost.

## Interface

```python
def solve(instance: dict) -> dict:
    """
    Args:
        instance: dict with 'n_bus', 'B', 'P_load', 'Pgen_min', 'Pgen_max', 'cost_c0', 'cost_c1', 'cost_c2', 'gen_bus'
    Returns:
        {'total_cost': float} or {'Pg': list[float], ...} so total_cost can be computed
    """
```

## Scoring

```
combined_score = min(1.0, HUMAN_BEST_COST / total_cost)
HUMAN_BEST_COST = 26.0
```

Valid: power balance satisfied, Pgen within [Pgen_min, Pgen_max]. Invalid → score 0.

## Baseline / Human Best

- Baseline: naive dispatch (e.g. equal sharing) → valid, score > 0.
- Human best: 26.0 (optimal for embedded case).
