# Fuel-Minimizing Ship Weather Routing

Route a ship across a weather-routing case family and minimize hidden-case average fuel use under arrival constraints.

## What Changed

- The task now evaluates multiple public and hidden routing cases.
- The baseline is an explicit graph search balancing fuel and time.
- Scoring uses hidden-case average fuel, with deadlines enforced per case.

## What You Edit

- Target file: `scripts/init.py`
- Entry point: `solve(instance)`

## Source of Truth

- `Task.md`
- `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## Environment

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/OperationsResearch/FuelMinimizingShipWeatherRouting/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/OperationsResearch/FuelMinimizingShipWeatherRouting/verification/evaluator.py \
  benchmarks/OperationsResearch/FuelMinimizingShipWeatherRouting/scripts/init.py \
  --metrics-out /tmp/FuelMinimizingShipWeatherRouting_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_fuel`
- `valid`
- `public_avg_fuel`
- `hidden_avg_fuel`
- `baseline_hidden_avg_fuel`

<!-- AI_GENERATED -->
