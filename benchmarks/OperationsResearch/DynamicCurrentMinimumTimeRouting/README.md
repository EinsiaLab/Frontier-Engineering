# Dynamic-Current Minimum-Time Routing

Route a ship across a routing case family and minimize hidden-case average travel time under current and draft constraints.

## What Changed

- The task now evaluates multiple public and hidden maps.
- The baseline is an explicit shortest-time graph search, not a runtime helper export.
- Scoring uses hidden-case average travel time.

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
pip install -r benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/verification/evaluator.py \
  benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/scripts/init.py \
  --metrics-out /tmp/DynamicCurrentMinimumTimeRouting_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_time_h`
- `valid`
- `public_avg_time_h`
- `hidden_avg_time_h`
- `baseline_hidden_avg_time_h`

<!-- AI_GENERATED -->
