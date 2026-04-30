# Narrow Passage Planning

Plan collision-free paths through a narrow-passage case family and minimize hidden-case average path cost.

## What Changed

- The evaluator now uses multiple public and hidden bottleneck maps.
- The baseline is an explicit A* planner, not a runtime-exported path.
- Scoring uses hidden-case average cost.

## What You Edit

- Target file: `scripts/init.py`
- Entry point: `plan_path(grid, start, goal)`

## Source of Truth

- `Task.md`
- `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## Environment

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/Robotics/NarrowPassagePlanning/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/Robotics/NarrowPassagePlanning/verification/evaluator.py \
  benchmarks/Robotics/NarrowPassagePlanning/scripts/init.py \
  --metrics-out /tmp/NarrowPassagePlanning_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_cost`
- `valid`
- `public_avg_cost`
- `hidden_avg_cost`
- `baseline_hidden_avg_cost`

<!-- AI_GENERATED -->
