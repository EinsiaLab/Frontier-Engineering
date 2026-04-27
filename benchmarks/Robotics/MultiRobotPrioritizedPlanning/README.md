# Multi-Robot Prioritized Planning

Plan collision-free multi-robot paths on a grid-case family and minimize hidden-case average total cost.

## What Changed

- The evaluator now uses multiple public and hidden multi-robot grids.
- The baseline is an explicit prioritized planner, not a runtime-exported fixed solution.
- Scoring uses hidden-case average total cost, with makespan reported separately.

## What You Edit

- Target file: `scripts/init.py`
- Entry point: `plan_paths(grid, starts, goals)`

## Source of Truth

- `Task.md`
- `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## Environment

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/Robotics/MultiRobotPrioritizedPlanning/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/Robotics/MultiRobotPrioritizedPlanning/verification/evaluator.py \
  benchmarks/Robotics/MultiRobotPrioritizedPlanning/scripts/init.py \
  --metrics-out /tmp/MultiRobotPrioritizedPlanning_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_total_cost`
- `valid`
- `public_avg_total_cost`
- `hidden_avg_total_cost`
- `baseline_hidden_avg_total_cost`
- `hidden_avg_makespan`

<!-- AI_GENERATED -->
