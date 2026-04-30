# Grid Path Planning with Obstacles

Plan collision-free paths on a grid-case family and minimize hidden-case average path cost.

## What Changed

- The evaluator now runs multiple public and hidden occupancy grids.
- The baseline is an explicit A* planner, not a single frozen path.
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
pip install -r benchmarks/Robotics/GridPathPlanningWithObstacles/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/Robotics/GridPathPlanningWithObstacles/verification/evaluator.py \
  benchmarks/Robotics/GridPathPlanningWithObstacles/scripts/init.py \
  --metrics-out /tmp/GridPathPlanningWithObstacles_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_cost`
- `valid`
- `public_avg_cost`
- `hidden_avg_cost`
- `baseline_hidden_avg_cost`

<!-- AI_GENERATED -->
