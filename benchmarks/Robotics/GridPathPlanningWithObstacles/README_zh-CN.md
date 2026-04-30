# 带障碍栅格路径规划

在一组栅格 case 上规划无碰撞路径，并尽量降低 hidden case 的平均路径代价。

## 本轮同步后的变化

- 评测已改成多组 public / hidden 占据栅格。
- baseline 现在是显式 A*，不再是单条冻结路径。
- 分数改为 hidden case 平均路径代价。

## 你会改的文件

- 目标文件：`scripts/init.py`
- 入口函数：`plan_path(grid, start, goal)`

## 先看哪里

- `Task.md` / `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## 环境准备

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/Robotics/GridPathPlanningWithObstacles/verification/requirements.txt
```

## 快速运行

```bash
python benchmarks/Robotics/GridPathPlanningWithObstacles/verification/evaluator.py \
  benchmarks/Robotics/GridPathPlanningWithObstacles/scripts/init.py \
  --metrics-out /tmp/GridPathPlanningWithObstacles_metrics.json
```

## 主要指标

- `combined_score = -hidden_avg_cost`
- `valid`
- `public_avg_cost`
- `hidden_avg_cost`
- `baseline_hidden_avg_cost`

<!-- AI_GENERATED -->
