# 多机器人优先级路径规划

在一组多机器人栅格 case 上规划无碰撞路径集合，并尽量降低 hidden case 的平均总路径代价。

## 本轮同步后的变化

- 评测已改成多组 public / hidden 多机器人栅格。
- baseline 现在是显式 prioritized planning，不再是 runtime 导出的固定方案。
- 分数改为 hidden case 平均总路径代价，同时单独报告 makespan。

## 你会改的文件

- 目标文件：`scripts/init.py`
- 入口函数：`plan_paths(grid, starts, goals)`

## 先看哪里

- `Task.md` / `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## 环境准备

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/Robotics/MultiRobotPrioritizedPlanning/verification/requirements.txt
```

## 快速运行

```bash
python benchmarks/Robotics/MultiRobotPrioritizedPlanning/verification/evaluator.py \
  benchmarks/Robotics/MultiRobotPrioritizedPlanning/scripts/init.py \
  --metrics-out /tmp/MultiRobotPrioritizedPlanning_metrics.json
```

## 主要指标

- `combined_score = -hidden_avg_total_cost`
- `valid`
- `public_avg_total_cost`
- `hidden_avg_total_cost`
- `baseline_hidden_avg_total_cost`
- `hidden_avg_makespan`

<!-- AI_GENERATED -->
