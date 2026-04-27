# 最小燃料天气航线规划

在一组天气航线 case 上规划船舶路径，并在满足最晚到达约束的前提下尽量降低 hidden case 的平均燃料消耗。

## 本轮同步后的变化

- 任务已改成多组 public / hidden 航线 case。
- baseline 现在是显式图搜索，同时考虑燃料和航时。
- 分数改为 hidden case 平均燃料，且每个 case 都有到达时限。

## 你会改的文件

- 目标文件：`scripts/init.py`
- 入口函数：`solve(instance)`

## 先看哪里

- `Task.md` / `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## 环境准备

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/OperationsResearch/FuelMinimizingShipWeatherRouting/verification/requirements.txt
```

## 快速运行

```bash
python benchmarks/OperationsResearch/FuelMinimizingShipWeatherRouting/verification/evaluator.py \
  benchmarks/OperationsResearch/FuelMinimizingShipWeatherRouting/scripts/init.py \
  --metrics-out /tmp/FuelMinimizingShipWeatherRouting_metrics.json
```

## 主要指标

- `combined_score = -hidden_avg_fuel`
- `valid`
- `public_avg_fuel`
- `hidden_avg_fuel`
- `baseline_hidden_avg_fuel`

<!-- AI_GENERATED -->
