# 动态流场最短航时船舶路径规划

在一组航线 case 上规划船舶路径，并在满足流场与吃水约束的前提下尽量降低 hidden case 的平均航时。

## 本轮同步后的变化

- 任务已改成多组 public / hidden 地图。
- baseline 现在是显式最短航时图搜索，不再是 runtime helper 导出。
- 分数改为 hidden case 平均航时。

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
pip install -r benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/verification/requirements.txt
```

## 快速运行

```bash
python benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/verification/evaluator.py \
  benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/scripts/init.py \
  --metrics-out /tmp/DynamicCurrentMinimumTimeRouting_metrics.json
```

## 主要指标

- `combined_score = -hidden_avg_time_h`
- `valid`
- `public_avg_time_h`
- `hidden_avg_time_h`
- `baseline_hidden_avg_time_h`

<!-- AI_GENERATED -->
