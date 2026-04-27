# DuckDB 预聚合选择

在一组 DuckDB 报表 workload 上选择少量合法预聚合，并尽量降低 hidden case 的平均运行时间。

## 本轮同步后的变化

- 评测已改成多组 public / hidden 报表配置。
- baseline 现在是启发式物化选择，不再是空选择器。
- 候选方案必须在整个 case family 上保持报表语义不变。

## 你会改的文件

- 目标文件：`scripts/init.py`
- 入口函数：`select_preaggregations(workload_manifest)`

## 先看哪里

- `Task.md` / `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## 环境准备

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/ComputerSystems/DuckDBPreAggregationSelection/verification/requirements.txt
```

## 快速运行

```bash
python benchmarks/ComputerSystems/DuckDBPreAggregationSelection/verification/evaluator.py \
  benchmarks/ComputerSystems/DuckDBPreAggregationSelection/scripts/init.py \
  --metrics-out /tmp/DuckDBPreAggregationSelection_metrics.json
```

## 主要指标

- `combined_score = -hidden_avg_runtime_s`
- `valid`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`

<!-- AI_GENERATED -->
