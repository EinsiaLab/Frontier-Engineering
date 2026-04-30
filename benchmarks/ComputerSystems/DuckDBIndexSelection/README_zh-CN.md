# DuckDB 索引选择

在一组 DuckDB workload 上选择少量白名单索引，并尽量降低 hidden case 的平均运行时间。

## 本轮同步后的变化

- 评测已从单一 workload 改成 `PUBLIC_CASES + HIDDEN_CASES`。
- baseline 现在是启发式索引选择器，不再是空实现。
- 分数改为 hidden case 平均运行时间，而不是单个 manifest。

## 你会改的文件

- 目标文件：`scripts/init.py`
- 入口函数：`select_indexes(workload_manifest)`

## 先看哪里

- `Task.md` / `Task_zh-CN.md`：任务契约
- `runtime/problem.py`：case family 与运行辅助逻辑
- `baseline/solution.py`：启发式 baseline
- `verification/evaluator.py`：本地评测入口

## 环境准备

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/ComputerSystems/DuckDBIndexSelection/verification/requirements.txt
```

## 快速运行

```bash
python benchmarks/ComputerSystems/DuckDBIndexSelection/verification/evaluator.py \
  benchmarks/ComputerSystems/DuckDBIndexSelection/scripts/init.py \
  --metrics-out /tmp/DuckDBIndexSelection_metrics.json
```

## 主要指标

- `combined_score = -hidden_avg_runtime_s`
- `valid`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`

<!-- AI_GENERATED -->
