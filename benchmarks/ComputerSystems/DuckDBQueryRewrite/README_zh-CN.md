# DuckDB 查询重写

对一组分析型 SQL 做语义等价改写，并尽量降低 hidden case 的平均运行时间。

## 本轮同步后的变化

- 评测已改成多组 public / hidden SQL case。
- baseline 改写现在会按 case 选择具体 SQL，不再原样返回输入。
- 每个 case 都会先做语义等价检查，只有等价后才比较运行时间。

## 你会改的文件

- 目标文件：`scripts/init.py`
- 入口函数：`rewrite_query(sql, workload_manifest)`

## 先看哪里

- `Task.md` / `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## 环境准备

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/ComputerSystems/DuckDBQueryRewrite/verification/requirements.txt
```

## 快速运行

```bash
python benchmarks/ComputerSystems/DuckDBQueryRewrite/verification/evaluator.py \
  benchmarks/ComputerSystems/DuckDBQueryRewrite/scripts/init.py \
  --metrics-out /tmp/DuckDBQueryRewrite_metrics.json
```

## 主要指标

- `combined_score = -hidden_avg_runtime_s`
- `valid`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`

<!-- AI_GENERATED -->
