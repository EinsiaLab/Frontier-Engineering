# DuckDB Query Rewrite

Rewrite analytical SQL for a query family while preserving exact results and minimizing hidden-case average runtime.

## What Changed

- The evaluator now runs multiple public and hidden SQL cases.
- Baseline rewrites are case-aware and no longer just echo the input SQL.
- Semantic equivalence is checked on every case before runtime matters.

## What You Edit

- Target file: `scripts/init.py`
- Entry point: `rewrite_query(sql, workload_manifest)`

## Source of Truth

- `Task.md`
- `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## Environment

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/ComputerSystems/DuckDBQueryRewrite/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/ComputerSystems/DuckDBQueryRewrite/verification/evaluator.py \
  benchmarks/ComputerSystems/DuckDBQueryRewrite/scripts/init.py \
  --metrics-out /tmp/DuckDBQueryRewrite_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_runtime_s`
- `valid`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`

<!-- AI_GENERATED -->
