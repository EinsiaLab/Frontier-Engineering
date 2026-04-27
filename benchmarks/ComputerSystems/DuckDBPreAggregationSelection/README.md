# DuckDB Pre-Aggregation Selection

Choose a small whitelist subset of legal pre-aggregations for a workload family and minimize hidden-case average runtime.

## What Changed

- The task now evaluates multiple public and hidden report configurations.
- The baseline is a heuristic materialization choice, not a null selector.
- Candidate designs must preserve report semantics across the whole case family.

## What You Edit

- Target file: `scripts/init.py`
- Entry point: `select_preaggregations(workload_manifest)`

## Source of Truth

- `Task.md`
- `Task_zh-CN.md`
- `runtime/problem.py`
- `baseline/solution.py`
- `verification/evaluator.py`

## Environment

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/ComputerSystems/DuckDBPreAggregationSelection/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/ComputerSystems/DuckDBPreAggregationSelection/verification/evaluator.py \
  benchmarks/ComputerSystems/DuckDBPreAggregationSelection/scripts/init.py \
  --metrics-out /tmp/DuckDBPreAggregationSelection_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_runtime_s`
- `valid`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`

<!-- AI_GENERATED -->
