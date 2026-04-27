# DuckDB Index Selection Task

## Problem

Choose a small whitelist subset of DuckDB indexes for an analytical workload family and minimize hidden-case average runtime.

This benchmark is no longer a single frozen lookup workload. The evaluator now uses multiple public and hidden workload manifests that vary query mix, recency filters, and lookup intensity. Good submissions should choose indexes that generalize across these manifests rather than overfitting one case.

## What Is Frozen

- The local DuckDB schema and data generator in `benchmarks/ComputerSystems/duckdb_local_workload.py`.
- The legal whitelist of index names and per-case index budget in each workload manifest.
- The timing protocol: create the selected indexes, warm up once, then time repeated workload execution for each case.

## Submission Contract

Submit one Python file that defines:

```python
def select_indexes(workload_manifest):
    ...
```

Return a list of whitelist index names. A dict with key `indexes` is also accepted.

## Evaluation

1. Load `PUBLIC_CASES` and `HIDDEN_CASES` from `runtime/problem.py`.
2. For each case, pass the case-specific manifest into `select_indexes(...)`.
3. Create the selected indexes, run the case workload, and measure total runtime.
4. Aggregate public and hidden runtimes separately; scoring uses the hidden average.

## Metrics

- `combined_score`: `-hidden_avg_runtime_s`
- `valid`: `1.0` only if all cases execute successfully and every selected index is legal
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`
- `num_public_cases`
- `num_hidden_cases`

## Invalid Submissions

- `select_indexes(...)` is missing or crashes
- The return value cannot be parsed into a list of names
- Any selected name is outside the whitelist
- Any case exceeds its index budget
- Index creation or workload execution fails on any public or hidden case

<!-- AI_GENERATED -->
