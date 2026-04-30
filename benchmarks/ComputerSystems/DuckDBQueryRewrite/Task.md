# DuckDB Query Rewrite Task

## Problem

Rewrite analytical SQL queries for a workload family while preserving exact results and minimizing hidden-case average runtime.

This task is no longer a single frozen SQL statement. The evaluator now uses multiple public and hidden SQL cases with different grouping keys, filters, and rollups. A good rewrite strategy should preserve semantics exactly and improve runtime across the query family.

## What Is Frozen

- The local DuckDB schema and data generator in `benchmarks/ComputerSystems/duckdb_local_workload.py`.
- The case-specific baseline SQL stored in `PUBLIC_CASES` and `HIDDEN_CASES`.
- The semantic check: candidate rows must match the frozen baseline query exactly, up to floating-point tolerance.

## Submission Contract

Submit one Python file that defines:

```python
def rewrite_query(sql, workload_manifest):
    ...
```

Return a rewritten SQL string. A dict with key `sql` is also accepted by the runtime helper.

## Evaluation

1. For each public and hidden case, pass the baseline SQL and case manifest into `rewrite_query(...)`.
2. Execute both the baseline SQL and the candidate SQL on fresh DuckDB databases.
3. Reject the candidate if any query result differs from the baseline result.
4. Measure runtime across the case family; scoring uses the hidden-case average.

## Metrics

- `combined_score`: `-hidden_avg_runtime_s`
- `valid`: `1.0` only if every rewritten query is semantically equivalent and all cases run successfully
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`
- `num_public_cases`
- `num_hidden_cases`

## Invalid Submissions

- `rewrite_query(...)` is missing or crashes
- The returned value cannot be interpreted as SQL
- Any public or hidden case changes the query result
- Any rewritten query fails to execute
- Any reported runtime becomes non-finite

<!-- AI_GENERATED -->
