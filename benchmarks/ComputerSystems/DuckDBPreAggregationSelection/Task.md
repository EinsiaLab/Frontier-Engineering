# DuckDB Pre-Aggregation Selection Task

## Problem

Choose a small whitelist subset of legal pre-aggregations for an analytical workload family and minimize hidden-case average runtime.

The evaluator now uses multiple public and hidden report configurations rather than one frozen workload. Each case changes segment filters, time windows, top-k settings, or report emphasis. The goal is to pick pre-aggregations that generalize across these report shapes without changing query semantics.

## What Is Frozen

- The local DuckDB schema and data generator in `benchmarks/ComputerSystems/duckdb_local_workload.py`.
- The whitelist of legal pre-aggregation names and the per-case pre-aggregation budget.
- The semantics check: every candidate design must preserve the results of the frozen report family.

## Submission Contract

Submit one Python file that defines:

```python
def select_preaggregations(workload_manifest):
    ...
```

Return a list of whitelist pre-aggregation names. A dict with key `preaggregations` is also accepted.

## Evaluation

1. Load case manifests from `PUBLIC_CASES` and `HIDDEN_CASES`.
2. For each case, call `select_preaggregations(...)` with the case manifest.
3. Materialize the selected pre-aggregations and verify that report outputs remain unchanged.
4. Aggregate runtime across cases; scoring uses the hidden-case average.

## Metrics

- `combined_score`: `-hidden_avg_runtime_s`
- `valid`: `1.0` only if all reports stay semantically correct and all cases run successfully
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`
- `num_public_cases`
- `num_hidden_cases`

## Invalid Submissions

- `select_preaggregations(...)` is missing or crashes
- The return value cannot be parsed into a list of names
- Any selected name is outside the whitelist
- Any case exceeds its pre-aggregation budget
- Candidate pre-aggregations change any report result
- Setup or evaluation fails on any public or hidden case

<!-- AI_GENERATED -->
