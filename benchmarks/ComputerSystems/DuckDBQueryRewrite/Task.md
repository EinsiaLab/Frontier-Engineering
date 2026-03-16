# DuckDB Query Rewrite Task

## Objective

Rewrite a frozen DuckDB analytical SQL query to preserve results while reducing total runtime.

## Submission Contract

Submit one Python file that defines:

```python
def rewrite_query(sql, workload_manifest):
    ...
```

Return a rewritten SQL string. A dict with key `sql` is also accepted.

## Evaluation

The evaluator will:

1. Build the frozen DuckDB workload.
2. Execute the original SQL to get the reference result.
3. Execute your rewritten SQL and verify exact result equivalence.
4. Time the candidate query over repeated runs and log the baseline rewrite runtime for context.

## Metrics

- `combined_score`: `-candidate_runtime_s`
- `valid`: `1.0` only if the rewritten query preserves results
- `candidate_runtime_s`
- `baseline_runtime_s`
- `row_count`
