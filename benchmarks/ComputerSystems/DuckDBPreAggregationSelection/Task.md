# DuckDB Pre-Aggregation Selection Task

## Objective

Choose a small set of pre-aggregation tables for a frozen DuckDB reporting workload.

## Submission Contract

Submit one Python file that defines:

```python
def select_preaggregations(workload_manifest):
    ...
```

Return a list of candidate pre-aggregation names from the whitelist in `workload_manifest["candidate_preaggregations"]`.
A dict with key `preaggregations` is also accepted.

## Evaluation

The evaluator will:

1. Build the frozen DuckDB workload.
2. Create the selected pre-aggregation tables.
3. Run the fixed reporting workload and verify result equivalence.
4. Measure candidate total runtime as setup cost plus repeated report execution, and log the baseline for context.

## Metrics

- `combined_score`: `-candidate_total_runtime_s`
- `valid`: `1.0` only if all selected names are valid and results stay unchanged
- `candidate_total_runtime_s`
- `baseline_total_runtime_s`
- `candidate_setup_runtime_s`
- `candidate_workload_runtime_s`
