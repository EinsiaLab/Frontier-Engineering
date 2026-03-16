# DuckDB Index Selection Task

## Objective

Choose a small set of DuckDB indexes for a frozen analytical lookup workload.

## Submission Contract

Submit one Python file that defines:

```python
def select_indexes(workload_manifest):
    ...
```

Return a list of candidate index names from the whitelist in `workload_manifest["candidate_indexes"]`.
A dict with key `indexes` is also accepted.

## Evaluation

The evaluator will:

1. Build the frozen DuckDB workload.
2. Create the selected indexes.
3. Run the fixed lookup workload four times.
4. Record the candidate total runtime and log the no-index baseline for context.

## Metrics

- `combined_score`: `-candidate_total_runtime_s`
- `valid`: `1.0` only if every selected index name is valid and execution succeeds
- `candidate_total_runtime_s`
- `baseline_total_runtime_s`
- `candidate_setup_runtime_s`
- `candidate_workload_runtime_s`
