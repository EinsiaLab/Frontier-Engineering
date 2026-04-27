# DuckDB Index Selection

Choose a small whitelist subset of DuckDB indexes for a workload family and minimize hidden-case average runtime.

## What Changed

- The task now evaluates `PUBLIC_CASES` and `HIDDEN_CASES` instead of one frozen workload.
- The baseline is a simple heuristic index selector, not an empty placeholder.
- The evaluator scores hidden-case average runtime rather than one manifest.

## What You Edit

- Target file: `scripts/init.py`
- Entry point: `select_indexes(workload_manifest)`

## Source of Truth

- `Task.md`: full task contract
- `Task_zh-CN.md`: Chinese task contract
- `runtime/problem.py`: case family and runtime helper
- `baseline/solution.py`: heuristic baseline
- `verification/evaluator.py`: local evaluator

## Environment

From repository root:

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/ComputerSystems/DuckDBIndexSelection/verification/requirements.txt
```

## Quick Run

```bash
python benchmarks/ComputerSystems/DuckDBIndexSelection/verification/evaluator.py \
  benchmarks/ComputerSystems/DuckDBIndexSelection/scripts/init.py \
  --metrics-out /tmp/DuckDBIndexSelection_metrics.json
```

## Main Metrics

- `combined_score = -hidden_avg_runtime_s`
- `valid`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`

<!-- AI_GENERATED -->
