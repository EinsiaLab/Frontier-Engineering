# DuckDB Pre-Aggregation Selection

Choose a small set of pre-aggregation tables for a frozen DuckDB reporting workload.

## Provenance

- Provenance class: `traceable local workload with DuckDB/TPC-H schema lineage`
- Engine lineage: `DuckDB`
- Data asset: benchmark-local deterministic SQL-generated tables
- Full provenance note: see `references/source_manifest.md`

## File Layout

- `Task.md`: task contract and scoring rules.
- `Task_zh-CN.md`: Chinese translation.
- `README_zh-CN.md`: Chinese overview.
- `scripts/init.py`: initial candidate file exposed to agents.
- `baseline/solution.py`: reference implementation.
- `runtime/problem.py`: task-local interface to the frozen workload.
- `verification/evaluator.py`: evaluator entry.
- `references/source_manifest.md`: provenance and authenticity notes.

## Quick Run

From repository root:

```bash
.venv/bin/python benchmarks/ComputerSystems/DuckDBPreAggregationSelection/verification/evaluator.py \
  benchmarks/ComputerSystems/DuckDBPreAggregationSelection/scripts/init.py \
  --metrics-out /tmp/DuckDBPreAggregationSelection_metrics.json
```
