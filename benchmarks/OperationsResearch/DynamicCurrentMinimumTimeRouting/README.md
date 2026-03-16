# Dynamic-Current Minimum-Time Routing

Route a ship across a frozen coastal grid while minimizing travel time under deterministic current and depth fields.

## Provenance

- Provenance class: `benchmark-local synthetic environment with traceable upstream routing lineage`
- Upstream lineage: see `references/source_manifest.md`
- Data asset: fixed synthetic coastal grid and deterministic environmental fields embedded in `runtime/problem.py`
- Redistribution status: no upstream environmental rasters are vendored

## File Layout

- `Task.md`: task contract and scoring rules
- `Task_zh-CN.md`: Chinese translation
- `README_zh-CN.md`: Chinese overview
- `scripts/init.py`: initial candidate file exposed to agents
- `baseline/solution.py`: reference baseline
- `runtime/problem.py`: frozen instance generator, validation logic, and reference costs
- `verification/evaluator.py`: evaluator entry
- `references/source_manifest.md`: provenance notes

## Quick Run

From repository root:

```bash
.venv/bin/python benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/verification/evaluator.py \
  benchmarks/OperationsResearch/DynamicCurrentMinimumTimeRouting/scripts/init.py \
  --metrics-out /tmp/DynamicCurrentMinimumTimeRouting_metrics.json
```
