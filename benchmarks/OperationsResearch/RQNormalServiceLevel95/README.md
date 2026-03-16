# Normal (r,Q) with 95% Service-Level Constraint

Select reorder point and lot size for Normal-demand (r,Q) instances with a hard cycle-service-level target.

## Provenance

- Upstream lineage: `Stockpyl` single-echelon `(r,Q)` routines for Normal demand.
- Data asset: benchmark-local frozen parameter tables defined in `runtime/problem.py`.
- Full provenance note: see `references/source_manifest.md`.

## File Layout

- `Task.md`: task contract and scoring rules.
- `Task_zh-CN.md`: Chinese translation of the contract.
- `scripts/init.py`: initial candidate file exposed to agents.
- `baseline/solution.py`: reference implementation.
- `runtime/problem.py`: frozen cases, baseline solver, and scoring helpers.
- `verification/evaluator.py`: evaluator entry.
- `verification/requirements.txt`: minimal dependencies for this benchmark.

## Quick Run

From repository root:

```bash
python benchmarks/OperationsResearch/RQNormalServiceLevel95/verification/evaluator.py   benchmarks/OperationsResearch/RQNormalServiceLevel95/scripts/init.py   --metrics-out /tmp/RQNormalServiceLevel95_metrics.json
```

Run through `frontier_eval` with:

```bash
python -m frontier_eval   task=unified   task.benchmark=OperationsResearch/RQNormalServiceLevel95   algorithm.iterations=0
```
