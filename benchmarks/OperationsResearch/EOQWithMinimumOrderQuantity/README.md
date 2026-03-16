# EOQ with Minimum Order Quantity

Optimize annual cost for deterministic EOQ instances with a hard minimum order quantity.

## Provenance

- Upstream lineage: `Stockpyl` EOQ routines and the classic deterministic EOQ model family.
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
python benchmarks/OperationsResearch/EOQWithMinimumOrderQuantity/verification/evaluator.py   benchmarks/OperationsResearch/EOQWithMinimumOrderQuantity/scripts/init.py   --metrics-out /tmp/EOQWithMinimumOrderQuantity_metrics.json
```

Run through `frontier_eval` with:

```bash
python -m frontier_eval   task=unified   task.benchmark=OperationsResearch/EOQWithMinimumOrderQuantity   algorithm.iterations=0
```
