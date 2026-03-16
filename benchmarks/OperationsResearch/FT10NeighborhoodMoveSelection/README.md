# FT10 Neighborhood Move Selection

Guide an adjacent-swap local search on the canonical FT10 Fisher-Thompson 10x10 job shop instance.

## Provenance

The frozen instance is copied from the canonical benchmark set distributed in `job_shop_lib/benchmarking/benchmark_instances.json`.
The instance id is `ft10`, and the published optimum used for scoring reference is `930`.

## File Layout

- `Task.md`: task contract and scoring rules.
- `Task_zh-CN.md`: Chinese version.
- `scripts/init.py`: initial candidate file exposed to agents.
- `baseline/solution.py`: baseline heuristic.
- `runtime/problem.py`: frozen instance, scheduling runtime, baseline, and evaluator helpers.
- `runtime/instance.json`: vendored canonical benchmark instance.
- `verification/evaluator.py`: evaluator entry.
- `references/source_manifest.md`: instance provenance.

## Quick Run

```bash
python benchmarks/OperationsResearch/FT10NeighborhoodMoveSelection/verification/evaluator.py   benchmarks/OperationsResearch/FT10NeighborhoodMoveSelection/scripts/init.py   --metrics-out /tmp/FT10NeighborhoodMoveSelection_metrics.json
```

Run with `frontier_eval`:

```bash
python -m frontier_eval   task=unified   task.benchmark=OperationsResearch/FT10NeighborhoodMoveSelection   task.runtime.use_conda_run=false   task.runtime.python_path=/mnt/shared-storage-user/p1-shared/luotianwei/Frontier-Engineering/.venv/bin/python   algorithm.iterations=0
```
