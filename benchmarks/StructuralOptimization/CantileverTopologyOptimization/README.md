# Cantilever Compliance Topology Optimization

Minimize compliance on a frozen cantilever beam using pyMOTO's SIMP formulation and a fixed material budget.

## Provenance

- Provenance class: `official-example-derived`
- Frozen geometry: `cantilever`
- Solver lineage: `pyMOTO` compliance + SIMP density optimization
- Full provenance note: see `references/source_manifest.md`

## File Layout

- `Task.md`: task contract and scoring rules.
- `Task_zh-CN.md`: Chinese translation.
- `scripts/init.py`: initial candidate file exposed to agents.
- `baseline/solution.py`: OC-style baseline update rule.
- `runtime/problem.py`: frozen physics, constraints, and optimization loop.
- `verification/evaluator.py`: evaluator entry.
- `references/source_manifest.md`: source and provenance notes.

## Quick Run

From repository root:

```bash
/mnt/shared-storage-user/p1-shared/luotianwei/Frontier-Engineering/.venv/bin/python \
  benchmarks/StructuralOptimization/CantileverTopologyOptimization/verification/evaluator.py \
  benchmarks/StructuralOptimization/CantileverTopologyOptimization/scripts/init.py \
  --metrics-out /tmp/CantileverTopologyOptimization_metrics.json
```

Run with `frontier_eval`:

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=StructuralOptimization/CantileverTopologyOptimization \
  task.runtime.use_conda_run=false \
  task.runtime.python_path=/mnt/shared-storage-user/p1-shared/luotianwei/Frontier-Engineering/.venv/bin/python \
  algorithm.iterations=0
```
