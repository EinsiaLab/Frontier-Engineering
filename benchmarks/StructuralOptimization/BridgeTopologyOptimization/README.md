# Bridge Topology Optimization

Minimize compliance on a frozen bridge-like topology optimization case with a passive-solid deck and distributed load.

## Provenance

- Provenance class: `traceable literature-derived local instance`
- Frozen geometry: `bridge_half`
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
  benchmarks/StructuralOptimization/BridgeTopologyOptimization/verification/evaluator.py \
  benchmarks/StructuralOptimization/BridgeTopologyOptimization/scripts/init.py \
  --metrics-out /tmp/BridgeTopologyOptimization_metrics.json
```

Run with `frontier_eval`:

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=StructuralOptimization/BridgeTopologyOptimization \
  task.runtime.use_conda_run=false \
  task.runtime.python_path=/mnt/shared-storage-user/p1-shared/luotianwei/Frontier-Engineering/.venv/bin/python \
  algorithm.iterations=0
```
