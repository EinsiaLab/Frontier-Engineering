# Additive Manufacturing

This domain contains additive-manufacturing optimization tasks adapted into Frontier-Engineering from real process datasets, toolpaths, or simulation workflows.

## Domain Overview

Additive manufacturing optimization typically requires balancing:

- process quality and defect avoidance,
- thermal consistency and melt-pool stability,
- physical feasibility constraints,
- manufacturing efficiency and experimental cost.

## Task Index

- `DiffSimThermalControl/`
  - adapted from the real build geometry and toolpath published in `differentiable-simulation-am`,
  - uses upstream `0.k` and `toolpath.crs` files directly,
  - reformulates the published case into a reproducible thermal-control benchmark for `frontier_eval`.

## Quick Run

```bash
python benchmarks/AdditiveManufacturing/DiffSimThermalControl/verification/evaluator.py \
  benchmarks/AdditiveManufacturing/DiffSimThermalControl/scripts/init.py
```

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=AdditiveManufacturing/DiffSimThermalControl \
  task.runtime.conda_env=<your_env> \
  algorithm.iterations=0
```
