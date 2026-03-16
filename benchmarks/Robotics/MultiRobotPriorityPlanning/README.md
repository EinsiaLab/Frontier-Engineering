# Multi-Robot Priority Planning

Plan collision-free paths for three robots on a frozen occupancy grid while improving over a fixed priority order baseline.

## Provenance

- Provenance class: `fixed synthetic grid with official algorithm lineage`
- Map asset: frozen synthetic occupancy grid
- Algorithm lineage: `motion-planners`
- Full provenance note: see `references/source_manifest.md`

## Quick Run

From repository root:

```bash
python benchmarks/Robotics/MultiRobotPriorityPlanning/verification/evaluator.py \
  benchmarks/Robotics/MultiRobotPriorityPlanning/scripts/init.py \
  --metrics-out /tmp/MultiRobotPriorityPlanning_metrics.json
```

Run with `frontier_eval`:

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=Robotics/MultiRobotPriorityPlanning \
  algorithm.iterations=0
```
