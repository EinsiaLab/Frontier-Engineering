# Narrow Passage Path Planning

Plan a collision-free path through a single-cell narrow passage on a frozen occupancy grid.

## Provenance

- Provenance class: `fixed synthetic grid with official algorithm lineage`
- Map asset: frozen synthetic occupancy grid
- Algorithm lineage: `motion-planners`
- Full provenance note: see `references/source_manifest.md`

## Quick Run

From repository root:

```bash
python benchmarks/Robotics/NarrowPassagePathPlanning/verification/evaluator.py   benchmarks/Robotics/NarrowPassagePathPlanning/scripts/init.py   --metrics-out /tmp/NarrowPassagePathPlanning_metrics.json
```

Run with `frontier_eval`:

```bash
python -m frontier_eval   task=unified   task.benchmark=Robotics/NarrowPassagePathPlanning   algorithm.iterations=0
```
