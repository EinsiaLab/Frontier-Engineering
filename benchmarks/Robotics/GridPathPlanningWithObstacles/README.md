# Grid Path Planning with Obstacles

Plan a collision-free path on a frozen 2D occupancy grid with static obstacles.

## Provenance

- Provenance class: `fixed synthetic grid with official algorithm lineage`
- Map asset: frozen synthetic occupancy grid
- Algorithm lineage: `motion-planners`
- Full provenance note: see `references/source_manifest.md`

## Quick Run

From repository root:

```bash
python benchmarks/Robotics/GridPathPlanningWithObstacles/verification/evaluator.py   benchmarks/Robotics/GridPathPlanningWithObstacles/scripts/init.py   --metrics-out /tmp/GridPathPlanningWithObstacles_metrics.json
```

Run with `frontier_eval`:

```bash
python -m frontier_eval   task=unified   task.benchmark=Robotics/GridPathPlanningWithObstacles   algorithm.iterations=0
```
