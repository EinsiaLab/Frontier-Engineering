# Robotics

This domain contains robotics control and planning tasks for unified evaluation.

## Tasks

- `CoFlyersVasarhelyiTuning`
  - Unified benchmark: `task=coflyers_vasarhelyi_tuning`
  - Quick run: `python -m frontier_eval task=coflyers_vasarhelyi_tuning algorithm.iterations=0`
- `DynamicObstacleAvoidanceNavigation`
- `PIDTuning`
- `QuadrupedGaitOptimization`
- `RobotArmCycleTimeOptimization`
- `UAVInspectionCoverageWithWind`

### Unified quick runs

- `DynamicObstacleAvoidanceNavigation`: `python -m frontier_eval task=unified task.benchmark=Robotics/DynamicObstacleAvoidanceNavigation algorithm.iterations=0`
- `PIDTuning`: `python -m frontier_eval task=unified task.benchmark=Robotics/PIDTuning algorithm.iterations=0`
- `QuadrupedGaitOptimization`: `python -m frontier_eval task=unified task.benchmark=Robotics/QuadrupedGaitOptimization task.runtime.conda_env=<your_env> algorithm.iterations=0`
- `RobotArmCycleTimeOptimization`: `python -m frontier_eval task=unified task.benchmark=Robotics/RobotArmCycleTimeOptimization task.runtime.conda_env=<your_env> algorithm.iterations=0`
- `UAVInspectionCoverageWithWind`: `python -m frontier_eval task=unified task.benchmark=Robotics/UAVInspectionCoverageWithWind algorithm.iterations=0`
