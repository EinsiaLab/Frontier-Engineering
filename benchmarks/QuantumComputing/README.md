# Agent-Evolve Quantum Optimization Tasks

This folder contains three benchmark-driven optimization tasks based on this repository's `mqt.bench` APIs.

## Environment
Use the requested interpreter:

```bash
pip install mqt.bench
```

## Task List
- `task_01_routing_qftentangled`: mapped-level routing optimization on IBM Falcon.
- `task_02_clifford_t_synthesis`: native-gates (`clifford+t`) synthesis optimization.
- `task_03_cross_target_qaoa`: one strategy evaluated on both IBM and IonQ targets.

## Unified Per-Task Structure
Each task now uses the same structure:
- `baseline/solve.py`: evolve entrypoint.
- `baseline/structural_optimizer.py`: current weak baseline logic.
- `verification/evaluate.py`: single evaluation entrypoint that includes candidate and `opt0..opt3` references.
- `verification/utils.py`: helper functions.
- `tests/case_*.json`: multiple differentiated test cases.
- `README*.md` and `TASK*.md`: run guide and task definition.

## Eval
```
python -m frontier_eval task=unified task.benchmark=QuantumComputing task_01_routing_qftentangled/task_02_clifford_t_synthesis/task_03_cross_target_qaoa task.runtime.conda_env=mqt algorithm=openevolve algorithm.iterations=0
```
