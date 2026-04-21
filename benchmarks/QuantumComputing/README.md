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

Current baseline strategies:
- `task_01`: local rewrite preprocessing followed by target-aware multi-seed transpile search.
- `task_02`: `local rewrite -> clifford+t transpile(opt=3) -> local rewrite`.
- `task_03`: target-aware transpilation with backend-specific equivalence registration and transpile settings.

## Unified Per-Task Structure
Each task now uses the same structure:
- `baseline/solve.py`: evolve entrypoint with the task-specific baseline strategy.
- `baseline/structural_optimizer.py`: task-local local-rewrite helper reused by `solve.py`.
- `verification/evaluate.py`: single evaluation entrypoint that includes candidate and `opt0..opt3` references.
- `verification/utils.py`: helper functions.
- `tests/case_*.json`: multiple differentiated test cases.
- `README*.md` and `TASK*.md`: run guide and task definition.

## Eval
```bash
.venvs/frontier-eval-driver/bin/python -m frontier_eval task=unified task.benchmark=QuantumComputing/task_01_routing_qftentangled task.runtime.env_name=frontier-v1-main algorithm=openevolve algorithm.iterations=0
.venvs/frontier-eval-driver/bin/python -m frontier_eval task=unified task.benchmark=QuantumComputing/task_02_clifford_t_synthesis task.runtime.env_name=frontier-v1-main algorithm=openevolve algorithm.iterations=0
.venvs/frontier-eval-driver/bin/python -m frontier_eval task=unified task.benchmark=QuantumComputing/task_03_cross_target_qaoa task.runtime.env_name=frontier-v1-main algorithm=openevolve algorithm.iterations=0
```
