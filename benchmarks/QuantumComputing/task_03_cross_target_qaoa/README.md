# Task 03 README

This file focuses on run commands and directory layout.
For task definition (goal, I/O, scoring), see [TASK.md](TASK.md).

## Environment
Use the requested interpreter:

```bash
/data_storage/chihh2311/.conda/envs/mqt/bin/python
```

## Run
From this task directory:

```bash
cd /DATA_EDS2/haohan.chi.2311/bench/agent_evolve_quantum_tasks/task_03_cross_target_qaoa
/data_storage/chihh2311/.conda/envs/mqt/bin/python verification/evaluate.py
```

Optional arguments:
- `--artifact-dir <path>`: custom output directory for generated QASM/PNG artifacts.
- `--json-out <path>`: save the evaluation report as JSON.

## File Structure
- `baseline/solve.py`: the only file intended for agent evolve.
- `baseline/structural_optimizer.py`: current rule-based baseline optimizer implementation.
- `verification/evaluate.py`: single evaluation entrypoint; includes candidate and `opt0..opt3` reference comparison for each target.
- `verification/utils.py`: task-local helper functions.
- `tests/case_*.json`: differentiated test cases.
- `TASK.md`: task details in English.
- `TASK_zh-CN.md`: task details in Chinese.
- `runs/`: generated artifacts for each evaluation run.
