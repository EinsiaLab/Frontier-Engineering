# SustainableDataCenterControl

This benchmark family packages SustainDC-style data-center control tasks and exposes them as standalone verifiers plus `frontier_eval` unified benchmarks.

All commands below are repository-relative and avoid absolute paths so the setup is reproducible on another machine.

## Available Subtasks

| Subtask Folder | Core Objective | Notes |
|---|---|---|
| `hand_written_control` | write a deterministic control policy for load shifting, cooling, and battery dispatch | baseline edits only `baseline/solution.py`; evaluator compares against a noop reference |

## Environment

Verified setup:

- `.venvs/frontier-v1-sustaindc` for direct verification and benchmark runtime
- `.venvs/frontier-eval-driver` for `python -m frontier_eval`

Example setup from repository root:

```bash
bash init.sh
RUN_VALIDATION=0 bash scripts/env/setup_v1_task_envs.sh
source .venvs/frontier-eval-driver/bin/activate
```

`benchmarks/SustainableDataCenterControl/requirements.txt` currently delegates to the vendored upstream requirement set under `hand_written_control/sustaindc/requirements.txt`.

## Subtask Index and Unified Run Commands

- `hand_written_control/`
  - direct verification:
    ```bash
    .venvs/frontier-v1-sustaindc/bin/python benchmarks/SustainableDataCenterControl/hand_written_control/verification/evaluate.py
    ```
  - unified run:
    ```bash
    python -m frontier_eval \
      task=unified \
      task.benchmark=SustainableDataCenterControl/hand_written_control \
      task.runtime.env_name=frontier-v1-sustaindc \
      algorithm=openevolve \
      algorithm.iterations=0
    ```
  - measured runtime on the verified setup:
    - direct verification: about `19.8s`
    - unified `algorithm.iterations=0`: about `25.8s`
  - runtime note: this task is not long-running on the validated setup, and both commands completed comfortably under the default unified timeout of `300s`.

## Notes

- The vendored `hand_written_control/sustaindc/` tree corresponds to upstream `dc-rl` commit `a92b475`.
- `hand_written_control/patches/sustaindc_optional_runtime.patch` records the small runtime-only delta we use for benchmark compatibility.
- For the full task contract and a fresh-clone reproduction path, see `hand_written_control/README.md`.
