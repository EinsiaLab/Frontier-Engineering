# Adaptive A1: Constrained DM Control

This task focuses on **single-step constrained control** for deformable mirror (DM) commands.

## Why this task matters

In AO systems, the naive control law `u = R @ s` (reconstructor times slopes) often violates actuator limits.
Hard clipping (`clip`) is common in practice, but clipping after an unconstrained solve is not generally optimal.

This task asks the agent to improve control quality under strict voltage constraints.

## Folder Structure

```text
task1_constrained_dm_control/
  baseline/
    controller.py                  # agent edits this file
  verification/
    evaluate.py                    # validity + baseline/reference comparison
    reference_controller.py        # stronger reference implementation
    outputs/                       # generated after running evaluate.py
  README.md
  README_zh-CN.md
  Task.md
  Task_zh-CN.md
```

## Environment Dependencies

- Python: `3.10+` (tested with `/data_storage/chihh2311/.conda/envs/aotools/bin/python`)
- Baseline candidate runtime: `numpy`
- Verification runtime: `numpy`, `matplotlib`, local `aotools` package (this repository)
- Task-specific oracle dependency: `scipy` (used by `verification/reference_controller.py`, `scipy.optimize.lsq_linear`)
- Recommended one-shot install from repo root: `python -m pip install -r benchmarks/Optics/requirements.txt`

## How to Run

```bash
cd /DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/Optics/adaptive_constrained_dm_control
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

Optional candidate path:

```bash
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py \
  --candidate /abs/path/to/controller.py
```

## Outputs

- `verification/outputs/metrics.json`
- `verification/outputs/metrics_comparison.png`
- `verification/outputs/example_visualization.png`

`metrics.json` includes candidate baseline metrics and reference metrics under identical random seed/scenario settings.

## Baseline vs Oracle Policy

- Baseline target (`baseline/controller.py`) should remain lightweight (`numpy` + provided matrices).
- Reference oracle uses third-party SciPy bounded least squares (`scipy.optimize.lsq_linear`).
- Current profile is `v3_delay_and_model_mismatch` (delayed sensing + actuator lag + model mismatch).
- This separation is intentional to keep the comparison non-trivial for agent evolve benchmarking.
