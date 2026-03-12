# Adaptive A2: Temporal Smooth Control

This task targets **time-series AO control** with command smoothness requirements.

## Why this task matters

In real systems, command jitter can be harmful:
- actuator wear,
- mirror vibration,
- unstable loop behavior.

Pure frame-wise control may minimize instantaneous residual, but produce noisy command trajectories.

This task optimizes a practical objective that balances correction quality and command smoothness.

## Folder Structure

```text
task2_temporal_smooth_control/
  baseline/
    controller.py
  verification/
    evaluate.py
    reference_controller.py
    outputs/
  README.md
  README_zh-CN.md
  Task.md
  Task_zh-CN.md
```

## Environment Dependencies

- Python: `3.10+` (tested with `/data_storage/chihh2311/.conda/envs/aotools/bin/python`)
- Baseline candidate runtime: `numpy`
- Verification runtime: `numpy`, `matplotlib`, local `aotools` package (this repository)
- Task-specific oracle dependency: none (reference is analytical and does not require extra third-party solver)
- Recommended one-shot install from repo root: `python -m pip install -r benchmarks/Optics/requirements.txt`

## How to Run

```bash
cd /DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/Optics/adaptive_temporal_smooth_control
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

## Outputs

- `verification/outputs/metrics.json`
- `verification/outputs/metrics_comparison.png`
- `verification/outputs/example_visualization.png`

## Baseline vs Oracle Policy

- Baseline target is `baseline/controller.py` and should avoid heavy third-party solvers.
- Reference oracle is a delay-compensated analytical smooth controller (still no external optimizer).
- Current profile is `v3_delay_and_model_mismatch`, including delayed sensing and actuator constraints.
