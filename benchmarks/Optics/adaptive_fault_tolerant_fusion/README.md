# Adaptive A4: Fault-Tolerant Sensor Fusion

This task focuses on robust control under multi-sensor corruption.

## Why this task matters

In multi-WFS systems, one channel can be mis-calibrated or noisy.
Naively averaging all sensors can propagate outliers directly into DM commands.

This task asks the agent to fuse multi-sensor slopes robustly before control.

## Folder Structure

```text
task4_fault_tolerant_fusion/
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
- Task-specific oracle dependency: `scikit-learn` (used by `verification/reference_controller.py`, `sklearn.ensemble.IsolationForest`)
- Recommended one-shot install from repo root: `python -m pip install -r benchmarks/Optics/requirements.txt`

## Run

```bash
cd /DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/Optics/adaptive_fault_tolerant_fusion
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

## Outputs

- `verification/outputs/metrics.json`
- `verification/outputs/metrics_comparison.png`
- `verification/outputs/example_visualization.png`

## Baseline vs Oracle Policy

- Baseline target (`baseline/controller.py`) is intentionally simple and should avoid heavy third-party dependencies.
- Reference oracle uses third-party `scikit-learn` `IsolationForest` + weighted inlier fusion.
- Current profile is `v3_fault_stress` (fault-dominant multi-sensor corruption benchmark).
- Verification scenario is a severe fault stress test (multiple corrupted sensors per case) to keep the baseline-oracle gap meaningful.
