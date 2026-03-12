# Adaptive A3: Energy-Aware Control

This task introduces an engineering trade-off between correction quality and command energy.

## Why this task matters

In practical AO hardware, command magnitude is linked to:
- actuator power/heat,
- long-run reliability,
- drive electronics margin.

Residual-only control tends to produce dense/high-energy commands.
This task asks for better overall efficiency.

## Folder Structure

```text
task3_energy_aware_control/
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
- Task-specific oracle dependency: `scikit-learn` (used by `verification/reference_controller.py`, `sklearn.linear_model.Lasso`)
- Recommended one-shot install from repo root: `python -m pip install -r benchmarks/Optics/requirements.txt`

## Run

```bash
cd /DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/Optics/adaptive_energy_aware_control
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

## Outputs

- `verification/outputs/metrics.json`
- `verification/outputs/metrics_comparison.png`
- `verification/outputs/example_visualization.png`

## Baseline vs Oracle Policy

- Baseline target (`baseline/controller.py`) should stay lightweight and avoid heavy third-party solvers.
- Reference oracle uses third-party `scikit-learn` Lasso as a standardized sparse-control solver.
- Current profile is `v3_delay_and_model_mismatch` (delayed sensing + actuator lag + model mismatch).
- This is intentional to produce a stronger, harder-to-copy comparison target.
