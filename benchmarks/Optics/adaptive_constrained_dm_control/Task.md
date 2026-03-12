# Adaptive A1 Specification: Constrained DM Control

## 1. Background for CS Readers

You can treat this task as a **constrained vector optimization problem in a noisy control loop**.

- The system state (optical wavefront distortion) is observed indirectly.
- The observation is a slope vector `s` from a wavefront sensor (WFS).
- We need to output actuator commands `u` for a deformable mirror (DM).
- Commands are physically limited: each channel must stay in `[-Vmax, Vmax]`.

If you only use a linear map `u = R @ s` and then clip, the result is valid but often not optimal under hard bounds.

## 2. What You Need to Do

Implement a better controller in **one function**:

- Editable file: `baseline/controller.py`
- Target function:

```python
def compute_dm_commands(slopes, reconstructor, control_model, prev_commands=None, max_voltage=0.15):
    ...
```

Goal:
- Improve leaderboard score `score_0_to_1_higher_is_better`.
- Keep output always valid (shape, finite values, bounds).

## 3. Input / Output Contract

### Inputs

- `slopes: np.ndarray`, shape `(2 * n_subap,)`
  - Current WFS slope measurement.
- `reconstructor: np.ndarray`, shape `(n_act, 2 * n_subap)`
  - Baseline linear map from slopes to commands.
- `control_model: dict`
  - Precomputed matrices and constants prepared by `verification/evaluate.py`.
  - Available keys include:
    - `normal_matrix`: `H^T H + lambda I`
    - `h_t`: `H^T`
    - `h_matrix`: `H`
    - `pgd_step`, `pgd_iters`
    - `ridge_design_matrix`: stacked matrix `[H; sqrt(beta) I]`
    - `ridge_rhs_zeros`: zero tail for augmented least squares target
    - `lag_comp_gain`: delay compensation gain
- `prev_commands: np.ndarray | None`, shape `(n_act,)`
  - Previously applied DM command (can be used for delay-aware compensation).
- `max_voltage: float`
  - Per-channel command bound.

### Output

- `dm_commands: np.ndarray`, shape `(n_act,)`
  - Must satisfy all validity rules:
    - correct shape
    - no NaN/Inf
    - every entry within `[-max_voltage, max_voltage]`

## 4. Verification Scenario (v3_delay_and_model_mismatch)

`verification/evaluate.py` generates a dynamic AO benchmark with realistic difficulties:

1. Time-correlated low-order turbulence-like modes.
2. Additional small high-order perturbations.
3. Delayed and noisy slope measurements.
4. Actuator lag (`ACTUATOR_LAG`) so commanded value is not instantly applied.
5. Plant mismatch: true DM gain differs from nominal reconstruction model.

This means robust constrained control is more important than one-step algebraic fit.

## 5. Metrics and Score (0 to 1, Higher is Better)

Primary leaderboard field:
- `score_0_to_1_higher_is_better` in `[0, 1]`.
- `score_percent = 100 * score_0_to_1_higher_is_better`.

Raw metrics in `metrics.json`:
- `mean_rms`: average residual RMS wavefront error (lower better)
- `worst_rms`: worst residual RMS (lower better)
- `mean_strehl`: imaging quality proxy (higher better)
- `mean_saturation_ratio`: fraction of channels at voltage limit (lower better)

Score is a weighted utility aggregation:
- `0.20 * U(mean_rms)`
- `0.10 * U(worst_rms)`
- `0.15 * U(mean_strehl)`
- `0.55 * U(mean_saturation_ratio)`

Utility anchors (from evaluator):
- lower-better:
  - `mean_rms`: good `1.35`, bad `2.10`
  - `worst_rms`: good `2.10`, bad `3.10`
  - `mean_saturation_ratio`: good `0.02`, bad `0.35`
- higher-better:
  - `mean_strehl`: good `0.24`, bad `0.08`

`raw_cost_lower_is_better` is kept for diagnostics only; optimize leaderboard score.

## 6. Baseline Implementation

Current baseline in `baseline/controller.py`:
1. Compute linear command: `u = reconstructor @ slopes`
2. Enforce box constraint by hard clipping

Why baseline is weak:
- It does not solve a constrained objective explicitly.
- It ignores delayed sensing / model mismatch structure.
- It tends to produce heavy saturation under this benchmark.

## 7. Oracle / Reference Implementation

Reference in `verification/reference_controller.py` uses SciPy:
- solver: `scipy.optimize.lsq_linear`
- objective form (augmented ridge LS):
  - minimize `||H u - s||^2 + beta ||u||^2`
  - subject to `u_i in [-Vmax, Vmax]`
- also applies delay compensation term using `prev_commands` and `h_matrix`

Why this is a strong comparator:
- It uses a mature third-party bounded least-squares solver.
- It directly handles constraints instead of clip-after-solve.

## 8. Verification Outputs: What They Mean

After running:

```bash
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

you get files in `verification/outputs/`:

- `metrics.json`
  - Machine-readable summary for baseline(candidate) and reference.
  - Includes score, raw metrics, benchmark profile, score anchors/weights.
  - Main file for leaderboard ingestion and regression checks.
- `metrics_comparison.png`
  - Bar chart comparing baseline vs reference on key metrics.
  - Useful for quick sanity check of improvement direction.
- `example_visualization.png`
  - Visual comparison on a representative case:
    - input phase map
    - residual phase after correction
    - log10 PSF image
  - Helps interpret whether improved score matches physically better correction.

## 9. Dependency and Policy

- Baseline is expected to stay lightweight (`numpy` + provided matrices).
- Reference is allowed to use third-party SciPy.
- Thread-count tuning is not a valid optimization strategy for this task.
