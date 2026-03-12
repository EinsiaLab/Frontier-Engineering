# Holographic H3 Specification: Multi-Wavelength Focusing/Splitting

## 0. Who this doc is for

This task is written for engineers who know optimization/code, not necessarily optics.

## 1. Engineering background (plain-language)

Different wavelengths (colors) naturally propagate differently.

In many products, one optical element must satisfy all of them simultaneously:

- each wavelength should go to a designated spatial target,
- total power split across wavelengths should match a desired ratio.

This is similar to a multi-domain optimization problem:

- domains = wavelengths,
- each domain has a spatial target,
- global coupling constraint = spectral ratio.

Applications:

- color imaging optics,
- WDM optical routing,
- chromatic compensation components.

Economic value:

- better multi-wavelength behavior improves product quality without adding extra hardware channels.

## 2. What you need to do

Improve baseline optimization in `baseline/solver.py` to get higher score under shared-hardware constraints.

Editable target:

- `baseline/solver.py`

Read-only in challenge setup:

- `verification/evaluate.py`
- `verification/reference_solver.py`

## 3. Core file/function to modify

Main function:

- `solve(spec, device=None, seed=0)` in `baseline/solver.py`

Keep return structure unchanged.

## 4. Input contract (`spec`)

Key fields:

- `wavelengths`: list of wavelengths.
- `target_centers`: one target coordinate per wavelength.
- `target_spectral_ratios`: desired power ratio among wavelengths.
- optical geometry: `shape`, `spacing`, `layer_z`, `output_z`, `waist_radius`.
- `roi_radius_m`: ROI size for energy measurement.

Evaluator-injected constants:

- `score_eff_target`, `score_spectral_scale`,
- `valid_*` thresholds,
- reference comparison margins.

## 5. Output contract (`solve`)

Must return at least:

- `system`: shared optical system.
- `input_fields`: list of input fields, one per wavelength.
- `loss_history`.
- `spec` (recommended).

Evaluator will run each wavelength through the returned system and compute metrics.

## 6. Baseline implementation (current)

Current baseline is intentionally minimal:

1. Build one shared multi-wavelength phase system.
2. For each wavelength, optimize only target-ROI efficiency.
3. Average loss across wavelengths.

Missing pieces (deliberate):

- no explicit crosstalk suppression,
- no explicit spectral-ratio matching,
- no advanced initialization.

## 7. Oracle implementation (current)

Reference is stronger and less hardware-constrained:

1. Use `slmsuite` WGS to generate seed phase map per wavelength.
2. Build two candidates:
   - direct wavelength-specific phase execution,
   - wavelength-specific phase fine-tuning.
3. Select candidate with better task score.

Because phases are wavelength-specific (not one strictly shared mask), this behaves as an upper-bound reference.

## 8. Metrics and score (0~1)

Per wavelength `i`:

- `target_efficiency_i = P_target_i / P_total_i`
- `designated_crosstalk_i = P_other_designated_i / (P_target_i + P_other_designated_i)`
- `shape_cosine_i = cosine(I_pred_norm_i, I_target_norm_i)`

Global:

- `mean_target_efficiency`
- `mean_crosstalk`
- `mean_shape_cosine`
- `spectral_ratio_mae`

Derived components:

- `efficiency_score = min(1, mean_target_efficiency / score_eff_target)`
- `isolation_score = 1 - mean_crosstalk`
- `spectral_score = exp(-spectral_ratio_mae / score_spectral_scale)`

Final score:

- `mean_score = (efficiency_score^0.45) * (isolation_score^0.25) * (spectral_score^0.20) * (mean_shape_cosine^0.10)`

Interpretation:

- high score needs both spatial correctness and spectral correctness,
- strong spectral ratio with poor spatial map (or inverse) cannot get top score.

## 9. Valid and better-than-baseline logic

Baseline is valid iff:

- `mean_target_efficiency >= valid_mean_target_efficiency_min`
- `mean_crosstalk <= valid_mean_crosstalk_max`
- `mean_score >= valid_mean_score_min`

Reference is better iff:

- `reference_mean_score >= baseline_mean_score + better_score_margin`
- `reference_mean_shape_cosine >= baseline_mean_shape_cosine + better_shape_margin`

## 10. Verification artifacts and their meaning

Outputs under `verification/artifacts/`:

- `summary.json`
  - all metrics, scores, timing, and pass/fail flags.
- `spectral_intensity_maps.png`
  - per wavelength: target vs baseline vs reference maps.
- `loss_and_spectral_ratios.png`
  - loss curves,
  - bar chart of target/baseline/reference spectral ratios.

Typical debugging signals:

- low spectral score + decent maps: need explicit ratio loss,
- high crosstalk: need stronger designated-vs-other suppression,
- low shape cosine: improve initialization/objective balance.

## 11. Run verification

```bash
PY=python3
$PY benchmarks/Optics/holographic_multispectral_focusing/verification/evaluate.py --device cpu
```

