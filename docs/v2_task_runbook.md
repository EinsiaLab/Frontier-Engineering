# V2 Task-Set Runbook

This runbook documents the v2 task set as a repository-local workflow. It must be reproducible from a fresh clone of this repository and must not depend on any external personal notes or helper directories.

## Isolation rule

- Do not modify `scripts/env/setup_v1_task_envs.sh`.
- Do not modify any `scripts/env/specs/frontier-v1-*.json` spec.
- Do not modify `scripts/env/specs/frontier-eval-driver.json`.
- Add v2-only dependencies only to `.venvs/frontier-v2-*` environments.
- Use `.venvs/openff-dev` only for the repository's MolecularMechanics runtime.

Check isolation after environment work:

```bash
git diff -- scripts/env/setup_v1_task_envs.sh \
  scripts/env/specs/frontier-v1-main.json \
  scripts/env/specs/frontier-v1-summit.json \
  scripts/env/specs/frontier-eval-driver.json
```

No output is expected. This proves the repository configuration was not changed; it does not prove a local `.venvs/*` directory was never modified by hand.

## Environment mapping

| Task | Environment | Status | Notes |
|---|---|---|---|
| `MaterialEngineering/MicrowaveAbsorberDesign` | `.venvs/frontier-v2-extra` | verified | Direct baseline and unified smoke both succeeded on mainline. |
| `ParticlePhysics/MuonTomography` | `.venvs/frontier-v2-extra` | verified | Direct baseline plus evaluator succeeded; unified v2 run succeeded after using the v2 runtime. |
| `ParticlePhysics/PETScannerOptimization` | `.venvs/frontier-v2-extra` | verified | Direct baseline and unified smoke succeeded; evaluator now rejects malformed ring schemas. |
| `ParticlePhysics/ProtonTherapyPlanning` | `.venvs/frontier-v2-extra` | verified | `frontier_eval task=proton_therapy_planning algorithm.iterations=0` succeeded. |
| `SingleCellAnalysis/denoising` | none | blocked | Task README requires the external `openproblems-bio/task_denoising` repository and Docker container builds. |
| `SingleCellAnalysis/perturbation_prediction` | `.venvs/frontier-v2-extra` | verified | Baseline plus scorer succeeded after caching `de_train.h5ad`, `de_test.h5ad`, and `id_map.csv`. |
| `CommunicationEngineering/LDPCErrorFloor` | `.venvs/frontier-v2-extra` | hardened | Evaluator now owns sampling loop statistics; calibrated baseline is valid. |
| `CommunicationEngineering/PMDSimulation` | `.venvs/frontier-v2-extra` | hardened | Evaluator now owns sampling loop statistics; calibrated baseline is valid. |
| `CommunicationEngineering/RayleighFadingBER` | `.venvs/frontier-v2-extra` | hardened | Evaluator now owns sampling loop statistics; calibrated baseline is valid. |
| `ReactionOptimisation/dtlz2_pareto` | `.venvs/frontier-v2-summit-compat` | verified | Use the compat env that pins `scikit-learn < 1.3`. |
| `MolecularMechanics/weighted_parameter_coverage` | `.venvs/openff-dev` | verified | Non-uv OpenFF runtime works; unified run succeeded. |
| `MolecularMechanics/diverse_conformer_portfolio` | `.venvs/openff-dev` | verified | Non-uv OpenFF runtime works; unified run succeeded. |
| `MolecularMechanics/torsion_profile_fitting` | `.venvs/openff-dev` | verified | Non-uv OpenFF runtime works; unified run succeeded. |
| `Optics/adaptive_constrained_dm_control` | `.venvs/frontier-v2-optics` | verified | Unified v2 run succeeded. |
| `Optics/adaptive_energy_aware_control` | `.venvs/frontier-v2-optics` | verified | Unified v2 run succeeded. |
| `Optics/phase_weighted_multispot_single_plane` | `.venvs/frontier-v2-optics` | verified | Requires host `libGL.so.1` and `opencv-python`. |
| `Optics/phase_large_scale_weighted_spot_array` | `.venvs/frontier-v2-optics` | verified | Requires host `libGL.so.1` and `opencv-python`. |

## Build environments

From the repository root:

```bash
bash scripts/env/setup_v2_task_envs.sh
```

This builds:

- `.venvs/frontier-v2-extra`
- `.venvs/frontier-v2-summit`
- `.venvs/frontier-v2-summit-compat`
- `.venvs/frontier-v2-optics`

Optics tasks using `slmsuite` and OpenCV require host `libGL.so.1`. On Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

MolecularMechanics tasks are not uv-only tasks. They require the repository's OpenFF runtime:

```bash
bash scripts/bootstrap/install_openff_dev.sh
```

This path requires a working `mamba` or `conda` installation.

## Smoke commands

Use the repository-local unified helper when a task should run through `task=unified` with the v2 runtime:

```bash
bash scripts/run_v2_unified.sh MaterialEngineering/MicrowaveAbsorberDesign \
  algorithm=openevolve \
  algorithm.iterations=0
```

```bash
bash scripts/run_v2_unified.sh ParticlePhysics/MuonTomography \
  algorithm=openevolve \
  algorithm.iterations=0
```

```bash
cd benchmarks/ParticlePhysics/PETScannerOptimization
../../../.venvs/frontier-v2-extra/bin/python baseline/solution.py
../../../.venvs/frontier-v2-extra/bin/python verification/evaluator.py solution.json
```

```bash
bash scripts/run_v2_unified.sh ParticlePhysics/PETScannerOptimization \
  algorithm=openevolve \
  algorithm.iterations=0
```

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=proton_therapy_planning \
  algorithm=openevolve \
  algorithm.iterations=0
```

```bash
bash scripts/run_v2_unified.sh CommunicationEngineering/LDPCErrorFloor \
  algorithm=openevolve \
  algorithm.iterations=0 \
  algorithm.oe.evaluator.timeout=60
```

```bash
bash scripts/run_v2_unified.sh CommunicationEngineering/PMDSimulation \
  algorithm=openevolve \
  algorithm.iterations=0
```

```bash
bash scripts/run_v2_unified.sh CommunicationEngineering/RayleighFadingBER \
  algorithm=openevolve \
  algorithm.iterations=0
```

```bash
bash scripts/run_v2_unified.sh ReactionOptimisation/dtlz2_pareto \
  task.runtime.python_path=uv-env:frontier-v2-summit-compat \
  algorithm=openevolve \
  algorithm.iterations=0
```

```bash
FRONTIER_EVAL_UNIFIED_RUNTIME_ENV=frontier-v2-optics \
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=Optics/adaptive_constrained_dm_control \
  algorithm=openevolve \
  algorithm.iterations=0
```

For `perturbation_prediction`, fetch data and run the baseline/scorer:

```bash
bash scripts/data/fetch_perturbation_prediction.sh
bash scripts/run_perturbation_prediction_baseline.sh
```

The data script downloads:

| File | Size observed in validation |
|---|---:|
| `de_train.h5ad` | 183168750 bytes |
| `de_test.h5ad` | 109139040 bytes |
| `id_map.csv` | 3860 bytes |

The files are cached in:

```text
benchmarks/SingleCellAnalysis/perturbation_prediction/resources_cache/neurips-2023-data/
```

## Current results and timing ledger

The timing ledger records whether a result includes setup or dataset download. Missing exact timings must be filled by rerunning the listed commands on the target machine.

| Task | Result | Exact wall time | Evaluator `runtime_s` | Reproduction command |
|---|---:|---:|---:|---|
| `MaterialEngineering/MicrowaveAbsorberDesign` | `combined_score=0.26620516373737335`, `valid=1.0` | TODO: rerun direct shell timing if needed; unified smoke succeeded | `0.8660` from unified smoke | `bash scripts/run_v2_unified.sh MaterialEngineering/MicrowaveAbsorberDesign algorithm=openevolve algorithm.iterations=0` |
| `ParticlePhysics/MuonTomography` | `combined_score=199.32012533144325`, `valid=1.0` | TODO: rerun required | TODO: rerun required | `bash scripts/run_v2_unified.sh ParticlePhysics/MuonTomography algorithm=openevolve algorithm.iterations=0` |
| `ParticlePhysics/PETScannerOptimization` | `combined_score=598.1942761314276`, `valid=1.0` | TODO: rerun direct shell timing if needed; unified smoke succeeded | `0.7759` from unified smoke | `bash scripts/run_v2_unified.sh ParticlePhysics/PETScannerOptimization algorithm=openevolve algorithm.iterations=0` |
| `ParticlePhysics/ProtonTherapyPlanning` | `valid=1.0` | TODO: rerun required | TODO: rerun required | `.venvs/frontier-v2-extra/bin/python -m frontier_eval task=proton_therapy_planning algorithm=openevolve algorithm.iterations=0` |
| `SingleCellAnalysis/denoising` | blocked | N/A | N/A | Requires external Docker workflow. |
| `SingleCellAnalysis/perturbation_prediction` | `combined_score=0.5401216273566543`, `valid=1.0` | TODO: rerun required; exclude data download unless stated | TODO: rerun required | `bash scripts/run_perturbation_prediction_baseline.sh` |
| `CommunicationEngineering/LDPCErrorFloor` | `combined_score=173.55873302857728`, `valid=1.0` | `5.394720554351807s` direct evaluator | `5.1566126346588135s` | `bash scripts/run_v2_unified.sh CommunicationEngineering/LDPCErrorFloor algorithm=openevolve algorithm.iterations=0 algorithm.oe.evaluator.timeout=60` |
| `CommunicationEngineering/PMDSimulation` | `combined_score=14109.80093471527`, `valid=1.0` | `2.4655303955078125s` direct evaluator | `0.6930792331695557s` | `bash scripts/run_v2_unified.sh CommunicationEngineering/PMDSimulation algorithm=openevolve algorithm.iterations=0` |
| `CommunicationEngineering/RayleighFadingBER` | `combined_score=3302.3160509043173`, `valid=1.0` | `0.20431160926818848s` direct evaluator | `0.006053924560546875s` | `bash scripts/run_v2_unified.sh CommunicationEngineering/RayleighFadingBER algorithm=openevolve algorithm.iterations=0` |
| `ReactionOptimisation/dtlz2_pareto` | `combined_score=15.448643079753017`, `valid=1.0` | TODO: rerun required | TODO: rerun required | `bash scripts/run_v2_unified.sh ReactionOptimisation/dtlz2_pareto task.runtime.python_path=uv-env:frontier-v2-summit-compat algorithm=openevolve algorithm.iterations=0` |
| `MolecularMechanics/weighted_parameter_coverage` | `combined_score=9.077764`, `valid=1.0` | TODO: rerun required | TODO: rerun required | `.venvs/frontier-v2-extra/bin/python -m frontier_eval task=molecular_mechanics_weighted_parameter_coverage algorithm=openevolve algorithm.iterations=0` |
| `MolecularMechanics/diverse_conformer_portfolio` | `combined_score=278.215531`, `valid=1.0` | TODO: rerun required | TODO: rerun required | `.venvs/frontier-v2-extra/bin/python -m frontier_eval task=molecular_mechanics_diverse_conformer_portfolio algorithm=openevolve algorithm.iterations=0` |
| `MolecularMechanics/torsion_profile_fitting` | `combined_score=34.744169`, `valid=1.0` | TODO: rerun required | TODO: rerun required | `.venvs/frontier-v2-extra/bin/python -m frontier_eval task=molecular_mechanics_torsion_profile_fitting algorithm=openevolve algorithm.iterations=0` |
| `Optics/adaptive_constrained_dm_control` | `combined_score=0.20516512992698066`, `valid=1.0` | TODO: rerun required | TODO: rerun required | See Optics command above. |
| `Optics/adaptive_energy_aware_control` | `combined_score=0.18625759723077598`, `valid=1.0` | TODO: rerun required | TODO: rerun required | Replace `task.benchmark` with `Optics/adaptive_energy_aware_control`. |
| `Optics/phase_weighted_multispot_single_plane` | `combined_score=0.3726921481949858`, `valid=1.0` | TODO: rerun required | TODO: rerun required | Replace `task.benchmark` with `Optics/phase_weighted_multispot_single_plane`. |
| `Optics/phase_large_scale_weighted_spot_array` | `combined_score=24.782923596284522`, `valid=1.0` | TODO: rerun required | TODO: rerun required | Replace `task.benchmark` with `Optics/phase_large_scale_weighted_spot_array`. |

`perturbation_prediction` previously produced `combined_score=0.5722050143282681` before the scorer added `mean_rowwise_topk_sign_agreement`. The current score after that scorer change is `0.5401216273566543`.

## Code-change audit notes

- `benchmarks/MaterialEngineering/MicrowaveAbsorberDesign/*` was added directly on mainline using benchmark-local `frontier_eval/` metadata for `task=unified`. Direct baseline and unified smoke both succeeded.
- `benchmarks/ParticlePhysics/PETScannerOptimization/*` was added directly on mainline using benchmark-local `frontier_eval/` metadata for `task=unified`. The evaluator now requires exactly 20 rings with unique contiguous `ring_id` values and rejects malformed schemas outright.
- `benchmarks/ParticlePhysics/MuonTomography/frontier_eval/evaluator.py` now prefers the benchmark-local verifier before falling back to the repository verifier. This keeps copied benchmark sandboxes from depending on a full repository tree.
- `benchmarks/ParticlePhysics/MuonTomography/baseline/solution.json` only gained a trailing newline; no semantic baseline change is intended.
- `benchmarks/CommunicationEngineering/LDPCErrorFloor/verification/evaluator.py`, `benchmarks/CommunicationEngineering/PMDSimulation/verification/evaluator.py`, and `benchmarks/CommunicationEngineering/RayleighFadingBER/verification/evaluator.py` now run evaluator-owned simulations. Candidate `sample()` provides samples and biased log pdf values; the evaluator computes true log pdf, importance weights, event indicators, probabilities, variance, and convergence.
- `benchmarks/SingleCellAnalysis/perturbation_prediction/verification/evaluate_perturbation_prediction.py` added `mean_rowwise_topk_sign_agreement` and includes it in `combined_score`.
- `scripts/env/specs/frontier-v2-*` and `scripts/env/requirements/frontier-v2-*` define isolated v2 runtimes.

## Unified vs. special-case tasks

Most tasks in this v2 subset are benchmark-local `task=unified` benchmarks.

The current exceptions are:

- `ParticlePhysics/ProtonTherapyPlanning`
- `SingleCellAnalysis/perturbation_prediction`

These are still part of the v2 task set, but they currently use their own canonical reproduction paths rather than benchmark-local unified metadata.

## Evaluator hardening status

The three CommunicationEngineering rare-event evaluators are hardened against the earlier self-reported-statistics attack. A malicious candidate that self-reports the reference probability, `actual_std=0`, and `converged=True` through `simulate_variance_controlled()` is invalid because scoring no longer consumes that return value.

The remaining trusted extension point is `sample()`:

- The evaluator checks sample shapes, finite sampled values, and finite biased log pdf values.
- The evaluator computes true log pdf, importance weights, event indicators, probability estimates, variance, and convergence.
- `simulate_variance_controlled()` may remain on candidate classes for task-interface compatibility, but it is not a scoring input.

Validation smoke results for malicious self-reporting candidates:

| Task | Malicious `valid` | Notes |
|---|---:|---|
| `LDPCErrorFloor` | `0.0` | Self-reported reference ignored; evaluator-owned decoding saw a different error rate. |
| `PMDSimulation` | `0.0` | Self-reported reference ignored; evaluator-owned PMD run saw no outage convergence. |
| `RayleighFadingBER` | `0.0` | Self-reported reference ignored; evaluator-owned BER run failed anchor/validity. |

For `perturbation_prediction`, the top-k sign metric improves consistency checking but remains a statistical proxy. It does not prove deeper biological validity.
