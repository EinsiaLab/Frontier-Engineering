# Perturbation Prediction

This benchmark is adapted from OpenProblems Bio:

- Task repo: `openproblems-bio/task_perturbation_prediction`
- Benchmark page: https://openproblems.bio/benchmarks/perturbation_prediction
- NeurIPS 2023 / Kaggle competition: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations

It uses the public OpenProblems dataset hosted on `openproblems-data` (S3) and reproduces the core evaluation metrics.

## Directory structure

- `baseline/`: simple reference methods (outputs `prediction.h5ad`)
- `verification/`: dataset downloader + scoring script
- `Task.md`: full task specification

## Quick start

Generate a baseline prediction:

```bash
python benchmarks/SingleCellAnalysis/perturbation_prediction/baseline/run_mean_across_compounds.py \
  --output prediction.h5ad
```

Evaluate a prediction:

```bash
python benchmarks/SingleCellAnalysis/perturbation_prediction/verification/evaluate_perturbation_prediction.py \
  --prediction prediction.h5ad
```

