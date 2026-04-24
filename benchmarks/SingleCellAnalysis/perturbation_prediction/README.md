# Perturbation Prediction

This benchmark is adapted from OpenProblems Bio:

- Task repo: `openproblems-bio/task_perturbation_prediction`
- Benchmark page: https://openproblems.bio/benchmarks/perturbation_prediction
- NeurIPS 2023 / Kaggle competition: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations

It uses the public OpenProblems dataset hosted on `openproblems-data` (S3) and reproduces the core evaluation metrics.

## Directory structure

- `baseline/`: simple reference methods (outputs `prediction.h5ad`)
- `verification/`: dataset downloader + scoring script
- `scripts/`: initialization helper for the v2 task set
- `Task.md`: full task specification

## Quick start

This task is part of the current v2 task set, uses `.venvs/frontier-v2-extra`, and now also supports benchmark-local `task=unified`.

Its canonical reproduction path remains:

1. download/cache the public dataset
2. generate a prediction
3. run the scorer

Fetch data:

```bash
bash scripts/data/fetch_perturbation_prediction.sh
```

Generate a baseline prediction:

```bash
.venvs/frontier-v2-extra/bin/python benchmarks/SingleCellAnalysis/perturbation_prediction/baseline/run_mean_across_compounds.py \
  --output prediction.h5ad
```

Evaluate a prediction:

```bash
.venvs/frontier-v2-extra/bin/python benchmarks/SingleCellAnalysis/perturbation_prediction/verification/evaluate_perturbation_prediction.py \
  --prediction prediction.h5ad
```

Unified smoke run:

```bash
bash scripts/run_v2_unified.sh SingleCellAnalysis/perturbation_prediction \
  algorithm=openevolve \
  algorithm.iterations=0
```
