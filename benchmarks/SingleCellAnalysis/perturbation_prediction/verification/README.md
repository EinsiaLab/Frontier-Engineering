# Perturbation Prediction — Verification

This folder provides a lightweight scorer that reproduces the key OpenProblems perturbation-prediction metrics on the
public `neurips-2023-data` dataset hosted on `openproblems-data` (S3).

## Setup

Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r benchmarks/SingleCellAnalysis/perturbation_prediction/verification/requirements-perturbation_prediction.txt
```

## Generate a baseline prediction

```bash
python benchmarks/SingleCellAnalysis/perturbation_prediction/baseline/run_mean_across_compounds.py \
  --output prediction.h5ad
```

## Score a prediction

```bash
python benchmarks/SingleCellAnalysis/perturbation_prediction/verification/evaluate_perturbation_prediction.py \
  --prediction prediction.h5ad
```

## Data download / cache

The first run will download OpenProblems files into:

`benchmarks/SingleCellAnalysis/perturbation_prediction/resources_cache/neurips-2023-data/`

Approximate sizes:

- `de_train.h5ad`: ~175 MB
- `de_test.h5ad`: ~105 MB
- `id_map.csv`: ~4 KB

