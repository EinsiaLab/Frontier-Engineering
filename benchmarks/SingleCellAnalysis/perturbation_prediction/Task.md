# Perturbation Prediction (OpenProblems / NeurIPS 2023 scPerturb)

## 1. Background
Single-cell perturbation screens measure how a chemical (small-molecule) perturbation changes gene expression in
different cell types. These experiments are expensive; accurate prediction models can accelerate hypothesis generation
and drug discovery.

This benchmark is adapted from the OpenProblems Bio perturbation prediction task repository:
`openproblems-bio/task_perturbation_prediction`, which is based on the NeurIPS 2023 Open Problems competition track
(Kaggle: *Open Problems – Single-Cell Perturbations*).

## 2. Dataset (source)
We use the OpenProblems processed dataset published to the public `openproblems-data` S3 bucket:

- dataset id: `neurips-2023-data`
- files: `de_train.h5ad`, `de_test.h5ad`, `id_map.csv`

`de_train.h5ad` and `de_test.h5ad` contain differential expression (DE) features per *(compound, cell_type)* pair across
genes, in multiple layers. The default target layer for evaluation is:

- `clipped_sign_log10_pval` (truth)

## 3. Task definition
Given training DE profiles, predict the DE profile for each test *(compound, cell_type)* pair.

### 3.1 Input
Your method can use:

- `de_train.h5ad` (training DE profiles)
- `id_map.csv` (the list/order of test *(compound, cell_type)* pairs to predict)

### 3.2 Output
Write a prediction file `prediction.h5ad` as an AnnData object:

- `layers["prediction"]`: a matrix of shape `(n_test, n_genes)` with predicted DE values
- `obs_names`: must match the `id` column of `id_map.csv` (as strings), in the same order
- `var_names`: gene names aligned to `de_test.var_names`
- `uns["dataset_id"]`: string (copied from the dataset)
- `uns["method_id"]`: string (your method name)

## 4. Evaluation
The evaluator compares `de_test.layers["clipped_sign_log10_pval"]` (truth) to `prediction.layers["prediction"]`
(prediction), row-wise across genes.

Reported metrics (matching the OpenProblems task metrics):

- `mean_rowwise_rmse`
- `mean_rowwise_mae`
- `mean_rowwise_pearson`
- `mean_rowwise_spearman`
- `mean_rowwise_cosine`

### 4.1 Combined score
We convert correlation and error into a single maximization score:

- `corr_score = (mean_rowwise_pearson + 1) / 2`
- `err_score = 1 / (1 + mean_rowwise_rmse)`
- `combined_score = (corr_score + err_score) / 2`

Higher is better. Invalid submissions receive score 0.

## 5. References
- OpenProblems Bio task repo: https://github.com/openproblems-bio/task_perturbation_prediction
- Kaggle competition: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations
- Dataset/bench context: NeurIPS 2023 Open Problems competition track.

