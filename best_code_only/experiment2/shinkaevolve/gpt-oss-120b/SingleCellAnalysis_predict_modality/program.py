# IMPORTANT (OpenEvolve contract):
# - The evaluator runs this script as:
#     python <program.py> --output prediction.h5ad --dataset-dir <CACHE_DIR>
# - Do NOT change these CLI flags or introduce additional REQUIRED args.
# - You MUST write a valid AnnData to --output with:
#     - layers["normalized"] shape (n_test_cells, n_mod2_features)
#     - obs matching test_mod1.obs (same cells/order)
#     - var matching train_mod2.var (same features/order)
#     - uns["dataset_id"] present and uns["method_id"] set

# EVOLVE-BLOCK-START
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix, issparse


DATASET_ID = "openproblems_neurips2021/bmmc_cite/normal/log_cp10k"
BASE_URL = (
    "https://openproblems-data.s3.amazonaws.com/"
    "resources/task_predict_modality/datasets/openproblems_neurips2021/bmmc_cite/normal/log_cp10k/"
)


def _repo_root(start: Path) -> Path:
    here = start.resolve()
    for parent in [here, *here.parents]:
        if (parent / "benchmarks").is_dir():
            return parent
    return here


def _download(url: str, dest: Path, *, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        tmp = dest.parent / f".{dest.name}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dest)
            return
        except Exception as e:  # pragma: no cover
            last_err = e
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(1.0)
    raise RuntimeError(f"Failed to download {url} -> {dest}: {last_err}")


def _ensure_inputs(dataset_dir: Path) -> tuple[Path, Path, Path]:
    test_mod1 = dataset_dir / "test_mod1.h5ad"
    train_mod1 = dataset_dir / "train_mod1.h5ad"
    train_mod2 = dataset_dir / "train_mod2.h5ad"
    if not test_mod1.is_file():
        _download(BASE_URL + "test_mod1.h5ad", test_mod1)
    if not train_mod1.is_file():
        _download(BASE_URL + "train_mod1.h5ad", train_mod1)
    if not train_mod2.is_file():
        _download(BASE_URL + "train_mod2.h5ad", train_mod2)
    return test_mod1, train_mod1, train_mod2


def _to_dense(X):
    """Convert sparse matrix to dense numpy array."""
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def _select_features_by_correlation(X_train, Y_train, n_top_genes=3000):
    """Select genes most correlated with protein expression targets.

    For each protein, compute correlation with all genes and keep genes
    that have high absolute correlation with at least one protein.
    """
    n_genes = X_train.shape[1]
    n_proteins = Y_train.shape[1]

    # Compute correlations efficiently
    # Standardize X and Y for correlation computation
    X_centered = X_train - X_train.mean(axis=0)
    Y_centered = Y_train - Y_train.mean(axis=0)

    X_std = X_centered / (X_centered.std(axis=0) + 1e-8)
    Y_std = Y_centered / (Y_centered.std(axis=0) + 1e-8)

    # Compute correlation matrix: genes x proteins
    corr_matrix = np.abs(X_std.T @ Y_std / X_train.shape[0])

    # For each gene, take max correlation across all proteins
    max_corr_per_gene = corr_matrix.max(axis=1)

    # Select top genes
    n_select = min(n_top_genes, n_genes)
    selected_indices = np.argsort(max_corr_per_gene)[-n_select:]

    return selected_indices


def _ridge_regression(X_train, Y_train, alpha=1.0):
    """Fit Ridge regression: Y = X @ W + b.

    Solves (X^T X + alpha * I) W = X^T Y for W using SVD for numerical stability.
    Returns weight matrix W and bias b.
    """
    n_samples, n_features = X_train.shape
    n_targets = Y_train.shape[1]

    # Center the data
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_centered = X_train - X_mean
    Y_centered = Y_train - Y_mean

    # Use SVD for numerical stability: X = U @ diag(S) @ Vt
    # Ridge solution: W = V @ diag(S^2/(S^2+alpha)) @ U.T @ Y
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Compute regularized inverse of singular values
    S_reg = S / (S ** 2 + alpha)

    # Compute weights: W = Vt.T @ diag(S_reg) @ U.T @ Y_centered
    W = Vt.T @ (S_reg[:, np.newaxis] * (U.T @ Y_centered))

    return W, X_mean, Y_mean


def run_mean_per_gene(*, dataset_dir: Path, output: Path) -> None:
    test_mod1_path, train_mod1_path, train_mod2_path = _ensure_inputs(dataset_dir)

    # Load all required data
    input_test_mod1 = ad.read_h5ad(str(test_mod1_path))
    input_train_mod1 = ad.read_h5ad(str(train_mod1_path))
    input_train_mod2 = ad.read_h5ad(str(train_mod2_path))

    if "normalized" not in input_train_mod2.layers:
        raise ValueError("train_mod2.h5ad missing layers['normalized']")
    if "normalized" not in input_train_mod1.layers:
        raise ValueError("train_mod1.h5ad missing layers['normalized']")
    if "normalized" not in input_test_mod1.layers:
        raise ValueError("test_mod1.h5ad missing layers['normalized']")

    # Get training and test data as dense arrays
    X_train = _to_dense(input_train_mod1.layers["normalized"])
    Y_train = _to_dense(input_train_mod2.layers["normalized"])
    X_test = _to_dense(input_test_mod1.layers["normalized"])

    # Select most informative genes based on correlation with proteins
    selected_genes = _select_features_by_correlation(X_train, Y_train, n_top_genes=3000)
    X_train_selected = X_train[:, selected_genes]
    X_test_selected = X_test[:, selected_genes]

    # Fit Ridge regression model on selected features
    W, X_mean, Y_mean = _ridge_regression(X_train_selected, Y_train, alpha=1.0)

    # Predict: Y_pred = (X_test - X_mean) @ W + Y_mean
    Y_pred = (X_test_selected - X_mean) @ W + Y_mean

    # Ensure predictions are non-negative (count-like data)
    Y_pred = np.maximum(Y_pred, 0)

    prediction = csc_matrix(Y_pred)

    out = ad.AnnData(
        layers={"normalized": prediction},
        shape=prediction.shape,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,
        uns={"dataset_id": input_test_mod1.uns.get("dataset_id", DATASET_ID), "method_id": "ridge_regression"},
    )
    out.write_h5ad(str(output), compression="gzip")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("prediction.h5ad"))
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded OpenProblems files (default: <benchmark>/resources_cache).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo_root = _repo_root(Path(__file__).resolve())
    if args.dataset_dir is None:
        args.dataset_dir = (
            repo_root
            / "benchmarks"
            / "SingleCellAnalysis"
            / "predict_modality"
            / "resources_cache"
            / "openproblems_neurips2021__bmmc_cite__normal__log_cp10k"
        )
    run_mean_per_gene(dataset_dir=args.dataset_dir, output=args.output)
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
# EVOLVE-BLOCK-END