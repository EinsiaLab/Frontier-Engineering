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
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def run_mean_per_gene(*, dataset_dir: Path, output: Path) -> None:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    test_mod1_path, train_mod1_path, train_mod2_path = _ensure_inputs(dataset_dir)
    input_test_mod1 = ad.read_h5ad(str(test_mod1_path))
    input_train_mod1 = ad.read_h5ad(str(train_mod1_path))
    input_train_mod2 = ad.read_h5ad(str(train_mod2_path))

    if "normalized" not in input_train_mod2.layers:
        raise ValueError("train_mod2.h5ad missing layers['normalized']")

    # Get training data
    X_train = input_train_mod1.layers.get("normalized", input_train_mod1.X)
    Y_train = input_train_mod2.layers["normalized"]
    X_test = input_test_mod1.layers.get("normalized", input_test_mod1.X)

    # Convert to dense
    X_train = _to_dense(X_train).astype(np.float32)
    Y_train = _to_dense(Y_train).astype(np.float32)
    X_test = _to_dense(X_test).astype(np.float32)

    # Feature selection: keep genes with highest variance
    gene_var = np.var(X_train, axis=0)
    n_top_genes = min(5000, X_train.shape[1])
    top_gene_idx = np.argsort(gene_var)[-n_top_genes:]
    top_gene_idx = np.sort(top_gene_idx)
    X_train_sel = X_train[:, top_gene_idx]
    X_test_sel = X_test[:, top_gene_idx]

    # Dimensionality reduction with TruncatedSVD - more components
    n_components = min(500, X_train_sel.shape[1] - 1, X_train_sel.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train_sel)
    X_test_svd = svd.transform(X_test_sel)

    # Standardize SVD features for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_svd)
    X_test_scaled = scaler.transform(X_test_svd)

    # Ridge regression with cross-validated alpha
    alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    ridge = RidgeCV(alphas=alphas, fit_intercept=True)
    ridge.fit(X_train_svd, Y_train)
    pred_ridge = ridge.predict(X_test_svd)

    # KNN regression on scaled SVD features
    # Use fewer components for KNN to avoid curse of dimensionality
    n_knn_components = min(100, n_components)
    X_train_knn = X_train_scaled[:, :n_knn_components]
    X_test_knn = X_test_scaled[:, :n_knn_components]

    knn = KNeighborsRegressor(
        n_neighbors=50,
        weights='distance',
        metric='euclidean',
        n_jobs=-1,
    )
    knn.fit(X_train_knn, Y_train)
    pred_knn = knn.predict(X_test_knn)

    # Blend predictions: weighted average favoring Ridge (which is usually better)
    blend_weight_ridge = 0.7
    pred = blend_weight_ridge * pred_ridge + (1.0 - blend_weight_ridge) * pred_knn

    # Ensure predictions are float32
    pred = pred.astype(np.float32)

    prediction = csc_matrix(pred)

    out = ad.AnnData(
        layers={"normalized": prediction},
        shape=prediction.shape,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,
        uns={"dataset_id": input_test_mod1.uns.get("dataset_id", DATASET_ID), "method_id": "ridge_knn_ensemble"},
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