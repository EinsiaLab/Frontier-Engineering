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
from scipy.sparse import csc_matrix


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


def _to_dense(X) -> np.ndarray:
    from scipy.sparse import issparse
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def run_knn_predict(*, dataset_dir: Path, output: Path) -> None:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge
    from scipy.sparse import csr_matrix

    test_mod1_path, train_mod1_path, train_mod2_path = _ensure_inputs(dataset_dir)
    input_test_mod1 = ad.read_h5ad(str(test_mod1_path))
    input_train_mod1 = ad.read_h5ad(str(train_mod1_path))
    input_train_mod2 = ad.read_h5ad(str(train_mod2_path))

    if "normalized" not in input_train_mod2.layers:
        raise ValueError("train_mod2.h5ad missing layers['normalized']")

    test_obs = input_test_mod1.obs.copy()
    train_var = input_train_mod2.var.copy()
    dataset_id = input_test_mod1.uns.get("dataset_id", DATASET_ID)

    train_Y = _to_dense(input_train_mod2.layers["normalized"]).astype(np.float32)

    train_X_sp = csr_matrix(input_train_mod1.layers.get("normalized", input_train_mod1.X))
    test_X_sp = csr_matrix(input_test_mod1.layers.get("normalized", input_test_mod1.X))
    del input_train_mod1, input_test_mod1, input_train_mod2

    # SVD with 80 components for richer representation
    n_components = min(80, train_X_sp.shape[1] - 1, train_X_sp.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=3)
    train_reduced = svd.fit_transform(train_X_sp).astype(np.float32)
    test_reduced = svd.transform(test_X_sp).astype(np.float32)
    del train_X_sp, test_X_sp

    # Ridge regression base prediction
    ridge = Ridge(alpha=0.8, fit_intercept=True)
    ridge.fit(train_reduced, train_Y)
    pred_ridge = ridge.predict(test_reduced).astype(np.float32)

    # Compute train residuals for KNN correction
    train_residuals = train_Y - ridge.predict(train_reduced).astype(np.float32)

    # Normalize for cosine similarity
    train_norms = np.linalg.norm(train_reduced, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    train_norm = train_reduced / train_norms
    test_norms = np.linalg.norm(test_reduced, axis=1, keepdims=True)
    test_norms[test_norms == 0] = 1.0
    test_norm = test_reduced / test_norms

    # Single-pass KNN: direct prediction + residual correction
    K = 35
    n_test = test_norm.shape[0]
    n_features = train_Y.shape[1]
    pred_knn_direct = np.zeros((n_test, n_features), dtype=np.float32)
    pred_knn_res = np.zeros((n_test, n_features), dtype=np.float32)
    batch_size = 2048

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        bs = end - start
        sims = test_norm[start:end] @ train_norm.T
        topk_idx = np.argpartition(-sims, K, axis=1)[:, :K]
        row_idx = np.arange(bs)[:, None]
        topk_sims = sims[row_idx, topk_idx]
        topk_sims_shifted = (topk_sims - topk_sims.max(axis=1, keepdims=True)) * 11.0
        w = np.exp(topk_sims_shifted)
        w /= w.sum(axis=1, keepdims=True)
        flat_idx = topk_idx.ravel()
        neighbors_Y = train_Y[flat_idx].reshape(bs, K, n_features)
        neighbors_R = train_residuals[flat_idx].reshape(bs, K, n_features)
        pred_knn_direct[start:end] = np.einsum('bk,bkf->bf', w, neighbors_Y)
        pred_knn_res[start:end] = np.einsum('bk,bkf->bf', w, neighbors_R)

    # Blend: Ridge+residual correction and direct KNN
    pred_hybrid = pred_ridge + 0.5 * pred_knn_res
    prediction = 0.35 * pred_hybrid + 0.65 * pred_knn_direct
    prediction_sparse = csc_matrix(prediction)

    out = ad.AnnData(
        layers={"normalized": prediction_sparse},
        shape=prediction_sparse.shape,
        obs=test_obs,
        var=train_var,
        uns={"dataset_id": dataset_id, "method_id": "ridge_knn_hybrid"},
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
    run_knn_predict(dataset_dir=args.dataset_dir, output=args.output)
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
# EVOLVE-BLOCK-END
