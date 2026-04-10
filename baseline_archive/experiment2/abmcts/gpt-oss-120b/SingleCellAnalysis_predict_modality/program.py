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
    """Download required files and return their paths.

    Returns:
        test_mod1_path: Path to test_mod1.h5ad (RNA input for test cells).
        train_mod1_path: Path to train_mod1.h5ad (RNA input for training cells).
        train_mod2_path: Path to train_mod2.h5ad (protein targets for training cells).
    """
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


def _extract_normalized_layer(adata: ad.AnnData) -> np.ndarray:
    """Return the normalized matrix as a dense ``np.ndarray``."""
    if "normalized" in adata.layers:
        mat = adata.layers["normalized"]
    else:
        # Fallback to .X if normalized layer is missing
        mat = adata.X
    if isinstance(mat, np.ndarray):
        return mat
    # Assume a scipy sparse matrix
    return mat.toarray()


def run_mean_per_gene(*, dataset_dir: Path, output: Path) -> None:
    """Predict protein expression for test cells using ridge regression on training data.

    The method learns a linear mapping from RNA (mod1) to protein (mod2) on the
    training set and applies it to the test RNA data.  A small L2 regularization
    term improves stability without adding heavy dependencies.
    """
    test_mod1_path, train_mod1_path, train_mod2_path = _ensure_inputs(dataset_dir)

    # Load AnnData objects
    test_mod1 = ad.read_h5ad(str(test_mod1_path))
    train_mod1 = ad.read_h5ad(str(train_mod1_path))
    train_mod2 = ad.read_h5ad(str(train_mod2_path))

    # Extract normalized matrices
    X_train = _extract_normalized_layer(train_mod1)  # shape (n_train_cells, n_rna_features)
    Y_train = _extract_normalized_layer(train_mod2)  # shape (n_train_cells, n_protein_features)
    X_test = _extract_normalized_layer(test_mod1)    # shape (n_test_cells, n_rna_features)

    # Center the data (remove column means) to improve numerical stability
    X_mean = X_train.mean(axis=0, keepdims=True)
    Y_mean = Y_train.mean(axis=0, keepdims=True)

    Xc = X_train - X_mean
    Yc = Y_train - Y_mean

    # Ridge regression closed‑form solution: W = (XᵀX + λI)⁻¹ Xᵀ Y
    # λ is a small regularization constant.
    lam = 1.0
    XtX = Xc.T @ Xc
    n_features = XtX.shape[0]
    regularized = XtX + lam * np.eye(n_features)

    # Solve for each protein column simultaneously
    # Using np.linalg.solve for stability
    W = np.linalg.solve(regularized, Xc.T @ Yc)  # shape (n_rna_features, n_protein_features)

    # Predict on test set and add back the target mean
    pred_dense = (X_test - X_mean) @ W + Y_mean  # shape (n_test_cells, n_protein_features)

    # Ensure non‑negative predictions (protein counts cannot be negative)
    pred_dense = np.clip(pred_dense, a_min=0.0, a_max=None)

    # Convert to sparse CSC matrix for storage efficiency
    prediction = csc_matrix(pred_dense)

    out = ad.AnnData(
        layers={"normalized": prediction},
        shape=prediction.shape,
        obs=test_mod1.obs,
        var=train_mod2.var,
        uns={
            "dataset_id": test_mod1.uns.get("dataset_id", DATASET_ID),
            "method_id": "ridge_regression",
        },
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
