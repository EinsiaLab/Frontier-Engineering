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
    """
    Download the three AnnData files required for a linear‑regression baseline:
    * test_mod1.h5ad – source modality for which we must predict the target
    * train_mod1.h5ad – source modality of the training set
    * train_mod2.h5ad – target modality of the training set (ground truth)
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


def run_mean_per_gene(*, dataset_dir: Path, output: Path) -> None:
    """
    Linear‑regression baseline:
    * Learn B from train_mod1 (X) → train_mod2 (Y) via ordinary least squares.
    * Predict Ŷ for the test cells using X_test @ B.
    This remains extremely fast while giving a much richer predictor than a constant mean.
    """
    test_mod1_path, train_mod1_path, train_mod2_path = _ensure_inputs(dataset_dir)

    # Load AnnData objects
    test_mod1 = ad.read_h5ad(str(test_mod1_path))
    train_mod1 = ad.read_h5ad(str(train_mod1_path))
    train_mod2 = ad.read_h5ad(str(train_mod2_path))

    # Verify required layers
    for adata, name in [(train_mod1, "train_mod1"), (train_mod2, "train_mod2")]:
        if "normalized" not in adata.layers:
            raise ValueError(f"{name}.h5ad missing layers['normalized']")

    # Convert to dense float32 matrices (they are small enough)
    # Convert possible sparse layers to dense float32 matrices.
    layer_x = train_mod1.layers["normalized"]
    if hasattr(layer_x, "toarray"):
        X = layer_x.toarray().astype(np.float32)   # (n_train_cells, n_source_features)
    else:
        X = np.array(layer_x, dtype=np.float32)

    layer_y = train_mod2.layers["normalized"]
    if hasattr(layer_y, "toarray"):
        Y = layer_y.toarray().astype(np.float32)   # (n_train_cells, n_target_features)
    else:
        Y = np.array(layer_y, dtype=np.float32)

    # ----- Center and standardise the training data ---------------------------------
    # Centering (subtract column means) improves correlation metrics.
    # Standardising (divide by column std) puts all features on a comparable scale,
    # which often yields a more stable ridge solution and can boost the combined score.
    X_mean = X.mean(axis=0, keepdims=True)          # (1, n_source_features)
    Y_mean = Y.mean(axis=0, keepdims=True)          # (1, n_target_features)

    Xc = X - X_mean                                 # centred X
    Yc = Y - Y_mean                                 # centred Y

    # Compute per‑feature standard deviations; avoid division by zero.
    X_std = Xc.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    Y_std = Yc.std(axis=0, keepdims=True)
    Y_std[Y_std == 0] = 1.0

    # Standardise
    Xs = Xc / X_std
    Ys = Yc / Y_std

    # ----- Ridge‑regression on standardised data ------------------------------------
    # A smaller regularisation strength works better on this dataset.
    # It reduces bias while keeping the solution stable.
    lam = 0.1
    XtX = Xs.T @ Xs
    reg = lam * np.eye(Xs.shape[1], dtype=Xs.dtype)
    # Solve (XsᵀXs + λI) B_std = XsᵀYs
    B_std = np.linalg.solve(XtX + reg, Xs.T @ Ys)   # (n_source_features, n_target_features)

    # ----- Predict for test cells ----------------------------------------------------
    # Convert possible sparse test layer to dense float32 matrix.
    layer_test = test_mod1.layers["normalized"]
    if hasattr(layer_test, "toarray"):
        X_test = layer_test.toarray().astype(np.float32)
    else:
        X_test = np.array(layer_test, dtype=np.float32)

    # Apply the same centering and standardisation as training,
    # then reverse the standardisation of the target and add back the target mean.
    X_testc = X_test - X_mean                       # centre
    Xs_test = X_testc / X_std                       # standardise
    pred_std = Xs_test @ B_std                      # prediction in standardised space
    # Transform back to the original scale
    pred_dense = pred_std * Y_std + Y_mean
    # Protein abundances cannot be negative – clip to zero to improve error metrics
    pred_dense = np.maximum(pred_dense, 0.0)

    # Convert to CSC sparse format expected by the benchmark
    prediction = csc_matrix(pred_dense)

    out = ad.AnnData(
        layers={"normalized": prediction},
        shape=prediction.shape,
        obs=test_mod1.obs,
        var=train_mod2.var,
        uns={
            "dataset_id": test_mod1.uns.get("dataset_id", DATASET_ID),
            "method_id": "linear_regression",
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
