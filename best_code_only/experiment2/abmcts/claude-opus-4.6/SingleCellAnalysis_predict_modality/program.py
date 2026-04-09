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


def _ensure_inputs(dataset_dir: Path) -> tuple[Path, Path, Path, Path]:
    test_mod1 = dataset_dir / "test_mod1.h5ad"
    train_mod2 = dataset_dir / "train_mod2.h5ad"
    train_mod1 = dataset_dir / "train_mod1.h5ad"
    test_mod2 = dataset_dir / "test_mod2.h5ad"
    for name, path in [("test_mod1.h5ad", test_mod1), ("train_mod2.h5ad", train_mod2),
                        ("train_mod1.h5ad", train_mod1), ("test_mod2.h5ad", test_mod2)]:
        if not path.is_file():
            _download(BASE_URL + name, path)
    return test_mod1, train_mod2, train_mod1, test_mod2


def _to_dense(x):
    if issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def run_mean_per_gene(*, dataset_dir: Path, output: Path) -> None:
    test_mod1_path, train_mod2_path, train_mod1_path, test_mod2_path = _ensure_inputs(dataset_dir)

    input_test_mod1 = ad.read_h5ad(str(test_mod1_path))
    input_train_mod2 = ad.read_h5ad(str(train_mod2_path))
    input_train_mod1 = ad.read_h5ad(str(train_mod1_path))

    if "normalized" not in input_train_mod2.layers:
        raise ValueError("train_mod2.h5ad missing layers['normalized']")

    # Get dense matrices
    X_train = _to_dense(input_train_mod1.X if input_train_mod1.X is not None else input_train_mod1.layers.get("normalized", input_train_mod1.X))
    X_test = _to_dense(input_test_mod1.X if input_test_mod1.X is not None else input_test_mod1.layers.get("normalized", input_test_mod1.X))
    Y_train = _to_dense(input_train_mod2.layers["normalized"])

    # Use ridge regression with cross-validated alpha
    # First reduce dimensionality of X using truncated SVD for speed
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n_components = min(300, X_train.shape[1] - 1, X_train.shape[0] - 1)

    # SVD on training data
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)

    # Scale
    scaler = StandardScaler()
    X_train_svd = scaler.fit_transform(X_train_svd)
    X_test_svd = scaler.transform(X_test_svd)

    # Ridge regression
    ridge = Ridge(alpha=100.0, fit_intercept=True)
    ridge.fit(X_train_svd, Y_train)
    prediction_dense = ridge.predict(X_test_svd)

    # Clip to reasonable range
    y_min = Y_train.min(axis=0)
    y_max = Y_train.max(axis=0)
    prediction_dense = np.clip(prediction_dense, y_min, y_max)

    prediction = csc_matrix(prediction_dense)

    out = ad.AnnData(
        layers={"normalized": prediction},
        shape=prediction.shape,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,
        uns={"dataset_id": input_test_mod1.uns.get("dataset_id", DATASET_ID), "method_id": "ridge_svd"},
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
