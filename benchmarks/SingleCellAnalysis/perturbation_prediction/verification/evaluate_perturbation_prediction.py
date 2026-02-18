from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


DATASET_ID = "neurips-2023-data"
BASE_URL = (
    "https://openproblems-data.s3.amazonaws.com/"
    "resources/task_perturbation_prediction/datasets/neurips-2023-data/"
)


def _repo_root(start: Path) -> Path:
    here = start.resolve()
    for parent in [here, *here.parents]:
        if (parent / "benchmarks").is_dir():
            return parent
    return here


def _download(url: str, dest: Path, *, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
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


def _ensure_dataset(dataset_dir: Path) -> dict[str, Path]:
    files = {
        "de_test": dataset_dir / "de_test.h5ad",
        "id_map": dataset_dir / "id_map.csv",
    }
    for key, path in files.items():
        if path.is_file():
            continue
        _download(BASE_URL + path.name, path)
    return files


def _as_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _rowwise_pearson(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t = truth.astype(np.float64, copy=False)
    p = pred.astype(np.float64, copy=False)
    t = np.nan_to_num(t, copy=False)
    p = np.nan_to_num(p, copy=False)

    t_mean = t.mean(axis=1, keepdims=True)
    p_mean = p.mean(axis=1, keepdims=True)
    tc = t - t_mean
    pc = p - p_mean
    num = np.sum(tc * pc, axis=1)
    den = np.linalg.norm(tc, axis=1) * np.linalg.norm(pc, axis=1)
    out = np.where(den > 0, num / den, 0.0)
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float64, copy=False)


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.nan_to_num(x, copy=False)
    y = np.nan_to_num(y, copy=False)
    xc = x - x.mean()
    yc = y - y.mean()
    den = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if den <= 0:
        return 0.0
    v = float(np.dot(xc, yc) / den)
    if not math.isfinite(v):
        return 0.0
    return v


def _rowwise_spearman(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t = truth.astype(np.float64, copy=False)
    p = pred.astype(np.float64, copy=False)
    t = np.nan_to_num(t, copy=False)
    p = np.nan_to_num(p, copy=False)

    out = np.zeros(t.shape[0], dtype=np.float64)
    for i in range(t.shape[0]):
        rt = rankdata(t[i], method="average")
        rp = rankdata(p[i], method="average")
        out[i] = _pearson_1d(rt, rp)
    return out


def _rowwise_cosine(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t = truth.astype(np.float64, copy=False)
    p = pred.astype(np.float64, copy=False)
    t = np.nan_to_num(t, copy=False)
    p = np.nan_to_num(p, copy=False)

    num = np.sum(t * p, axis=1)
    den = np.linalg.norm(t, axis=1) * np.linalg.norm(p, axis=1)
    out = np.where(den > 0, num / den, 0.0)
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float64, copy=False)


def evaluate(
    prediction_path: str,
    *,
    dataset_dir: Path,
    truth_layer: str = "clipped_sign_log10_pval",
    pred_layer: str = "prediction",
) -> Any:
    start = time.time()
    paths = _ensure_dataset(dataset_dir)

    de_test = ad.read_h5ad(paths["de_test"])
    id_map = pd.read_csv(paths["id_map"])
    pred = ad.read_h5ad(prediction_path)

    if de_test.uns.get("dataset_id") and pred.uns.get("dataset_id"):
        if str(de_test.uns["dataset_id"]) != str(pred.uns["dataset_id"]):
            raise ValueError("Prediction and de_test have differing dataset_ids")

    expected_ids = id_map["id"].astype(str).tolist()
    if list(map(str, pred.obs_names.tolist())) != expected_ids:
        # allow reordering, but require exact set
        if set(map(str, pred.obs_names.tolist())) != set(expected_ids):
            raise ValueError("Prediction obs_names must match id_map.csv id column.")
        pred = pred[expected_ids]

    if pred_layer not in pred.layers:
        raise ValueError(f"Missing prediction layer '{pred_layer}' in {prediction_path}")
    if truth_layer not in de_test.layers:
        raise ValueError(f"Missing truth layer '{truth_layer}' in de_test.h5ad")

    # Align genes (default: de_test genes)
    de_genes = de_test.var_names.astype(str)
    pr_genes = pred.var_names.astype(str)
    if not np.array_equal(de_genes, pr_genes):
        common = np.intersect1d(de_genes, pr_genes)
        if common.size < 1000:
            raise ValueError(
                f"Too few common genes between de_test ({len(de_genes)}) and prediction ({len(pr_genes)})."
            )
        de_test = de_test[:, common]
        pred = pred[:, common]

    truth_x = _as_dense(de_test.layers[truth_layer])
    pred_x = _as_dense(pred.layers[pred_layer])

    if truth_x.shape != pred_x.shape:
        raise ValueError(f"Shape mismatch: truth {truth_x.shape} vs pred {pred_x.shape}")

    pred_x = np.nan_to_num(pred_x, copy=False)

    diff = truth_x.astype(np.float64) - pred_x.astype(np.float64)
    row_rmse = np.sqrt(np.mean(diff * diff, axis=1))
    row_mae = np.mean(np.abs(diff), axis=1)

    row_pearson = _rowwise_pearson(truth_x, pred_x)
    row_spearman = _rowwise_spearman(truth_x, pred_x)
    row_cosine = _rowwise_cosine(truth_x, pred_x)

    mean_rmse = float(np.mean(row_rmse))
    mean_mae = float(np.mean(row_mae))
    mean_pearson = float(np.mean(row_pearson))
    mean_spearman = float(np.mean(row_spearman))
    mean_cosine = float(np.mean(row_cosine))

    corr_score = (mean_pearson + 1.0) / 2.0
    err_score = 1.0 / (1.0 + mean_rmse)
    combined = float((corr_score + err_score) / 2.0)

    metrics = {
        "combined_score": combined,
        "mean_rowwise_rmse": mean_rmse,
        "mean_rowwise_mae": mean_mae,
        "mean_rowwise_pearson": mean_pearson,
        "mean_rowwise_spearman": mean_spearman,
        "mean_rowwise_cosine": mean_cosine,
        "n_test": float(truth_x.shape[0]),
        "n_genes": float(truth_x.shape[1]),
        "runtime_s": float(time.time() - start),
        "dataset_id": str(de_test.uns.get("dataset_id", DATASET_ID)),
        "method_id": str(pred.uns.get("method_id", "")),
        "valid": 1.0,
    }

    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts={})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", type=Path, required=True, help="Path to prediction.h5ad")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded OpenProblems files (default: <benchmark>/resources_cache).",
    )
    p.add_argument("--truth-layer", type=str, default="clipped_sign_log10_pval")
    p.add_argument("--pred-layer", type=str, default="prediction")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo_root = _repo_root(Path(__file__).resolve())
    if args.dataset_dir is None:
        args.dataset_dir = (
            repo_root
            / "benchmarks"
            / "SingleCellAnalysis"
            / "perturbation_prediction"
            / "resources_cache"
            / DATASET_ID
        )
    result = evaluate(
        str(args.prediction),
        dataset_dir=args.dataset_dir,
        truth_layer=args.truth_layer,
        pred_layer=args.pred_layer,
    )
    try:
        metrics = result.metrics  # type: ignore[attr-defined]
    except Exception:
        metrics = result
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
