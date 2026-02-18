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
import pandas as pd
from scipy import sparse


DATASET_ID = "neurips-2023-data"
BASE_URL = (
    "https://openproblems-data.s3.amazonaws.com/"
    "resources/task_perturbation_prediction/datasets/neurips-2023-data/"
)


def _repo_root(start: Path) -> Path:
    env_root = os.environ.get("FRONTIER_ENGINEERING_ROOT", "")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "benchmarks").is_dir():
            return candidate

    here = start.resolve()
    if here.is_file():
        here = here.parent
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
        "de_train": dataset_dir / "de_train.h5ad",
        "id_map": dataset_dir / "id_map.csv",
    }
    for _, path in files.items():
        if path.is_file():
            continue
        _download(BASE_URL + path.name, path)
    return files


def _as_matrix(x):
    if sparse.issparse(x):
        return x
    return np.asarray(x)


def run_mean_across_compounds(*, dataset_dir: Path, layer: str, output: Path) -> None:
    paths = _ensure_dataset(dataset_dir)
    de_train = ad.read_h5ad(paths["de_train"])
    id_map = pd.read_csv(paths["id_map"])

    if layer not in de_train.layers:
        raise ValueError(f"Missing layer '{layer}' in de_train.h5ad")

    X = _as_matrix(de_train.layers[layer])
    sm_name = de_train.obs["sm_name"].astype(str).to_numpy()

    gene_names = de_train.var_names.astype(str).to_numpy()

    # Mean across all cell types for each compound (OpenProblems control method).
    unique = pd.unique(sm_name)
    mean_by_compound: dict[str, np.ndarray] = {}
    for comp in unique.tolist():
        mask = sm_name == comp
        if sparse.issparse(X):
            mean_vec = np.asarray(X[mask].mean(axis=0)).ravel()
        else:
            mean_vec = np.asarray(X[mask].mean(axis=0)).ravel()
        mean_by_compound[str(comp)] = mean_vec.astype(np.float32, copy=False)

    comps = id_map["sm_name"].astype(str).tolist()
    pred = np.stack([mean_by_compound[c] for c in comps], axis=0)

    out = ad.AnnData(
        layers={"prediction": pred},
        obs=pd.DataFrame(index=id_map["id"].astype(str)),
        var=pd.DataFrame(index=gene_names),
        uns={"dataset_id": str(de_train.uns.get("dataset_id", DATASET_ID)), "method_id": "mean_across_compounds"},
    )
    out.write_h5ad(str(output), compression="gzip")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("prediction.h5ad"))
    p.add_argument("--layer", type=str, default="clipped_sign_log10_pval")
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
            / "perturbation_prediction"
            / "resources_cache"
            / DATASET_ID
        )
    run_mean_across_compounds(dataset_dir=args.dataset_dir, layer=args.layer, output=args.output)
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
