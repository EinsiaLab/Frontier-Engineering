#!/usr/bin/env python3
# EVOLVE-BLOCK-START
"""Run TTT denoising on PBMC and Tabula Muris, then save outputs."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import scprep

from openproblems.tasks.denoising.datasets.pbmc import pbmc
from openproblems.tasks.denoising.datasets.tabula_muris_senis import tabula_muris_senis_lung_random

from denoise_ttt import magic_denoise


DATASET_LOADERS = {
    "pbmc": pbmc,
    "tabula": tabula_muris_senis_lung_random,
}


def run_ttt(loader, verbose=False):
    adata = loader(test=False)
    train = scprep.utils.toarray(adata.obsm["train"])
    test = scprep.utils.toarray(adata.obsm["test"])

    denoised = np.asarray(magic_denoise(train, verbose=verbose))
    if denoised.shape != train.shape:
        raise ValueError(f"Unexpected denoised shape: {denoised.shape}, expected: {train.shape}")

    return train, test, denoised


def save_results(output_path, results):
    arrays = {}
    for dataset_name, (train, test, denoised) in results.items():
        arrays[f"{dataset_name}_train"] = train
        arrays[f"{dataset_name}_test"] = test
        arrays[f"{dataset_name}_denoised"] = denoised

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ttt_results.npz"),
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs in TTT denoiser.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}

    for name, loader in DATASET_LOADERS.items():
        train, test, denoised = run_ttt(loader, verbose=args.verbose)
        results[name] = (train, test, denoised)
        print(f"[TTT] {name}: cells={train.shape[0]}, genes={train.shape[1]}")

    save_results(args.output, results)
    print(f"Saved TTT outputs to: {args.output}")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
