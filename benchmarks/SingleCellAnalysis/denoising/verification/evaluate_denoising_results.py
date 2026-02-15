#!/usr/bin/env python3
"""Evaluate MAGIC/TTT run outputs with the same metrics used in the notebook."""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from openproblems.tasks.denoising.metrics.mse import mse
from openproblems.tasks.denoising.metrics.poisson import poisson


DATASET_LABELS = {
    "pbmc": "PBMC",
    "tabula": "Tabula Muris",
}


def normalize_score(score, no_denoise_score, perfect_score):
    denom = no_denoise_score - perfect_score
    if denom == 0:
        return 0.0
    return (no_denoise_score - score) / denom


def build_adata(train, test, denoised):
    n_cells, n_genes = train.shape
    adata = anndata.AnnData(X=np.zeros((n_cells, n_genes), dtype=np.float32))
    adata.obsm["train"] = train
    adata.obsm["test"] = test
    adata.obsm["denoised"] = denoised
    return adata


def load_method_outputs(path):
    data = np.load(path)
    outputs = {}
    for dataset_name in DATASET_LABELS:
        outputs[dataset_name] = {
            "train": np.asarray(data[f"{dataset_name}_train"]),
            "test": np.asarray(data[f"{dataset_name}_test"]),
            "denoised": np.asarray(data[f"{dataset_name}_denoised"]),
        }
    return outputs


def assert_same_input_data(magic_outputs, ttt_outputs):
    for dataset_name in DATASET_LABELS:
        m_train = magic_outputs[dataset_name]["train"]
        t_train = ttt_outputs[dataset_name]["train"]
        m_test = magic_outputs[dataset_name]["test"]
        t_test = ttt_outputs[dataset_name]["test"]

        if m_train.shape != t_train.shape or m_test.shape != t_test.shape:
            raise ValueError(
                f"{dataset_name}: MAGIC and TTT input shape mismatch. "
                "Please regenerate both results with the same dataset split."
            )
        if not np.allclose(m_train, t_train) or not np.allclose(m_test, t_test):
            raise ValueError(
                f"{dataset_name}: MAGIC and TTT were run on different input data. "
                "Please regenerate both results with the same dataset split."
            )


def evaluate_one_dataset(dataset_name, magic_data, ttt_data):
    train = magic_data["train"]
    test = magic_data["test"]

    adata_no = build_adata(train, test, train)
    adata_perfect = build_adata(train, test, test)
    adata_magic = build_adata(train, test, magic_data["denoised"])
    adata_ttt = build_adata(train, test, ttt_data["denoised"])

    mse_none = mse(adata_no)
    mse_perfect = mse(adata_perfect)
    poisson_none = poisson(adata_no)
    poisson_perfect = poisson(adata_perfect)

    mse_magic = mse(adata_magic)
    poisson_magic = poisson(adata_magic)
    mse_norm_magic = normalize_score(mse_magic, mse_none, mse_perfect)
    poisson_norm_magic = normalize_score(poisson_magic, poisson_none, poisson_perfect)
    mean_magic = (mse_norm_magic + poisson_norm_magic) / 2.0

    mse_ttt = mse(adata_ttt)
    poisson_ttt = poisson(adata_ttt)
    mse_norm_ttt = normalize_score(mse_ttt, mse_none, mse_perfect)
    poisson_norm_ttt = normalize_score(poisson_ttt, poisson_none, poisson_perfect)
    mean_ttt = (mse_norm_ttt + poisson_norm_ttt) / 2.0

    dataset_table = pd.DataFrame(
        {
            "Method": ["No denoising", "Perfect denoising", "MAGIC (approx, rev norm)", "TTT (MSE)"],
            "MSE (raw)": [mse_none, mse_perfect, mse_magic, mse_ttt],
            "Poisson (raw)": [poisson_none, poisson_perfect, poisson_magic, poisson_ttt],
            "MSE (norm)": [0.0, 1.0, mse_norm_magic, mse_norm_ttt],
            "Poisson (norm)": [0.0, 1.0, poisson_norm_magic, poisson_norm_ttt],
            "Mean Score": [0.0, 1.0, mean_magic, mean_ttt],
        }
    )

    summary_rows = [
        {
            "Dataset": DATASET_LABELS[dataset_name],
            "Method": "MAGIC",
            "MSE (norm)": mse_norm_magic,
            "Poisson (norm)": poisson_norm_magic,
            "Mean Score": mean_magic,
        },
        {
            "Dataset": DATASET_LABELS[dataset_name],
            "Method": "TTT",
            "MSE (norm)": mse_norm_ttt,
            "Poisson (norm)": poisson_norm_ttt,
            "Mean Score": mean_ttt,
        },
    ]
    return dataset_table, summary_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--magic-results",
        type=Path,
        default=Path("magic_results.npz"),
        help="Path to MAGIC outputs (.npz).",
    )
    parser.add_argument(
        "--ttt-results",
        type=Path,
        default=Path("ttt_results.npz"),
        help="Path to TTT outputs (.npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If provided, save CSV summaries into this directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    magic_outputs = load_method_outputs(args.magic_results)
    ttt_outputs = load_method_outputs(args.ttt_results)
    assert_same_input_data(magic_outputs, ttt_outputs)

    pbmc_table, pbmc_summary = evaluate_one_dataset("pbmc", magic_outputs["pbmc"], ttt_outputs["pbmc"])
    tabula_table, tabula_summary = evaluate_one_dataset("tabula", magic_outputs["tabula"], ttt_outputs["tabula"])

    final_results = pd.DataFrame(pbmc_summary + tabula_summary)

    print("=" * 70)
    print("PBMC RESULTS")
    print("=" * 70)
    print(pbmc_table.round(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("TABULA MURIS RESULTS")
    print("=" * 70)
    print(tabula_table.round(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("FINAL COMPARISON (higher score = better, 0=no denoising, 1=perfect)")
    print("=" * 70)
    print(final_results.round(5).to_string(index=False))

    magic_mean = final_results.loc[final_results["Method"] == "MAGIC", "Mean Score"].mean()
    ttt_mean = final_results.loc[final_results["Method"] == "TTT", "Mean Score"].mean()
    print("\n" + "=" * 70)
    print("OVERALL MEAN SCORE (averaged across datasets)")
    print("=" * 70)
    print(f"  MAGIC (approx, rev norm): {magic_mean:.4f}")
    print(f"  TTT (MSE mode):           {ttt_mean:.4f}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pbmc_table.to_csv(args.output_dir / "pbmc_results.csv", index=False)
        tabula_table.to_csv(args.output_dir / "tabula_results.csv", index=False)
        final_results.to_csv(args.output_dir / "final_results.csv", index=False)
        print(f"\nSaved CSV files to: {args.output_dir}")


if __name__ == "__main__":
    main()
