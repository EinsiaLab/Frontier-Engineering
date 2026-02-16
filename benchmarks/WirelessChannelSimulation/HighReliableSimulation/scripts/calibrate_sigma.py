#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator, Philox

REPO_ROOT = Path(__file__).resolve().parents[4]
TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "runtime"))

from chase import ChaseDecoder
from code_linear import HammingCode
from sampler import BesselSampler, NaiveSampler


def build_code(seed: int = 0) -> HammingCode:
    code = HammingCode(r=7, decoder="binary")
    code.rng = Generator(Philox(seed))
    code.set_decoder(ChaseDecoder(code=code, t=3))
    return code


def run_variance_controlled(
    *,
    sigma: float,
    sampler_name: str,
    target_std: float,
    max_samples: int,
    batch_size: int,
    min_errors: int,
    repeats: int,
) -> dict[str, Any]:
    runtimes: list[float] = []
    err_rate_logs: list[float] = []
    err_ratios: list[float] = []
    actual_samples: list[float] = []
    actual_stds: list[float] = []
    convergeds: list[bool] = []

    for rep in range(repeats):
        code = build_code(seed=rep)
        if sampler_name == "bessel":
            sampler = BesselSampler(code)
        elif sampler_name == "naive":
            sampler = NaiveSampler(code)
        else:
            raise ValueError(f"unsupported sampler: {sampler_name}")
        sampler.rng = Generator(Philox(rep))

        start = time.time()
        out = code.simulate_variance_controlled(
            noise_std=sigma,
            target_std=target_std,
            max_samples=max_samples,
            sampler=sampler,
            batch_size=batch_size,
            fix_tx=True,
            min_errors=min_errors,
        )
        dt = time.time() - start

        errors_log, weights_log, err_ratio, total_samples, actual_std, converged = out
        err_rate_log = float(errors_log - weights_log)

        runtimes.append(float(dt))
        err_rate_logs.append(err_rate_log)
        err_ratios.append(float(err_ratio))
        actual_samples.append(float(total_samples))
        actual_stds.append(float(actual_std))
        convergeds.append(bool(converged))

    return {
        "sampler": sampler_name,
        "sigma": float(sigma),
        "repeats": repeats,
        "runtime_median": float(np.median(runtimes)),
        "runtime_mean": float(np.mean(runtimes)),
        "err_rate_log_median": float(np.median(err_rate_logs)),
        "err_rate_log_mean": float(np.mean(err_rate_logs)),
        "ber_median": float(math.exp(float(np.median(err_rate_logs)))),
        "ber_mean": float(np.mean(np.exp(np.array(err_rate_logs)))),
        "err_ratio_median": float(np.median(err_ratios)),
        "actual_samples_median": float(np.median(actual_samples)),
        "actual_std_median": float(np.median(actual_stds)),
        "converged_rate": float(np.mean(convergeds)),
        "per_repeat": {
            "runtime_s": runtimes,
            "err_rate_log": err_rate_logs,
            "err_ratio": err_ratios,
            "actual_samples": actual_samples,
            "actual_std": actual_stds,
            "converged": convergeds,
        },
    }


def search_sigmas(
    *,
    sigma_values: list[float],
    target_ber: float,
    target_std: float,
    max_samples: int,
    batch_size: int,
    min_errors: int,
    repeats: int,
) -> dict[str, Any]:
    target_log_ber = math.log(target_ber)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_abs_gap = float("inf")

    for sigma in sigma_values:
        ref = run_variance_controlled(
            sigma=sigma,
            sampler_name="bessel",
            target_std=target_std,
            max_samples=max_samples,
            batch_size=batch_size,
            min_errors=min_errors,
            repeats=repeats,
        )
        ref_log = float(ref["err_rate_log_median"])
        gap = abs(ref_log - target_log_ber)
        row = {
            "sigma": float(sigma),
            "ref_err_rate_log_median": ref_log,
            "ref_ber_median": float(ref["ber_median"]),
            "abs_log_gap_to_target": float(gap),
            "ref_runtime_median": float(ref["runtime_median"]),
            "ref_detail": ref,
        }
        rows.append(row)
        if gap < best_abs_gap:
            best_abs_gap = gap
            best_row = row

    assert best_row is not None
    sigma_star = float(best_row["sigma"])

    # 在候选 sigma* 下再测一次 naive runtime，作为 t0 参考
    naive = run_variance_controlled(
        sigma=sigma_star,
        sampler_name="naive",
        target_std=target_std,
        max_samples=max_samples,
        batch_size=batch_size,
        min_errors=min_errors,
        repeats=repeats,
    )

    return {
        "target_ber": target_ber,
        "target_log_ber": target_log_ber,
        "search_budget": {
            "target_std": target_std,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "min_errors": min_errors,
            "repeats": repeats,
        },
        "sigma_values": sigma_values,
        "candidates": rows,
        "selected": {
            "sigma_star": sigma_star,
            "r0_log": float(best_row["ref_err_rate_log_median"]),
            "r0": float(best_row["ref_ber_median"]),
            "t0": float(naive["runtime_median"]),
            "reference_sampler": "bessel",
            "baseline_sampler": "naive",
        },
        "naive_at_sigma_star": naive,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate sigma* for HighReliableSimulation")
    p.add_argument("--target-ber", type=float, default=1e-7)
    p.add_argument("--sigmas", nargs="+", type=float, required=True)
    p.add_argument("--target-std", type=float, default=0.08)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--min-errors", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "calibration_final.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = search_sigmas(
        sigma_values=args.sigmas,
        target_ber=float(args.target_ber),
        target_std=float(args.target_std),
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        min_errors=int(args.min_errors),
        repeats=int(args.repeats),
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] calibration saved: {output}")
    print(json.dumps(result["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
