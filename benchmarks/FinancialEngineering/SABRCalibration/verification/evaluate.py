"""
SABR Volatility Surface Calibration — Evaluator
================================================
Loads the candidate module, calls solve(INSTANCE),
validates the output, computes RMSE vs market vols,
and writes output/comparison.json.

Usage:
    python evaluate.py <candidate.py> [--output-dir <dir>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

# ──────────────────────────────────────────────
# Market data (generated from SABR with known true params)
# Source: representative SPX option vol surface structure
# True params: alpha=2.5, rho=-0.65, nu=0.40, beta=0.5
# See: Hagan et al. (2002) "Managing Smile Risk", Willmott Magazine
# ──────────────────────────────────────────────
INSTANCE = {
    "F": 100.0,           # ATM forward price (normalized)
    "beta": 0.5,          # SABR beta (fixed; 0.5 = log-normal backbone)
    "T_list": [0.25, 0.50, 1.00],   # option expiries in years
    "strikes": [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
    # Market implied vols (generated from true SABR, alpha=2.5, rho=-0.65, nu=0.40)
    # Rows = expiries [0.25, 0.50, 1.00], Cols = strikes above
    "market_vols": [
        # T=0.25
        [0.32159, 0.30742, 0.29428, 0.28196, 0.27046, 0.25976, 0.24980, 0.24060, 0.23201, 0.22410, 0.21686],
        # T=0.50
        [0.32141, 0.30724, 0.29410, 0.28179, 0.27029, 0.25959, 0.24965, 0.24045, 0.23186, 0.22395, 0.21671],
        # T=1.00
        [0.32076, 0.30664, 0.29352, 0.28122, 0.26973, 0.25903, 0.24911, 0.23992, 0.23133, 0.22342, 0.21619],
    ],
}

# Human best: properly calibrated SABR achieves RMSE < 1e-3
HUMAN_BEST_RMSE = 1e-3


def sabr_vol(F: float, K: float, T: float,
             alpha: float, rho: float, nu: float,
             beta: float = 0.5) -> float:
    """SABR implied volatility (Hagan 2002 approximation)."""
    if T <= 0 or F <= 0 or K <= 0 or alpha <= 0:
        return 0.0
    FK = F * K
    FKb = FK ** ((1 - beta) / 2)
    if abs(F - K) < 1e-6:  # ATM formula
        mid = (
            (1 - beta) ** 2 * alpha ** 2 / (24 * F ** (2 - 2 * beta))
            + rho * beta * nu * alpha / (4 * F ** (1 - beta))
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        )
        return max(1e-9, (alpha / F ** (1 - beta)) * (1 + mid * T))
    logFK = math.log(F / K)
    z = nu / alpha * FKb * logFK
    disc = 1 - 2 * rho * z + z ** 2
    if disc <= 0:
        return max(1e-9, abs(alpha / FKb))
    x_z = math.log((math.sqrt(disc) + z - rho) / (1 - rho))
    B = z / x_z if abs(x_z) > 1e-10 else 1.0
    A = alpha / (
        FKb * (1 + (1 - beta) ** 2 / 24 * logFK ** 2
               + (1 - beta) ** 4 / 1920 * logFK ** 4)
    )
    mid = (
        (1 - beta) ** 2 * alpha ** 2 / (24 * FK ** (1 - beta))
        + rho * beta * nu * alpha / (4 * FKb)
        + (2 - 3 * rho ** 2) * nu ** 2 / 24
    )
    return max(1e-9, A * B * (1 + mid * T))


def compute_rmse(F: float, beta: float, T_list: list,
                 strikes: list, market_vols: list,
                 alpha: float, rho: float, nu: float) -> float:
    """Root Mean Square Error of SABR fit vs market vols."""
    sq_errs = []
    for t_idx, T in enumerate(T_list):
        for k_idx, K in enumerate(strikes):
            mv = market_vols[t_idx][k_idx]
            pv = sabr_vol(F, K, T, alpha, rho, nu, beta)
            sq_errs.append((pv - mv) ** 2)
    return math.sqrt(sum(sq_errs) / len(sq_errs))


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(candidate_path: str, output_dir: str = "output") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Load and run candidate ──────────────────────────────
    t0 = time.perf_counter()
    try:
        mod = load_candidate(candidate_path)
        result = mod.solve(INSTANCE)
    except Exception as exc:
        result_json = {
            "valid": False,
            "error": str(exc),
            "combined_score": 0.0,
            "baseline_final_score": 0.0,
        }
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(result_json, f, indent=2)
        return result_json
    elapsed = time.perf_counter() - t0

    # ── Validate output ─────────────────────────────────────
    valid = True
    error_msg = ""

    if not isinstance(result, (list, tuple)) or len(result) != 3:
        valid = False
        error_msg = f"solve() must return [alpha, rho, nu] (got {type(result).__name__} len={len(result) if hasattr(result,'__len__') else 'N/A'})"

    if valid:
        alpha, rho, nu = float(result[0]), float(result[1]), float(result[2])
        if alpha <= 0:
            valid = False
            error_msg = f"alpha must be > 0 (got {alpha})"
        elif not (-1 < rho < 1):
            valid = False
            error_msg = f"rho must be in (-1, 1) (got {rho})"
        elif nu <= 0:
            valid = False
            error_msg = f"nu must be > 0 (got {nu})"

    if not valid:
        result_json = {
            "valid": False,
            "error": error_msg,
            "combined_score": 0.0,
            "baseline_final_score": 0.0,
        }
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(result_json, f, indent=2)
        return result_json

    # ── Compute score ────────────────────────────────────────
    F = INSTANCE["F"]
    beta = INSTANCE["beta"]
    T_list = INSTANCE["T_list"]
    strikes = INSTANCE["strikes"]
    market_vols = INSTANCE["market_vols"]

    rmse = compute_rmse(F, beta, T_list, strikes, market_vols, alpha, rho, nu)
    combined_score = min(1.0, HUMAN_BEST_RMSE / rmse) if rmse > 0 else 1.0

    # Baseline: alpha estimated from ATM vol, rho=0, nu=0.3
    atm_vol_1y = market_vols[2][strikes.index(100)]
    alpha_bl = atm_vol_1y * F ** (1 - beta)
    baseline_rmse = compute_rmse(F, beta, T_list, strikes, market_vols, alpha_bl, 0.0, 0.3)
    baseline_score = min(1.0, HUMAN_BEST_RMSE / baseline_rmse)

    result_json = {
        "valid": True,
        "alpha": alpha,
        "rho": rho,
        "nu": nu,
        "rmse": rmse,
        "combined_score": combined_score,
        "baseline_final_score": baseline_score,
        "human_best_rmse": HUMAN_BEST_RMSE,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(result_json, f, indent=2)

    print(f"alpha={alpha:.4f}, rho={rho:.4f}, nu={nu:.4f}")
    print(f"RMSE={rmse:.6f}, score={combined_score:.4f} (baseline={baseline_score:.4f})")
    return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", help="Path to candidate .py file")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    res = evaluate(args.candidate, args.output_dir)
    sys.exit(0 if res.get("valid") else 1)
