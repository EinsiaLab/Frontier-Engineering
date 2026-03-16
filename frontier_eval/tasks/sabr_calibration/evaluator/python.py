"""
Frontier-Eval evaluator for SABRCalibration task.
Imports candidate module, calls solve(INSTANCE), validates output,
computes RMSE vs market vols, returns EvaluationResult.
"""
from __future__ import annotations

import importlib.util
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────
# Market data (same as verification/evaluate.py)
# ──────────────────────────────────────────────
INSTANCE = {
    "F": 100.0,
    "beta": 0.5,
    "T_list": [0.25, 0.50, 1.00],
    "strikes": [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
    "market_vols": [
        [0.32159, 0.30742, 0.29428, 0.28196, 0.27046, 0.25976, 0.24980, 0.24060, 0.23201, 0.22410, 0.21686],
        [0.32141, 0.30724, 0.29410, 0.28179, 0.27029, 0.25959, 0.24965, 0.24045, 0.23186, 0.22395, 0.21671],
        [0.32076, 0.30664, 0.29352, 0.28122, 0.26973, 0.25903, 0.24911, 0.23992, 0.23133, 0.22342, 0.21619],
    ],
}

HUMAN_BEST_RMSE = 1e-3


def _sabr_vol(F: float, K: float, T: float,
              alpha: float, rho: float, nu: float,
              beta: float = 0.5) -> float:
    if T <= 0 or F <= 0 or K <= 0 or alpha <= 0:
        return 0.0
    FK = F * K
    FKb = FK ** ((1 - beta) / 2)
    if abs(F - K) < 1e-6:
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


def _compute_rmse(F, beta, T_list, strikes, market_vols,
                  alpha, rho, nu) -> float:
    sq = []
    for t_idx, T in enumerate(T_list):
        for k_idx, K in enumerate(strikes):
            mv = market_vols[t_idx][k_idx]
            pv = _sabr_vol(F, K, T, alpha, rho, nu, beta)
            sq.append((pv - mv) ** 2)
    return math.sqrt(sum(sq) / len(sq))


def evaluate(program_path: str, *, repo_root: Path | None = None) -> Any:
    start = time.time()
    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "timeout": 0.0,
        "runtime_s": 0.0,
    }
    artifacts: dict[str, str] = {}

    try:
        spec = importlib.util.spec_from_file_location("candidate", program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.solve(INSTANCE)
    except Exception as exc:
        artifacts["error_message"] = str(exc)
        artifacts["traceback"] = traceback.format_exc()[-4000:]
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    # Validate
    try:
        if not isinstance(result, (list, tuple)) or len(result) != 3:
            raise ValueError(
                f"solve() must return [alpha, rho, nu], got {type(result).__name__} len={len(result) if hasattr(result,'__len__') else '?'}"
            )
        alpha, rho, nu = float(result[0]), float(result[1]), float(result[2])
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0 (got {alpha})")
        if not (-1 < rho < 1):
            raise ValueError(f"rho must be in (-1, 1) (got {rho})")
        if nu <= 0:
            raise ValueError(f"nu must be > 0 (got {nu})")
    except Exception as exc:
        artifacts["error_message"] = str(exc)
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    # Score
    F = INSTANCE["F"]
    beta = INSTANCE["beta"]
    T_list = INSTANCE["T_list"]
    strikes = INSTANCE["strikes"]
    market_vols = INSTANCE["market_vols"]

    rmse = _compute_rmse(F, beta, T_list, strikes, market_vols, alpha, rho, nu)
    combined_score = min(1.0, HUMAN_BEST_RMSE / rmse) if rmse > 1e-15 else 1.0

    # Baseline for comparison
    atm_vol_1y = market_vols[2][strikes.index(100)]
    alpha_bl = atm_vol_1y * F ** (1 - beta)
    baseline_rmse = _compute_rmse(F, beta, T_list, strikes, market_vols, alpha_bl, 0.0, 0.3)
    baseline_score = min(1.0, HUMAN_BEST_RMSE / baseline_rmse) if baseline_rmse > 1e-15 else 1.0

    metrics.update({
        "combined_score": float(combined_score),
        "valid": 1.0,
        "rmse": float(rmse),
        "alpha": float(alpha),
        "rho": float(rho),
        "nu": float(nu),
    })
    artifacts["baseline_score"] = str(baseline_score)
    artifacts["human_best_rmse"] = str(HUMAN_BEST_RMSE)
    metrics["runtime_s"] = float(time.time() - start)
    return _wrap(metrics, artifacts)


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]) -> Any:
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
