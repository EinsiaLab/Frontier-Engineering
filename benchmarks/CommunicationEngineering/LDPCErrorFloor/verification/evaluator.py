"""Evaluator for LDPC Error Floor estimation task."""

from __future__ import annotations

import json
import math
import argparse
import os
import runpy
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.random import Generator, Philox

# Frozen evaluation constants
DEV_SIGMA = 0.6
TARGET_STD = 0.1
MAX_SAMPLES = 50
BATCH_SIZE = 50
MIN_ERRORS = 20
REPEATS = 1

EPSILON = 2.0  # Increased tolerance for initial submissions
INVALID_SCORE_SCALE = 0.1
INVALID_SCORE_CAP = 0.1
STD_TOL = 1e-9
ERR_RATIO_REL_TOL = 1e-6
ERR_RATIO_ABS_TOL = 1e-12
INTEGER_TOL = 1e-6
LOG_RATIO_TOL = 0.5
LOG_WEIGHT_CLIP = 100.0
# Reference values calibrated from the shipped baseline under evaluator-owned
# sampling. The randomly constructed short LDPC instance is intentionally tiny
# for smoke evaluation, so this anchor reflects the frozen benchmark constants
# rather than a production-code error-floor estimate.
R0_DEV = 0.89
R0_LOG_DEV = float(math.log(R0_DEV))
T0_DEV = 10.0  # Reference runtime


def _is_repo_root(path: Path) -> bool:
    return (path / "benchmarks").is_dir() and (path / "frontier_eval").is_dir()


def _find_repo_root() -> Path:
    env_root = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if _is_repo_root(parent):
            return parent
    return Path.cwd().resolve()


def _task_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_paths(repo_root: Path) -> None:
    import sys

    for p in (repo_root, _task_root()):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


def _import_sampler_base(repo_root: Path):
    _ensure_import_paths(repo_root)
    try:
        from benchmarks.CommunicationEngineering.LDPCErrorFloor.runtime.sampler import SamplerBase
        return SamplerBase
    except ModuleNotFoundError:
        from runtime.sampler import SamplerBase
        return SamplerBase


def _import_ldpc_code(repo_root: Path):
    _ensure_import_paths(repo_root)
    try:
        from benchmarks.CommunicationEngineering.LDPCErrorFloor.runtime.ldpc_code import LDPCCode
        return LDPCCode
    except ModuleNotFoundError:
        from runtime.ldpc_code import LDPCCode
        return LDPCCode


def _wrap(metrics: dict[str, float], artifacts: dict[str, str | bytes]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except ModuleNotFoundError:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _load_program_module(program_path: Path):
    if not program_path.is_file():
        raise RuntimeError(f"无法加载程序文件: {program_path}")
    namespace = runpy.run_path(str(program_path), run_name="candidate_program")
    return SimpleNamespace(**namespace)


def _resolve_program_path(program_path: str, repo_root: Path) -> Path:
    """Resolve candidate program path robustly."""
    raw = Path(program_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    
    cwd_path = (Path.cwd() / raw).resolve()
    if cwd_path.is_file():
        return cwd_path
    
    task_root = (
        repo_root
        / "benchmarks"
        / "CommunicationEngineering"
        / "LDPCErrorFloor"
    )
    task_path = (task_root / raw).resolve()
    return task_path


def _normalize_result(result: Any) -> tuple[float, float, float, float, float, float]:
    """Normalize output to: errors_log, weights_log, err_ratio, total_samples, actual_std, converged(0/1)"""
    if isinstance(result, dict):
        return (
            float(result["errors_log"]),
            float(result["weights_log"]),
            float(result.get("err_ratio", np.nan)),
            float(result.get("total_samples", np.nan)),
            float(result.get("actual_std", np.nan)),
            1.0 if bool(result.get("converged", False)) else 0.0,
        )
    
    if isinstance(result, (tuple, list)) and len(result) >= 6:
        return (
            float(result[0]),
            float(result[1]),
            float(result[2]),
            float(result[3]),
            float(result[4]),
            1.0 if bool(result[5]) else 0.0,
        )
    
    raise ValueError("simulate_variance_controlled 返回值格式不支持")


def _validate_result(payload: tuple[float, float, float, float, float, float]) -> dict[str, float | bool]:
    errors_log, weights_log, err_ratio, total_samples, actual_std, converged = payload

    if not np.isfinite(weights_log):
        raise ValueError("weights_log 必须是有限值")
    if np.isnan(errors_log) or errors_log == float("inf"):
        raise ValueError("errors_log 必须是有限值或 -inf")
    if not np.isfinite(total_samples) or total_samples <= 0:
        raise ValueError("total_samples 必须是正数")
    rounded_samples = int(round(total_samples))
    if abs(total_samples - rounded_samples) > INTEGER_TOL:
        raise ValueError("total_samples 必须是整数")
    if rounded_samples > MAX_SAMPLES:
        raise ValueError(f"total_samples={rounded_samples} 超过 max_samples={MAX_SAMPLES}")
    if np.isnan(actual_std) or actual_std < 0.0:
        raise ValueError("actual_std 必须是非负数或 inf")

    converged_value = bool(converged)
    if converged_value and (not np.isfinite(actual_std) or actual_std > TARGET_STD + STD_TOL):
        raise ValueError("converged=True 但 actual_std 未达到 target_std")

    if errors_log == float("-inf"):
        if not np.isfinite(err_ratio) or not math.isclose(err_ratio, 0.0, abs_tol=ERR_RATIO_ABS_TOL):
            raise ValueError("errors_log=-inf 时 err_ratio 必须为 0")
        if converged_value:
            raise ValueError("未观测到错误时不应标记 converged=True")
        derived_err_ratio = 0.0
        err_rate_log = -20.0
    else:
        if not np.isfinite(errors_log):
            raise ValueError("errors_log 必须是有限值或 -inf")
        if not np.isfinite(err_ratio) or err_ratio < 0.0 or err_ratio > 1.0 + ERR_RATIO_REL_TOL:
            raise ValueError("err_ratio 必须位于 [0, 1]")
        log_ratio = errors_log - weights_log
        if log_ratio > math.log1p(ERR_RATIO_REL_TOL):
            raise ValueError("errors_log 对应的误差权重不能超过总权重")
        derived_err_ratio = float(math.exp(log_ratio))
        err_rate_log = float(log_ratio)

    return {
        "errors_log": errors_log,
        "weights_log": weights_log,
        # Keep the candidate-reported ratio for diagnostics, but use the
        # log-domain reconstruction as the authoritative metric. In practice
        # some samplers expose `err_ratio` as a numerically smoothed helper
        # statistic instead of an exact exp(errors_log - weights_log).
        "err_ratio": float(err_ratio if np.isfinite(err_ratio) else derived_err_ratio),
        "derived_err_ratio": derived_err_ratio,
        "total_samples": float(rounded_samples),
        "actual_std": actual_std,
        "converged": converged_value,
        "err_rate_log": err_rate_log,
    }


def _as_1d_float_array(value: Any, *, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} shape must be ({expected_len},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_noise_batch(value: Any, *, expected_n: int, requested_batch: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != expected_n:
        raise ValueError(f"noise batch shape must be (batch, {expected_n}), got {arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[0] > requested_batch:
        raise ValueError(f"noise batch size must be in [1, {requested_batch}], got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("noise batch must contain only finite values")
    return arr


def _summarize_weighted_event_run(
    *,
    event_weights: list[float],
    total_weight: float,
    total_samples: int,
    contributions: list[float],
    min_events: int,
) -> dict[str, float | bool]:
    if total_samples <= 0 or not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("evaluator-owned simulation produced no positive total weight")

    if event_weights:
        event_sum = float(np.sum(event_weights))
        ratio = event_sum / total_weight
        ratio_log = float(math.log(max(ratio, ERR_RATIO_ABS_TOL)))
        event_weights_arr = np.asarray(event_weights, dtype=np.float64)
        contribution_arr = np.asarray(contributions, dtype=np.float64)
        # Standard error of the weighted event contribution normalized by total weight.
        actual_std = float(np.std(contribution_arr / (total_weight / total_samples)) / math.sqrt(total_samples))
        converged = bool(len(event_weights_arr) >= min_events and actual_std <= TARGET_STD + STD_TOL)
    else:
        ratio = 0.0
        ratio_log = -20.0
        actual_std = float("inf")
        converged = False

    return {
        "ratio": ratio,
        "ratio_log": ratio_log,
        "total_samples": float(total_samples),
        "actual_std": actual_std,
        "converged": converged,
        "event_count": float(len(event_weights)),
    }


def _run_evaluator_owned_simulation(sampler: Any, code: Any) -> dict[str, float | bool]:
    tx_bits = np.zeros(code.n, dtype=int)
    tx_signal = np.ones(code.n)
    total_weight = 0.0
    total_samples = 0
    event_weights: list[float] = []
    contributions: list[float] = []

    while total_samples < MAX_SAMPLES:
        requested_batch = min(BATCH_SIZE, MAX_SAMPLES - total_samples)
        try:
            noise, log_pdf_biased = sampler.sample(DEV_SIGMA, tx_bits, requested_batch)
        except Exception as e:
            raise RuntimeError(f"sample 执行失败: {e}") from e

        noise = _as_noise_batch(noise, expected_n=code.n, requested_batch=requested_batch)
        batch_size_actual = int(noise.shape[0])
        log_pdf_biased = _as_1d_float_array(log_pdf_biased, name="log_pdf_biased", expected_len=batch_size_actual)

        log_pdf_true = (
            -np.sum(noise**2, axis=1) / (2 * DEV_SIGMA**2)
            - code.n / 2 * np.log(2 * np.pi * DEV_SIGMA**2)
        )
        if not np.all(np.isfinite(log_pdf_true)):
            raise ValueError("true log pdf contains non-finite values")

        log_weights = np.clip(log_pdf_true - log_pdf_biased, -LOG_WEIGHT_CLIP, LOG_WEIGHT_CLIP)
        weights = np.exp(log_weights)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("importance weights must be finite and non-negative")

        for i in range(batch_size_actual):
            received = tx_signal + noise[i, :]
            llr = 2.0 * received / (DEV_SIGMA**2)
            decoded, _ = code.decode(llr)
            is_error = not np.array_equal(decoded, tx_bits)
            weight = float(weights[i])
            total_weight += weight
            contributions.append(weight if is_error else 0.0)
            if is_error:
                event_weights.append(weight)

        total_samples += batch_size_actual
        if len(event_weights) >= MIN_ERRORS:
            interim = _summarize_weighted_event_run(
                event_weights=event_weights,
                total_weight=total_weight,
                total_samples=total_samples,
                contributions=contributions,
                min_events=MIN_ERRORS,
            )
            if bool(interim["converged"]):
                break

    return _summarize_weighted_event_run(
        event_weights=event_weights,
        total_weight=total_weight,
        total_samples=total_samples,
        contributions=contributions,
        min_events=MIN_ERRORS,
    )


def _build_code(repo_root: Path, seed: int):
    LDPCCode = _import_ldpc_code(repo_root)
    
    # Create regular (3,6) LDPC code, length 1008
    code = LDPCCode.create_regular_ldpc(n=1008, dv=3, dc=6, seed=seed)
    code.rng = Generator(Philox(seed))
    return code


def evaluate(program_path: str, *, repo_root: Path | None = None):
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    program = _resolve_program_path(program_path, repo_root)
    
    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "runtime_s": 0.0,
        "error_log_ratio": float("inf"),
        "valid": 0.0,
        "timeout": 0.0,
    }
    artifacts: dict[str, str | bytes] = {}
    
    try:
        SamplerBase = _import_sampler_base(repo_root)
        
        try:
            module = _load_program_module(program)
        except Exception as e:
            raise RuntimeError(f"加载选手程序失败: {e}") from e
        
        if not hasattr(module, "TrappingSetSampler"):
            raise AttributeError("提交程序中未找到类 TrappingSetSampler")
        
        cls = module.TrappingSetSampler
        if not isinstance(cls, type) or not issubclass(cls, SamplerBase):
            raise TypeError("TrappingSetSampler 必须继承 SamplerBase")
        
        runtimes: list[float] = []
        err_logs: list[float] = []
        ratios: list[float] = []
        samples: list[float] = []
        stds: list[float] = []
        converged_flags: list[float] = []
        
        for rep in range(REPEATS):
            seed = rep
            code = _build_code(repo_root, seed=seed)
            try:
                sampler = cls(code=code, seed=seed)
            except Exception as e:
                raise RuntimeError(f"TrappingSetSampler 初始化失败: {e}") from e
            if hasattr(sampler, "rng"):
                sampler.rng = Generator(Philox(seed))
            
            if not hasattr(sampler, "sample"):
                raise AttributeError("TrappingSetSampler 缺少 sample 方法")
            
            t0 = time.time()
            result = _run_evaluator_owned_simulation(sampler, code)
            dt = time.time() - t0
            
            err_rate_log = float(result["ratio_log"])
            runtimes.append(float(dt))
            err_logs.append(err_rate_log)
            ratios.append(float(result["ratio"]))
            samples.append(float(result["total_samples"]))
            stds.append(float(result["actual_std"]))
            converged_flags.append(1.0 if bool(result["converged"]) else 0.0)
        
        runtime_median = float(np.median(runtimes))
        err_log_median = float(np.median(err_logs))
        err_log_ratio = float(abs(err_log_median - R0_LOG_DEV))
        
        variance_ok = float(np.nanmedian(stds) <= TARGET_STD + STD_TOL)
        convergence_ok = float(np.mean(converged_flags) >= 0.5)
        valid = float(err_log_ratio < EPSILON and variance_ok and convergence_ok)
        raw_score = float(T0_DEV / (runtime_median * err_log_ratio + 1e-6))
        if valid > 0:
            score = raw_score
        else:
            score = min(raw_score * INVALID_SCORE_SCALE, INVALID_SCORE_CAP)
        
        metrics.update(
            {
                "combined_score": score,
                "runtime_s": runtime_median,
                "error_log_ratio": err_log_ratio,
                "valid": valid,
                "timeout": 0.0,
                "err_rate_log_median": err_log_median,
                "err_ratio_median": float(np.nanmedian(ratios)),
                "actual_samples_median": float(np.nanmedian(samples)),
                "actual_std_median": float(np.nanmedian(stds)),
                "converged_rate": float(np.mean(converged_flags)),
                "variance_ok": variance_ok,
                "convergence_ok": convergence_ok,
                "sigma": DEV_SIGMA,
            }
        )
        artifacts["validity_reason"] = (
            "ok" if valid > 0 else f"anchor_ok={err_log_ratio < EPSILON},variance_ok={bool(variance_ok)},convergence_ok={bool(convergence_ok)}"
        )
        artifacts["dev_constants"] = json.dumps(
            {
                "sigma": DEV_SIGMA,
                "target_std": TARGET_STD,
                "max_samples": MAX_SAMPLES,
                "batch_size": BATCH_SIZE,
                "epsilon": EPSILON,
                "std_tol": STD_TOL,
                "log_ratio_tol": LOG_RATIO_TOL,
                "log_weight_clip": LOG_WEIGHT_CLIP,
                "simulation_owner": "evaluator",
                "r0_dev": R0_DEV,
                "t0_dev": T0_DEV,
                "repeats": REPEATS,
            },
            ensure_ascii=False,
            indent=2,
        )
        artifacts["per_repeat"] = json.dumps(
            {
                "runtime_s": runtimes,
                "err_rate_log": err_logs,
                "err_ratio": ratios,
                "actual_samples": samples,
                "actual_std": stds,
                "converged": converged_flags,
            },
            ensure_ascii=False,
            indent=2,
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        ImportError,
        ModuleNotFoundError,
        KeyError,
    ) as e:
        metrics["combined_score"] = 0.0
        metrics["valid"] = 0.0
        artifacts["error_message"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
    finally:
        metrics["runtime_s_total"] = float(time.time() - start)
    
    return _wrap(metrics, artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LDPC Error Floor submission.")
    parser.add_argument("program", help="Path to candidate program file, e.g. scripts/init.py")
    parser.add_argument("--repo-root", dest="repo_root", default=None, help="Optional repository root path.")
    parser.add_argument("--metrics-out", dest="metrics_out", default=None, help="Output metrics JSON file path.")
    args = parser.parse_args()
    
    repo_root = None if args.repo_root is None else Path(args.repo_root).expanduser().resolve()
    result = evaluate(args.program, repo_root=repo_root)
    if isinstance(result, dict):
        metrics = result
    else:
        metrics = result.metrics
    
    # Output to file if specified, otherwise stdout
    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.metrics_out:
        with open(args.metrics_out, 'w', encoding='utf-8') as f:
            f.write(metrics_json)
    else:
        print(metrics_json)


if __name__ == "__main__":
    main()
