"""Evaluator for Rayleigh Fading BER estimation task."""

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
SNR_DB = 10.0
TARGET_STD = 0.1
MAX_SAMPLES = 50_000
BATCH_SIZE = 5_000
MIN_ERRORS = 20
REPEATS = 3
NUM_BRANCHES = 4
DIVERSITY_TYPE = "MRC"
MODULATION = "BPSK"

EPSILON = 2.0  # Increased tolerance for initial submissions
# Reference values (to be calibrated with baseline solution)
R0_DEV = 1e-5  # Reference BER (adjusted for initial testing)
R0_LOG_DEV = float(math.log(R0_DEV))
T0_DEV = 10.0
ERR_RATIO_REL_TOL = 1e-6
ERR_RATIO_ABS_TOL = 1e-12
INTEGER_TOL = 1e-6
STD_TOL = 1e-9
LOG_WEIGHT_CLIP = 100.0


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
        from benchmarks.CommunicationEngineering.RayleighFadingBER.runtime.sampler import SamplerBase
        return SamplerBase
    except ModuleNotFoundError:
        from runtime.sampler import SamplerBase
        return SamplerBase


def _import_channel_model(repo_root: Path):
    _ensure_import_paths(repo_root)
    try:
        from benchmarks.CommunicationEngineering.RayleighFadingBER.runtime.channel_model import RayleighFadingChannel
        return RayleighFadingChannel
    except ModuleNotFoundError:
        from runtime.channel_model import RayleighFadingChannel
        return RayleighFadingChannel


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
    raw = Path(program_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_path = (Path.cwd() / raw).resolve()
    if cwd_path.is_file():
        return cwd_path
    task_root = repo_root / "benchmarks" / "CommunicationEngineering" / "RayleighFadingBER"
    return (task_root / raw).resolve()


def _normalize_result(result: Any) -> dict[str, float | bool]:
    required_keys = (
        "errors_log",
        "weights_log",
        "err_ratio",
        "total_samples",
        "actual_std",
        "converged",
    )
    if isinstance(result, dict):
        missing = [key for key in required_keys if key not in result]
        if missing:
            raise ValueError(f"simulate_variance_controlled 缺少字段: {missing}")
        payload = result
    elif isinstance(result, (tuple, list)) and len(result) == 6:
        payload = {
            "errors_log": result[0],
            "weights_log": result[1],
            "err_ratio": result[2],
            "total_samples": result[3],
            "actual_std": result[4],
            "converged": result[5],
        }
    else:
        raise ValueError("simulate_variance_controlled 返回值格式不支持")

    converged = payload["converged"]
    if isinstance(converged, (np.bool_, bool)):
        converged_value = bool(converged)
    elif isinstance(converged, (int, float)) and converged in (0, 1):
        converged_value = bool(converged)
    else:
        raise ValueError("converged 必须是布尔值或 0/1")

    return {
        "errors_log": float(payload["errors_log"]),
        "weights_log": float(payload["weights_log"]),
        "err_ratio": float(payload["err_ratio"]),
        "total_samples": float(payload["total_samples"]),
        "actual_std": float(payload["actual_std"]),
        "converged": converged_value,
    }


def _validate_result(payload: dict[str, float | bool]) -> dict[str, float | bool]:
    errors_log = float(payload["errors_log"])
    weights_log = float(payload["weights_log"])
    err_ratio = float(payload["err_ratio"])
    total_samples = float(payload["total_samples"])
    actual_std = float(payload["actual_std"])
    converged = bool(payload["converged"])

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
    if converged and (not np.isfinite(actual_std) or actual_std > TARGET_STD + ERR_RATIO_ABS_TOL):
        raise ValueError("converged=True 但 actual_std 未达到 target_std")

    if errors_log == float("-inf"):
        if not np.isfinite(err_ratio) or not math.isclose(err_ratio, 0.0, abs_tol=ERR_RATIO_ABS_TOL):
            raise ValueError("errors_log=-inf 时 err_ratio 必须为 0")
        if converged:
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
        if not math.isclose(
            err_ratio,
            derived_err_ratio,
            rel_tol=ERR_RATIO_REL_TOL,
            abs_tol=ERR_RATIO_ABS_TOL,
        ):
            raise ValueError(
                "err_ratio 与 errors_log/weights_log 推导出的误码率不一致"
            )
        err_rate_log = float(log_ratio)

    return {
        "errors_log": errors_log,
        "weights_log": weights_log,
        "err_ratio": derived_err_ratio,
        "total_samples": float(rounded_samples),
        "actual_std": actual_std,
        "converged": converged,
        "err_rate_log": err_rate_log,
    }


def _as_1d_float_array(value: Any, *, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} shape must be ({expected_len},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_channel_batch(value: Any, *, expected_branches: int, requested_batch: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != expected_branches:
        raise ValueError(f"h_magnitudes shape must be (batch, {expected_branches}), got {arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[0] > requested_batch:
        raise ValueError(f"channel batch size must be in [1, {requested_batch}], got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("h_magnitudes must contain only finite positive values")
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
        contribution_arr = np.asarray(contributions, dtype=np.float64)
        actual_std = float(np.std(contribution_arr / (total_weight / total_samples)) / math.sqrt(total_samples))
        converged = bool(len(event_weights) >= min_events and actual_std <= TARGET_STD + STD_TOL)
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


def _run_evaluator_owned_simulation(sampler: Any, channel: Any, *, seed: int) -> dict[str, float | bool]:
    rng = Generator(Philox(seed + 10_000))
    total_weight = 0.0
    total_samples = 0
    event_weights: list[float] = []
    contributions: list[float] = []
    sigma_h = float(channel.sigma_h)

    while total_samples < MAX_SAMPLES:
        requested_batch = min(BATCH_SIZE, MAX_SAMPLES - total_samples)
        try:
            h_magnitudes, log_pdf_biased = sampler.sample(
                num_branches=channel.num_branches,
                batch_size=requested_batch,
                sigma_h=sigma_h,
            )
        except Exception as e:
            raise RuntimeError(f"sample 执行失败: {e}") from e

        h_magnitudes = _as_channel_batch(
            h_magnitudes,
            expected_branches=channel.num_branches,
            requested_batch=requested_batch,
        )
        batch_size_actual = int(h_magnitudes.shape[0])
        log_pdf_biased = _as_1d_float_array(log_pdf_biased, name="log_pdf_biased", expected_len=batch_size_actual)

        log_pdf_true = np.sum(
            -h_magnitudes**2 / (2 * sigma_h**2) - np.log(sigma_h**2) + np.log(h_magnitudes),
            axis=1,
        )
        if not np.all(np.isfinite(log_pdf_true)):
            raise ValueError("true log pdf contains non-finite values")

        log_weights = np.clip(log_pdf_true - log_pdf_biased, -LOG_WEIGHT_CLIP, LOG_WEIGHT_CLIP)
        weights = np.exp(log_weights)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("importance weights must be finite and non-negative")

        combined_snr = channel.combine_snr(h_magnitudes, DIVERSITY_TYPE, SNR_DB)
        if combined_snr.shape != (batch_size_actual,) or not np.all(np.isfinite(combined_snr)):
            raise ValueError("combined SNR values must be finite with shape (batch,)")

        ber = np.asarray(channel.compute_ber(combined_snr, MODULATION), dtype=np.float64)
        if ber.shape != (batch_size_actual,) or not np.all(np.isfinite(ber)):
            raise ValueError("BER values must be finite with shape (batch,)")
        ber = np.clip(ber, 0.0, 1.0)
        error_draws = rng.random(batch_size_actual) < ber

        for i in range(batch_size_actual):
            is_error = bool(error_draws[i])
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


def _build_channel(repo_root: Path):
    RayleighFadingChannel = _import_channel_model(repo_root)
    return RayleighFadingChannel(num_branches=NUM_BRANCHES, sigma_h=1.0)


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
        
        if not hasattr(module, "DeepFadeSampler"):
            raise AttributeError("提交程序中未找到类 DeepFadeSampler")
        
        cls = module.DeepFadeSampler
        if not isinstance(cls, type) or not issubclass(cls, SamplerBase):
            raise TypeError("DeepFadeSampler 必须继承 SamplerBase")
        
        runtimes: list[float] = []
        err_logs: list[float] = []
        ratios: list[float] = []
        samples: list[float] = []
        stds: list[float] = []
        converged_flags: list[float] = []
        repetition_diagnostics: list[dict[str, float | bool]] = []
        
        for rep in range(REPEATS):
            channel = _build_channel(repo_root)
            try:
                sampler = cls(channel_model=channel, seed=rep)
            except Exception as e:
                raise RuntimeError(f"DeepFadeSampler 初始化失败: {e}") from e
            
            if not hasattr(sampler, "sample"):
                raise AttributeError("DeepFadeSampler 缺少 sample 方法")
            
            t0 = time.time()
            result = _run_evaluator_owned_simulation(sampler, channel, seed=rep)
            dt = time.time() - t0
            
            err_rate_log = float(result["ratio_log"])
            
            runtimes.append(float(dt))
            err_logs.append(err_rate_log)
            ratios.append(float(result["ratio"]))
            samples.append(float(result["total_samples"]))
            stds.append(float(result["actual_std"]))
            converged_flags.append(1.0 if bool(result["converged"]) else 0.0)
            repetition_diagnostics.append({
                "repeat": rep,
                "runtime_s": float(dt),
                "err_ratio": float(result["ratio"]),
                "err_rate_log": err_rate_log,
                "total_samples": float(result["total_samples"]),
                "actual_std": float(result["actual_std"]),
                "converged": bool(result["converged"]),
            })
        
        runtime_median = float(np.median(runtimes))
        err_log_median = float(np.median(err_logs))
        err_log_ratio = float(abs(err_log_median - R0_LOG_DEV))
        actual_std_median = float(np.nanmedian(stds))
        converged_rate = float(np.mean(converged_flags))
        variance_ok = actual_std_median <= TARGET_STD + STD_TOL
        convergence_ok = math.isclose(converged_rate, 1.0, abs_tol=ERR_RATIO_ABS_TOL)
        
        valid = float(err_log_ratio < EPSILON and variance_ok and convergence_ok)
        raw_score = float(T0_DEV / (runtime_median * err_log_ratio + 1e-6))
        score = raw_score if valid > 0 else 0.0
        
        metrics.update({
            "combined_score": score,
            "runtime_s": runtime_median,
            "error_log_ratio": err_log_ratio,
            "valid": valid,
            "timeout": 0.0,
            "err_rate_log_median": err_log_median,
            "err_ratio_median": float(np.nanmedian(ratios)),
            "actual_samples_median": float(np.nanmedian(samples)),
            "actual_std_median": actual_std_median,
            "converged_rate": converged_rate,
            "variance_ok": 1.0 if variance_ok else 0.0,
            "convergence_ok": 1.0 if convergence_ok else 0.0,
            "snr_db": SNR_DB,
        })
        artifacts["validity_reason"] = (
            "ok" if valid > 0 else f"anchor_ok={err_log_ratio < EPSILON},variance_ok={bool(variance_ok)},convergence_ok={bool(convergence_ok)}"
        )
        artifacts["dev_constants"] = json.dumps({
            "snr_db": SNR_DB,
            "target_std": TARGET_STD,
            "max_samples": MAX_SAMPLES,
            "batch_size": BATCH_SIZE,
            "epsilon": EPSILON,
            "std_tol": STD_TOL,
            "log_weight_clip": LOG_WEIGHT_CLIP,
            "simulation_owner": "evaluator",
            "r0_dev": R0_DEV,
            "t0_dev": T0_DEV,
            "repeats": REPEATS,
        }, ensure_ascii=False, indent=2)
        artifacts["replicate_diagnostics"] = json.dumps(
            repetition_diagnostics,
            ensure_ascii=False,
            indent=2,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError, KeyError) as e:
        metrics["combined_score"] = 0.0
        metrics["valid"] = 0.0
        artifacts["error_message"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
    finally:
        metrics["runtime_s_total"] = float(time.time() - start)
    
    return _wrap(metrics, artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Rayleigh Fading BER submission.")
    parser.add_argument("program", help="Path to candidate program file")
    parser.add_argument("--repo-root", dest="repo_root", default=None)
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
