"""Evaluator for PMD Simulation task."""

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
FIBER_LENGTH_KM = 100.0
PMD_COEFFICIENT = 0.5
DGD_THRESHOLD = 30.0
TARGET_STD = 0.1
MAX_SAMPLES = 50_000
BATCH_SIZE = 5_000
MIN_OUTAGES = 20
REPEATS = 3
NUM_SEGMENTS = 100

EPSILON = 2.0  # Increased tolerance for initial submissions
INVALID_SCORE_SCALE = 0.1
INVALID_SCORE_CAP = 0.1
STD_TOL = 1e-9
OUTAGE_PROB_REL_TOL = 1e-6
OUTAGE_PROB_ABS_TOL = 1e-12
INTEGER_TOL = 1e-6
LOG_WEIGHT_CLIP = 100.0
# Reference value calibrated from the shipped baseline under evaluator-owned
# sampling and the frozen PMD smoke-test constants.
R0_DEV = 2.3e-8
R0_LOG_DEV = float(math.log(R0_DEV))
T0_DEV = 10.0


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
        from benchmarks.CommunicationEngineering.PMDSimulation.runtime.sampler import SamplerBase
        return SamplerBase
    except ModuleNotFoundError:
        from runtime.sampler import SamplerBase
        return SamplerBase


def _import_fiber_model(repo_root: Path):
    _ensure_import_paths(repo_root)
    try:
        from benchmarks.CommunicationEngineering.PMDSimulation.runtime.fiber_model import PMDFiberModel
        return PMDFiberModel
    except ModuleNotFoundError:
        from runtime.fiber_model import PMDFiberModel
        return PMDFiberModel


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
    task_root = repo_root / "benchmarks" / "CommunicationEngineering" / "PMDSimulation"
    return (task_root / raw).resolve()


def _normalize_result(result: Any) -> tuple[float, float, float, float, float, float]:
    if isinstance(result, dict):
        return (
            float(result["outages_log"]),
            float(result["weights_log"]),
            float(result.get("outage_prob", np.nan)),
            float(result.get("total_samples", np.nan)),
            float(result.get("actual_std", np.nan)),
            1.0 if bool(result.get("converged", False)) else 0.0,
        )
    if isinstance(result, (tuple, list)) and len(result) >= 6:
        return (
            float(result[0]), float(result[1]), float(result[2]),
            float(result[3]), float(result[4]),
            1.0 if bool(result[5]) else 0.0,
        )
    raise ValueError("simulate_variance_controlled 返回值格式不支持")


def _validate_result(payload: tuple[float, float, float, float, float, float]) -> dict[str, float | bool]:
    outages_log, weights_log, outage_prob, total_samples, actual_std, converged = payload

    if not np.isfinite(weights_log):
        raise ValueError("weights_log 必须是有限值")
    if np.isnan(outages_log) or outages_log == float("inf"):
        raise ValueError("outages_log 必须是有限值或 -inf")
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

    if outages_log == float("-inf"):
        if not np.isfinite(outage_prob) or not math.isclose(outage_prob, 0.0, abs_tol=OUTAGE_PROB_ABS_TOL):
            raise ValueError("outages_log=-inf 时 outage_prob 必须为 0")
        if converged_value:
            raise ValueError("未观测到 outage 时不应标记 converged=True")
        derived_outage_prob = 0.0
        outage_prob_log = -20.0
    else:
        if not np.isfinite(outages_log):
            raise ValueError("outages_log 必须是有限值或 -inf")
        if not np.isfinite(outage_prob) or outage_prob < 0.0 or outage_prob > 1.0 + OUTAGE_PROB_REL_TOL:
            raise ValueError("outage_prob 必须位于 [0, 1]")
        log_ratio = outages_log - weights_log
        if log_ratio > math.log1p(OUTAGE_PROB_REL_TOL):
            raise ValueError("outages_log 对应的 outage 权重不能超过总权重")
        derived_outage_prob = float(math.exp(log_ratio))
        if not math.isclose(
            outage_prob,
            derived_outage_prob,
            rel_tol=OUTAGE_PROB_REL_TOL,
            abs_tol=OUTAGE_PROB_ABS_TOL,
        ):
            raise ValueError("outage_prob 与 outages_log/weights_log 推导出的概率不一致")
        outage_prob_log = float(log_ratio)

    return {
        "outages_log": outages_log,
        "weights_log": weights_log,
        "outage_prob": derived_outage_prob,
        "total_samples": float(rounded_samples),
        "actual_std": actual_std,
        "converged": converged_value,
        "outage_prob_log": outage_prob_log,
    }


def _as_1d_float_array(value: Any, *, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} shape must be ({expected_len},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_beta_batch(value: Any, *, expected_segments: int, requested_batch: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    expected_shape_tail = (expected_segments, 3)
    if arr.ndim != 3 or arr.shape[1:] != expected_shape_tail:
        raise ValueError(f"beta_vectors shape must be (batch, {expected_segments}, 3), got {arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[0] > requested_batch:
        raise ValueError(f"beta batch size must be in [1, {requested_batch}], got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("beta_vectors must contain only finite values")
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
        prob = event_sum / total_weight
        prob_log = float(math.log(max(prob, OUTAGE_PROB_ABS_TOL)))
        contribution_arr = np.asarray(contributions, dtype=np.float64)
        actual_std = float(np.std(contribution_arr / (total_weight / total_samples)) / math.sqrt(total_samples))
        converged = bool(len(event_weights) >= min_events and actual_std <= TARGET_STD + STD_TOL)
    else:
        prob = 0.0
        prob_log = -20.0
        actual_std = float("inf")
        converged = False

    return {
        "prob": prob,
        "prob_log": prob_log,
        "total_samples": float(total_samples),
        "actual_std": actual_std,
        "converged": converged,
        "event_count": float(len(event_weights)),
    }


def _run_evaluator_owned_simulation(sampler: Any, fiber: Any) -> dict[str, float | bool]:
    total_weight = 0.0
    total_samples = 0
    event_weights: list[float] = []
    contributions: list[float] = []

    while total_samples < MAX_SAMPLES:
        requested_batch = min(BATCH_SIZE, MAX_SAMPLES - total_samples)
        try:
            beta_vectors, log_pdf_biased = sampler.sample(
                num_segments=fiber.num_segments,
                batch_size=requested_batch,
            )
        except Exception as e:
            raise RuntimeError(f"sample 执行失败: {e}") from e

        beta_vectors = _as_beta_batch(
            beta_vectors,
            expected_segments=fiber.num_segments,
            requested_batch=requested_batch,
        )
        batch_size_actual = int(beta_vectors.shape[0])
        log_pdf_biased = _as_1d_float_array(log_pdf_biased, name="log_pdf_biased", expected_len=batch_size_actual)

        log_pdf_true = np.sum(
            -0.5 * np.sum(beta_vectors**2, axis=2) - 1.5 * np.log(2 * np.pi),
            axis=1,
        )
        if not np.all(np.isfinite(log_pdf_true)):
            raise ValueError("true log pdf contains non-finite values")

        log_weights = np.clip(log_pdf_true - log_pdf_biased, -LOG_WEIGHT_CLIP, LOG_WEIGHT_CLIP)
        weights = np.exp(log_weights)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("importance weights must be finite and non-negative")

        dgd = fiber.evolve_pmd(beta_vectors)
        if dgd.shape != (batch_size_actual,) or not np.all(np.isfinite(dgd)):
            raise ValueError("DGD values must be finite with shape (batch,)")

        for i in range(batch_size_actual):
            is_outage = bool(dgd[i] > DGD_THRESHOLD)
            weight = float(weights[i])
            total_weight += weight
            contributions.append(weight if is_outage else 0.0)
            if is_outage:
                event_weights.append(weight)

        total_samples += batch_size_actual
        if len(event_weights) >= MIN_OUTAGES:
            interim = _summarize_weighted_event_run(
                event_weights=event_weights,
                total_weight=total_weight,
                total_samples=total_samples,
                contributions=contributions,
                min_events=MIN_OUTAGES,
            )
            if bool(interim["converged"]):
                break

    return _summarize_weighted_event_run(
        event_weights=event_weights,
        total_weight=total_weight,
        total_samples=total_samples,
        contributions=contributions,
        min_events=MIN_OUTAGES,
    )


def _build_fiber(repo_root: Path):
    PMDFiberModel = _import_fiber_model(repo_root)
    return PMDFiberModel(
        length_km=FIBER_LENGTH_KM,
        pmd_coefficient=PMD_COEFFICIENT,
        num_segments=NUM_SEGMENTS,
    )


def evaluate(program_path: str, *, repo_root: Path | None = None):
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    program = _resolve_program_path(program_path, repo_root)
    
    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "runtime_s": 0.0,
        "outage_log_ratio": float("inf"),
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
        
        if not hasattr(module, "PMDSampler"):
            raise AttributeError("提交程序中未找到类 PMDSampler")
        
        cls = module.PMDSampler
        if not isinstance(cls, type) or not issubclass(cls, SamplerBase):
            raise TypeError("PMDSampler 必须继承 SamplerBase")
        
        runtimes: list[float] = []
        outage_logs: list[float] = []
        probs: list[float] = []
        samples: list[float] = []
        stds: list[float] = []
        converged_flags: list[float] = []
        
        for rep in range(REPEATS):
            fiber = _build_fiber(repo_root)
            try:
                sampler = cls(fiber_model=fiber, seed=rep)
            except Exception as e:
                raise RuntimeError(f"PMDSampler 初始化失败: {e}") from e
            
            if not hasattr(sampler, "sample"):
                raise AttributeError("PMDSampler 缺少 sample 方法")
            
            t0 = time.time()
            result = _run_evaluator_owned_simulation(sampler, fiber)
            dt = time.time() - t0
            
            outage_prob_log = float(result["prob_log"])
            runtimes.append(float(dt))
            outage_logs.append(outage_prob_log)
            probs.append(float(result["prob"]))
            samples.append(float(result["total_samples"]))
            stds.append(float(result["actual_std"]))
            converged_flags.append(1.0 if bool(result["converged"]) else 0.0)
        
        runtime_median = float(np.median(runtimes))
        outage_log_median = float(np.median(outage_logs))
        outage_log_ratio = float(abs(outage_log_median - R0_LOG_DEV))
        
        variance_ok = float(np.nanmedian(stds) <= TARGET_STD + STD_TOL)
        convergence_ok = float(np.mean(converged_flags) >= 0.5)
        valid = float(outage_log_ratio < EPSILON and variance_ok and convergence_ok)
        raw_score = float(T0_DEV / (runtime_median * outage_log_ratio + 1e-6))
        if valid > 0:
            score = raw_score
        else:
            score = min(raw_score * INVALID_SCORE_SCALE, INVALID_SCORE_CAP)
        
        metrics.update({
            "combined_score": score,
            "runtime_s": runtime_median,
            "outage_log_ratio": outage_log_ratio,
            "valid": valid,
            "timeout": 0.0,
            "outage_prob_log_median": outage_log_median,
            "outage_prob_median": float(np.nanmedian(probs)),
            "actual_samples_median": float(np.nanmedian(samples)),
            "actual_std_median": float(np.nanmedian(stds)),
            "converged_rate": float(np.mean(converged_flags)),
            "variance_ok": variance_ok,
            "convergence_ok": convergence_ok,
            "dgd_threshold": DGD_THRESHOLD,
        })
        artifacts["validity_reason"] = (
            "ok" if valid > 0 else f"anchor_ok={outage_log_ratio < EPSILON},variance_ok={bool(variance_ok)},convergence_ok={bool(convergence_ok)}"
        )
        artifacts["dev_constants"] = json.dumps({
            "fiber_length_km": FIBER_LENGTH_KM,
            "pmd_coefficient": PMD_COEFFICIENT,
            "dgd_threshold": DGD_THRESHOLD,
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
    except (AttributeError, TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError, KeyError) as e:
        metrics["combined_score"] = 0.0
        metrics["valid"] = 0.0
        artifacts["error_message"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
    finally:
        metrics["runtime_s_total"] = float(time.time() - start)
    
    return _wrap(metrics, artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PMD Simulation submission.")
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
