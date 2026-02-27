"""Evaluator for Rayleigh Fading BER estimation task."""

from __future__ import annotations

import json
import math
import argparse
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

EPSILON = 1.0
INVALID_SCORE_SCALE = 0.1
INVALID_SCORE_CAP = 0.1
R0_DEV = 1e-6
R0_LOG_DEV = float(math.log(R0_DEV))
T0_DEV = 10.0


def _is_repo_root(path: Path) -> bool:
    return (path / "benchmarks").is_dir() and (path / "frontier_eval").is_dir()


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if _is_repo_root(parent):
            return parent
    return Path.cwd().resolve()


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


def _normalize_result(result: Any) -> tuple[float, float, float, float, float, float]:
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
            float(result[0]), float(result[1]), float(result[2]),
            float(result[3]), float(result[4]),
            1.0 if bool(result[5]) else 0.0,
        )
    raise ValueError("simulate_variance_controlled 返回值格式不支持")


def _build_channel(repo_root: Path):
    import sys
    sys.path.insert(0, str(repo_root))
    from benchmarks.CommunicationEngineering.RayleighFadingBER.runtime.channel_model import RayleighFadingChannel
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
        import sys
        sys.path.insert(0, str(repo_root))
        from benchmarks.CommunicationEngineering.RayleighFadingBER.runtime.sampler import SamplerBase
        
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
        
        for rep in range(REPEATS):
            channel = _build_channel(repo_root)
            try:
                sampler = cls(channel_model=channel, seed=rep)
            except Exception as e:
                raise RuntimeError(f"DeepFadeSampler 初始化失败: {e}") from e
            
            if not hasattr(sampler, "simulate_variance_controlled"):
                raise AttributeError("DeepFadeSampler 缺少 simulate_variance_controlled 方法")
            
            t0 = time.time()
            try:
                result = sampler.simulate_variance_controlled(
                    channel_model=channel,
                    diversity_type=DIVERSITY_TYPE,
                    modulation=MODULATION,
                    snr_db=SNR_DB,
                    target_std=TARGET_STD,
                    max_samples=MAX_SAMPLES,
                    batch_size=BATCH_SIZE,
                    min_errors=MIN_ERRORS,
                )
            except Exception as e:
                raise RuntimeError(f"simulate_variance_controlled 执行失败: {e}") from e
            dt = time.time() - t0
            
            errors_log, weights_log, err_ratio, total_samples, actual_std, converged = _normalize_result(result)
            err_rate_log = float(errors_log - weights_log)
            
            if not np.isfinite(err_rate_log):
                raise ValueError("err_rate_log 非有限值")
            
            runtimes.append(float(dt))
            err_logs.append(err_rate_log)
            ratios.append(err_ratio)
            samples.append(total_samples)
            stds.append(actual_std)
            converged_flags.append(converged)
        
        runtime_median = float(np.median(runtimes))
        err_log_median = float(np.median(err_logs))
        err_log_ratio = float(abs(err_log_median - R0_LOG_DEV))
        
        valid = float(err_log_ratio < EPSILON)
        raw_score = float(T0_DEV / (runtime_median * err_log_ratio + 1e-6))
        if valid > 0:
            score = raw_score
        else:
            score = min(raw_score * INVALID_SCORE_SCALE, INVALID_SCORE_CAP)
        
        metrics.update({
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
            "snr_db": SNR_DB,
        })
        artifacts["dev_constants"] = json.dumps({
            "snr_db": SNR_DB,
            "target_std": TARGET_STD,
            "max_samples": MAX_SAMPLES,
            "batch_size": BATCH_SIZE,
            "epsilon": EPSILON,
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
    parser = argparse.ArgumentParser(description="Evaluate Rayleigh Fading BER submission.")
    parser.add_argument("program", help="Path to candidate program file")
    parser.add_argument("--repo-root", dest="repo_root", default=None)
    args = parser.parse_args()
    
    repo_root = None if args.repo_root is None else Path(args.repo_root).expanduser().resolve()
    result = evaluate(args.program, repo_root=repo_root)
    if isinstance(result, dict):
        metrics = result
    else:
        metrics = result.metrics
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

